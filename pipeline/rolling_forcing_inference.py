from collections.abc import Mapping
from typing import List, Optional
import copy
import gc
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        model_kwargs = getattr(args, "model_kwargs", {}) or {}
        if not isinstance(model_kwargs, Mapping):
            raise TypeError(
                "model_kwargs must be a mapping. Check YAML indentation near "
                "`model_kwargs:`; expected entries like `  timestep_shift: 5.0`, "
                f"got {type(model_kwargs).__name__}."
            )
        self.generator = WanDiffusionWrapper(
            **model_kwargs, is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache_clean = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.tokentrim_enabled = bool(getattr(args, "tokentrim_enabled", False))
        self.tokentrim_state = None
        self.tokentrim_rate_history = []
        self.tokentrim_prev_summary = None
        self.tokentrim_prev_start_frame = None
        self.tokentrim_trigger_mode = "tokentrim"
        self.tokentrim_rollback_windows = 0
        self.tokentrim_rollback_max_attempts = 0
        self.tokentrim_rollback_experimental = False
        self.tokentrim_rollback_suppress_cache = False
        self.tokentrim_rollback_reset_rng = False
        self.tokentrim_rollback_best_of_n = 1
        self.tokentrim_rollback_depths = None
        self.tokentrim_rollback_interventions = ["none"]
        self.tokentrim_rollback_include_original = False
        self.tokentrim_rollback_selector = "drift"
        self.tokentrim_selector_subject_weight = 1.0
        self.tokentrim_selector_boundary_weight = 1.0
        self.tokentrim_selector_motion_weight = 0.5
        self.tokentrim_selector_drift_weight = 0.05
        self.tokentrim_selector_rate_weight = 1.0
        self.tokentrim_selector_context_frames = 6
        self.tokentrim_dino_model_name = "vit_small_patch14_dinov2.lvd142m"
        self.tokentrim_dino_device = "cpu"
        self.tokentrim_dino_subject_weight = 1.0
        self.tokentrim_dino_temporal_weight = 0.25
        self.tokentrim_dino_max_frames = 8
        self.tokentrim_dino_image_size = 518
        self._tokentrim_dino_model = None
        self.tokentrim_rate_history_size = 8
        self.tokentrim_rate_warmup_steps = 3
        self.tokentrim_rate_z_threshold = 2.0
        self.tokentrim_rate_top_fraction = 0.10
        self.tokentrim_checkpoint_count = 0
        self.tokentrim_checkpoint_interval = 1
        self.tokentrim_checkpoint_device = "cpu"
        self.tokentrim_last_pruned = False
        self.tokentrim_last_token_indices = None
        self.tokentrim_last_severity = None
        self.tokentrim_last_threshold = None
        self.tokentrim_last_drift = None
        self.tokentrim_last_rate_score = None
        self.tokentrim_last_rate_components = None
        self.tokentrim_last_rate_token_indices = None
        if self.tokentrim_enabled:
            try:
                from tokentrim import TokenTrimConfig, TokenTrimState
                from tokentrim import suppress_rolling_forcing_cache_tokens, wan_latents_to_token_summary
            except ImportError as exc:
                raise ImportError(
                    "tokentrim_enabled=True requires installing the TokenTrim package, "
                    "for example: pip install -e ../TokenTrim"
                ) from exc

            self._tokentrim_state_cls = TokenTrimState
            self._tokentrim_latent_summary = wan_latents_to_token_summary
            self._tokentrim_suppress_cache = suppress_rolling_forcing_cache_tokens
            self.tokentrim_config = TokenTrimConfig(
                pruning_fraction=float(getattr(args, "tokentrim_pruning_fraction", 0.10)),
                lambda_threshold=float(getattr(args, "tokentrim_lambda_threshold", 2.0)),
                warmup_steps=int(getattr(args, "tokentrim_warmup_steps", 2)),
            )
            self.tokentrim_sink_blocks = int(getattr(args, "tokentrim_sink_blocks", 1))
            self.tokentrim_max_rerolls = int(getattr(args, "tokentrim_max_rerolls", 1))
            self.tokentrim_debug = bool(getattr(args, "tokentrim_debug", False))
            self.tokentrim_patch_size = tuple(getattr(args, "tokentrim_patch_size", [2, 2]))
            self.tokentrim_trigger_mode = str(getattr(args, "tokentrim_trigger_mode", "tokentrim"))
            valid_triggers = {"tokentrim", "rate_anomaly"}
            if self.tokentrim_trigger_mode not in valid_triggers:
                raise ValueError(
                    "tokentrim_trigger_mode must be one of "
                    f"{sorted(valid_triggers)}, got {self.tokentrim_trigger_mode!r}"
                )
            self.tokentrim_rollback_windows = int(getattr(args, "tokentrim_rollback_windows", 0))
            self.tokentrim_rollback_max_attempts = int(getattr(args, "tokentrim_rollback_max_attempts", 0))
            self.tokentrim_rollback_experimental = bool(getattr(args, "tokentrim_rollback_experimental", False))
            self.tokentrim_rollback_suppress_cache = bool(getattr(args, "tokentrim_rollback_suppress_cache", False))
            self.tokentrim_rollback_reset_rng = bool(getattr(args, "tokentrim_rollback_reset_rng", False))
            self.tokentrim_rollback_best_of_n = int(getattr(args, "tokentrim_rollback_best_of_n", 1))
            rollback_depths = getattr(args, "tokentrim_rollback_depths", None)
            self.tokentrim_rollback_depths = (
                [int(depth) for depth in rollback_depths]
                if rollback_depths
                else None
            )
            rollback_interventions = getattr(args, "tokentrim_rollback_interventions", ["none"])
            self.tokentrim_rollback_interventions = [str(item) for item in rollback_interventions]
            self.tokentrim_rollback_include_original = bool(
                getattr(args, "tokentrim_rollback_include_original", False)
            )
            self.tokentrim_rollback_selector = str(
                getattr(args, "tokentrim_rollback_selector", "drift")
            )
            valid_selectors = {"drift", "latent_temporal", "rate_anomaly", "dino"}
            if self.tokentrim_rollback_selector not in valid_selectors:
                raise ValueError(
                    "tokentrim_rollback_selector must be one of "
                    f"{sorted(valid_selectors)}, got {self.tokentrim_rollback_selector!r}"
                )
            self.tokentrim_selector_subject_weight = float(
                getattr(args, "tokentrim_selector_subject_weight", 1.0)
            )
            self.tokentrim_selector_boundary_weight = float(
                getattr(args, "tokentrim_selector_boundary_weight", 1.0)
            )
            self.tokentrim_selector_motion_weight = float(
                getattr(args, "tokentrim_selector_motion_weight", 0.5)
            )
            self.tokentrim_selector_drift_weight = float(
                getattr(args, "tokentrim_selector_drift_weight", 0.05)
            )
            self.tokentrim_selector_rate_weight = float(
                getattr(args, "tokentrim_selector_rate_weight", 1.0)
            )
            self.tokentrim_selector_context_frames = max(
                1,
                int(getattr(args, "tokentrim_selector_context_frames", 6)),
            )
            self.tokentrim_dino_model_name = str(
                getattr(args, "tokentrim_dino_model_name", "vit_small_patch14_dinov2.lvd142m")
            )
            self.tokentrim_dino_device = str(getattr(args, "tokentrim_dino_device", "cpu"))
            self.tokentrim_dino_subject_weight = float(
                getattr(args, "tokentrim_dino_subject_weight", 1.0)
            )
            self.tokentrim_dino_temporal_weight = float(
                getattr(args, "tokentrim_dino_temporal_weight", 0.25)
            )
            self.tokentrim_dino_max_frames = max(
                1,
                int(getattr(args, "tokentrim_dino_max_frames", 8)),
            )
            self.tokentrim_dino_image_size = max(
                14,
                int(getattr(args, "tokentrim_dino_image_size", 518)),
            )
            self.tokentrim_rate_history_size = max(
                1,
                int(getattr(args, "tokentrim_rate_history_size", 8)),
            )
            self.tokentrim_rate_warmup_steps = max(
                1,
                int(getattr(args, "tokentrim_rate_warmup_steps", 3)),
            )
            self.tokentrim_rate_z_threshold = float(
                getattr(args, "tokentrim_rate_z_threshold", 2.0)
            )
            self.tokentrim_rate_top_fraction = float(
                getattr(args, "tokentrim_rate_top_fraction", self.tokentrim_config.pruning_fraction)
            )
            if not 0.0 < self.tokentrim_rate_top_fraction < 1.0:
                raise ValueError("tokentrim_rate_top_fraction must be in (0, 1)")
            self.tokentrim_checkpoint_count = int(getattr(args, "tokentrim_checkpoint_count", 0))
            self.tokentrim_checkpoint_interval = max(1, int(getattr(args, "tokentrim_checkpoint_interval", 1)))
            self.tokentrim_checkpoint_device = str(getattr(args, "tokentrim_checkpoint_device", "cpu"))
            if self.tokentrim_rollback_experimental and self.tokentrim_max_rerolls > 0:
                print("TokenTrim rollback disables current-window rerolls to avoid KV cache snapshots.")
                self.tokentrim_max_rerolls = 0
            self.tokentrim_state = self._tokentrim_state_cls(self.tokentrim_config)

    @staticmethod
    def _clone_kv_cache(kv_cache):
        return [
            {
                key: value.clone() if torch.is_tensor(value) else value
                for key, value in layer_cache.items()
            }
            for layer_cache in kv_cache
        ]

    @staticmethod
    def _restore_kv_cache(target, snapshot):
        for target_layer, source_layer in zip(target, snapshot):
            for key, value in source_layer.items():
                if torch.is_tensor(value):
                    target_layer[key].copy_(value)
                else:
                    target_layer[key] = value

    @staticmethod
    def _cache_to_device(cache, device):
        return [
            {
                key: value.detach().to(device=device, copy=True) if torch.is_tensor(value) else value
                for key, value in layer_cache.items()
            }
            for layer_cache in cache
        ]

    @staticmethod
    def _restore_cache_from_device(target, snapshot):
        for target_layer, source_layer in zip(target, snapshot):
            for key, value in source_layer.items():
                if torch.is_tensor(value):
                    target_layer[key].copy_(value.to(
                        device=target_layer[key].device,
                        dtype=target_layer[key].dtype,
                    ))
                else:
                    target_layer[key] = value

    def _make_tokentrim_checkpoint(
            self,
            next_window_index,
            output,
            noisy_cache,
    ):
        checkpoint_device = torch.device(self.tokentrim_checkpoint_device)
        prev_summary = self.tokentrim_prev_summary
        return {
            "next_window_index": next_window_index,
            "output": output.detach().to(device=checkpoint_device, copy=True),
            "noisy_cache": noisy_cache.detach().to(device=checkpoint_device, copy=True),
            "kv_cache": self._cache_to_device(self.kv_cache_clean, checkpoint_device),
            "crossattn_cache": self._cache_to_device(self.crossattn_cache, checkpoint_device),
            "tokentrim_state": copy.deepcopy(self.tokentrim_state),
            "rate_history": [
                drift.detach().to(device=checkpoint_device, copy=True)
                for drift in self.tokentrim_rate_history
            ],
            "prev_summary": (
                None if prev_summary is None else prev_summary.detach().to(device=checkpoint_device, copy=True)
            ),
            "prev_start_frame": self.tokentrim_prev_start_frame,
        }

    def _restore_tokentrim_checkpoint(
            self,
            checkpoint,
            output,
            noisy_cache,
            cpu_rng_state=None,
            cuda_rng_state=None,
    ):
        output.copy_(checkpoint["output"].to(device=output.device, dtype=output.dtype))
        noisy_cache.copy_(checkpoint["noisy_cache"].to(device=noisy_cache.device, dtype=noisy_cache.dtype))
        self._restore_cache_from_device(self.kv_cache_clean, checkpoint["kv_cache"])
        self._restore_cache_from_device(self.crossattn_cache, checkpoint["crossattn_cache"])
        self.tokentrim_state = copy.deepcopy(checkpoint["tokentrim_state"])
        self.tokentrim_rate_history = [
            drift.to(device=output.device, copy=True)
            for drift in checkpoint.get("rate_history", [])
        ]
        self.tokentrim_prev_summary = (
            None if checkpoint["prev_summary"] is None
            else checkpoint["prev_summary"].to(device=output.device)
        )
        self.tokentrim_prev_start_frame = checkpoint["prev_start_frame"]
        if cpu_rng_state is not None:
            torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, output.device)
        return checkpoint["next_window_index"]

    @staticmethod
    def _select_tokentrim_checkpoint(checkpoints, target_window_index):
        eligible = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint["next_window_index"] <= target_window_index
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda checkpoint: checkpoint["next_window_index"])

    def _prune_tokentrim_checkpoints(self, checkpoints, current_window_index):
        if self.tokentrim_checkpoint_count <= 0:
            return []
        depths = self.tokentrim_rollback_depths or [self.tokentrim_rollback_windows]
        next_window_index = current_window_index + 1
        target_window_indices = [
            max(0, next_window_index - depth)
            for depth in depths
            if depth > 0
        ]
        useful = []
        seen_next_windows = set()
        for target_window_index in target_window_indices:
            checkpoint = self._select_tokentrim_checkpoint(checkpoints, target_window_index)
            if checkpoint is None:
                continue
            checkpoint_next_window = checkpoint["next_window_index"]
            if checkpoint_next_window in seen_next_windows:
                continue
            useful.append(checkpoint)
            seen_next_windows.add(checkpoint_next_window)

        latest_checkpoint = max(
            checkpoints,
            key=lambda checkpoint: checkpoint["next_window_index"],
        )
        latest_next_window = latest_checkpoint["next_window_index"]
        if latest_next_window not in seen_next_windows:
            useful.append(latest_checkpoint)

        max_stored = self.tokentrim_checkpoint_count + 1
        if len(useful) > max_stored:
            useful = useful[-max_stored:]
        return useful

    def _rollback_candidate_specs(self, failed_window_index, token_indices):
        depths = self.tokentrim_rollback_depths or [self.tokentrim_rollback_windows]
        interventions = self.tokentrim_rollback_interventions or ["none"]
        specs = []
        for depth in depths:
            target_window_index = max(0, failed_window_index - depth)
            if target_window_index >= failed_window_index:
                continue
            for intervention in interventions:
                specs.append({
                    "depth": depth,
                    "target_window_index": target_window_index,
                    "intervention": intervention,
                    "token_indices": None if token_indices is None else token_indices.detach(),
                })
        return specs

    @staticmethod
    def _parse_tokentrim_rollback_intervention(intervention):
        if intervention == "none":
            return "none", None
        if intervention == "suppress":
            return "suppress", None
        for prefix in ("rate_normalize:", "rate_norm:"):
            if intervention.startswith(prefix):
                strength = float(intervention.split(":", 1)[1])
                if not 0.0 <= strength <= 1.0:
                    raise ValueError(f"rate normalization strength must be in [0, 1], got {strength}")
                return "rate_normalize", strength
        for prefix in ("soft_suppress:", "soft:", "scale:"):
            if intervention.startswith(prefix):
                scale = float(intervention.split(":", 1)[1])
                if not 0.0 <= scale <= 1.0:
                    raise ValueError(f"soft suppression scale must be in [0, 1], got {scale}")
                return "soft_suppress", scale
        raise ValueError(
            "Unknown TokenTrim rollback intervention "
            f"{intervention!r}; expected none, suppress, soft_suppress:<scale>, "
            "or rate_normalize:<strength>."
        )

    @staticmethod
    def _latent_frame_embeddings(latents):
        embeddings = latents.float().mean(dim=(-1, -2))
        return torch.nn.functional.normalize(embeddings, dim=-1, eps=1e-6)

    @staticmethod
    def _format_tokentrim_selector_components(components):
        if not components:
            return "selector_components=none"
        return "selector_components=" + ",".join(
            f"{key}:{value:.4f}" for key, value in sorted(components.items())
        )

    def _transition_rate_signal(self, drift):
        drift = drift.detach()
        components = {
            "rate": 0.0,
            "rate_mean": 0.0,
            "rate_std": 0.0,
            "rate_z": 0.0,
        }
        if len(self.tokentrim_rate_history) < self.tokentrim_rate_warmup_steps:
            return 0.0, False, components, None

        history = torch.stack([
            item.to(device=drift.device, dtype=drift.dtype)
            for item in self.tokentrim_rate_history[-self.tokentrim_rate_history_size:]
        ], dim=0)
        expected_mean = history.mean(dim=0)
        expected_std = history.std(dim=0, unbiased=False).clamp_min(self.tokentrim_config.eps)
        z_scores = (drift - expected_mean) / expected_std

        token_count = z_scores.shape[-1]
        top_k = max(1, int(token_count * self.tokentrim_rate_top_fraction))
        top_rate = torch.topk(z_scores, k=top_k, dim=-1, largest=True, sorted=False)
        top_z = top_rate.values
        rate_score = top_z.mean()

        components["rate"] = float(drift.mean().detach().cpu().item())
        components["rate_mean"] = float(expected_mean.mean().detach().cpu().item())
        components["rate_std"] = float(expected_std.mean().detach().cpu().item())
        components["rate_z"] = float(rate_score.detach().cpu().item())
        return (
            components["rate_z"],
            components["rate_z"] > self.tokentrim_rate_z_threshold,
            components,
            top_rate.indices.detach(),
        )

    def _score_transition_rate_candidate(self, candidate_drift):
        if len(self.tokentrim_rate_history) < self.tokentrim_rate_warmup_steps:
            rate_z, _, components, _ = self._transition_rate_signal(candidate_drift)
            return abs(rate_z), components

        drift = candidate_drift.detach()
        history = torch.stack([
            item.to(device=drift.device, dtype=drift.dtype)
            for item in self.tokentrim_rate_history[-self.tokentrim_rate_history_size:]
        ], dim=0)
        expected_mean = history.mean(dim=0)
        expected_std = history.std(dim=0, unbiased=False).clamp_min(self.tokentrim_config.eps)
        normalized_error = torch.abs(drift - expected_mean) / expected_std

        token_count = normalized_error.shape[-1]
        top_k = max(1, int(token_count * self.tokentrim_rate_top_fraction))
        top_error = torch.topk(normalized_error, k=top_k, dim=-1, largest=True, sorted=False).values
        score = top_error.mean()

        components = {
            "rate": float(drift.mean().detach().cpu().item()),
            "rate_mean": float(expected_mean.mean().detach().cpu().item()),
            "rate_std": float(expected_std.mean().detach().cpu().item()),
            "rate_error": float(score.detach().cpu().item()),
        }
        return components["rate_error"], components

    def _accept_transition_rate(self, drift):
        if drift is None:
            return
        self.tokentrim_rate_history.append(drift.detach())
        if len(self.tokentrim_rate_history) > self.tokentrim_rate_history_size:
            self.tokentrim_rate_history = self.tokentrim_rate_history[-self.tokentrim_rate_history_size:]

    def _expected_transition_rate(self, drift):
        if len(self.tokentrim_rate_history) < self.tokentrim_rate_warmup_steps:
            return None
        history = torch.stack([
            item.to(device=drift.device, dtype=drift.dtype)
            for item in self.tokentrim_rate_history[-self.tokentrim_rate_history_size:]
        ], dim=0)
        return history.mean(dim=0)

    def _rate_normalize_candidate(
            self,
            denoised_pred,
            current_start_frame,
            strength,
            token_indices=None,
    ):
        if self.tokentrim_prev_summary is None:
            return denoised_pred, {"rate_normalized": 0.0}

        candidate_block = denoised_pred[:, :self.num_frame_per_block]
        current_summary = self._tokentrim_latent_summary(
            candidate_block,
            patch_size=self.tokentrim_patch_size,
        )
        drift = torch.linalg.vector_norm(current_summary - self.tokentrim_prev_summary, ord=2, dim=-1)
        expected_rate = self._expected_transition_rate(drift)
        if expected_rate is None:
            return denoised_pred, {"rate_normalized": 0.0}

        observed_rate = drift.clamp_min(self.tokentrim_config.eps)
        target_rate = expected_rate.to(device=observed_rate.device, dtype=observed_rate.dtype)
        shrink = (target_rate / observed_rate).clamp(max=1.0)
        token_scale = 1.0 - strength * (1.0 - shrink)
        if token_indices is not None:
            selected = torch.zeros_like(token_scale, dtype=torch.bool)
            selected.scatter_(dim=-1, index=token_indices.to(device=selected.device), value=True)
            token_scale = torch.where(selected, token_scale, torch.ones_like(token_scale))

        patch_h, patch_w = self.tokentrim_patch_size
        batch, frames, channels, height, width = candidate_block.shape
        if height % patch_h != 0 or width % patch_w != 0:
            raise ValueError(f"latent grid {(height, width)} is not divisible by patch size {self.tokentrim_patch_size}")

        previous_frame_summary = self.tokentrim_prev_summary.reshape(
            batch,
            height // patch_h,
            width // patch_w,
            channels,
            patch_h,
            patch_w,
        ).permute(0, 3, 1, 4, 2, 5).contiguous()
        previous_frame = previous_frame_summary.reshape(batch, channels, height, width)
        previous_frames = previous_frame[:, None].expand(-1, frames, -1, -1, -1)

        scale_map = token_scale.reshape(batch, height // patch_h, width // patch_w)
        scale_map = scale_map.repeat_interleave(patch_h, dim=1).repeat_interleave(patch_w, dim=2)
        scale_map = scale_map[:, None, None]

        repaired_block = previous_frames + (candidate_block - previous_frames) * scale_map
        repaired = denoised_pred.clone()
        repaired[:, :self.num_frame_per_block] = repaired_block.to(dtype=denoised_pred.dtype)
        changed_fraction = float((token_scale < 0.999).float().mean().detach().cpu().item())
        return repaired, {
            "rate_normalized": changed_fraction,
            "rate_norm_scale": float(token_scale.mean().detach().cpu().item()),
        }

    def _rate_normalize_strengths(self):
        strengths = []
        for intervention in self.tokentrim_rollback_interventions:
            intervention_kind, intervention_strength = self._parse_tokentrim_rollback_intervention(
                intervention
            )
            if intervention_kind == "rate_normalize":
                strengths.append(intervention_strength)
        for fallback_strength in (0.5, 0.75, 1.0):
            if fallback_strength not in strengths:
                strengths.append(fallback_strength)
        return strengths

    def _estimate_rate_normalized_candidate(
            self,
            denoised_pred,
            current_start_frame,
            severity,
            token_indices,
            strength,
    ):
        repaired, normalize_components = self._rate_normalize_candidate(
            denoised_pred=denoised_pred,
            current_start_frame=current_start_frame,
            strength=strength,
            token_indices=token_indices,
        )
        candidate_block = repaired[:, :self.num_frame_per_block]
        current_summary = self._tokentrim_latent_summary(
            candidate_block,
            patch_size=self.tokentrim_patch_size,
        )
        drift = torch.linalg.vector_norm(current_summary - self.tokentrim_prev_summary, ord=2, dim=-1)
        rate_score, rate_should_prune, rate_components, rate_token_indices = self._transition_rate_signal(drift)
        selector_components = dict(rate_components)
        selector_components["drift"] = float(severity)
        selector_score = (
            self.tokentrim_selector_rate_weight * float(rate_score)
            + self.tokentrim_selector_drift_weight * float(severity)
        )
        return {
            "strength": strength,
            "selector_score": float(selector_score),
            "selector_components": selector_components,
            "rate_components": rate_components,
            "rate_normalize_components": normalize_components,
            "pruned": rate_should_prune,
            "token_indices": rate_token_indices.detach() if rate_token_indices is not None else token_indices,
        }

    def _load_tokentrim_dino_model(self):
        if self._tokentrim_dino_model is None:
            import timm

            print(
                "TokenTrim DINO load:",
                f"model={self.tokentrim_dino_model_name}",
                f"device={self.tokentrim_dino_device}",
            )
            self._tokentrim_dino_model = timm.create_model(
                self.tokentrim_dino_model_name,
                pretrained=True,
                num_classes=0,
            ).eval().requires_grad_(False).to(self.tokentrim_dino_device)
        return self._tokentrim_dino_model

    def _dino_frame_embeddings(self, pixel_frames):
        model = self._load_tokentrim_dino_model()
        frames = pixel_frames.flatten(0, 1)
        if frames.shape[0] > self.tokentrim_dino_max_frames:
            frame_indices = torch.linspace(
                0,
                frames.shape[0] - 1,
                self.tokentrim_dino_max_frames,
                device=frames.device,
            ).round().long()
            frames = frames.index_select(0, frame_indices)
        frames = torch.nn.functional.interpolate(
            frames.float(),
            size=(self.tokentrim_dino_image_size, self.tokentrim_dino_image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        frames = (frames + 1.0) * 0.5
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=frames.device,
            dtype=frames.dtype,
        )[None, :, None, None]
        std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=frames.device,
            dtype=frames.dtype,
        )[None, :, None, None]
        frames = ((frames - mean) / std).to(self.tokentrim_dino_device)
        with torch.no_grad():
            embeddings = model(frames)
        return torch.nn.functional.normalize(embeddings.float(), dim=-1, eps=1e-6)

    def _score_dino_candidate(
            self,
            output,
            denoised_pred,
            current_start_frame,
            current_end_frame,
            severity,
    ):
        candidate_latents = denoised_pred[:, :self.num_frame_per_block]
        context_start_frame = max(
            0,
            current_start_frame - self.tokentrim_selector_context_frames,
        )
        context_latents = output[:, context_start_frame:current_start_frame]
        with torch.no_grad():
            if context_latents.shape[1] > 0:
                combined_pixels = self.vae.decode_to_pixel(
                    torch.cat([context_latents, candidate_latents], dim=1),
                    use_cache=False,
                )
                # Wan's two temporal upsampling stages produce four pixel frames
                # for every appended latent frame after the sequence begins.
                candidate_pixel_count = min(
                    combined_pixels.shape[1],
                    4 * candidate_latents.shape[1],
                )
                context_pixels = combined_pixels[:, :-candidate_pixel_count]
                candidate_pixels = combined_pixels[:, -candidate_pixel_count:]
            else:
                context_pixels = None
                candidate_pixels = self.vae.decode_to_pixel(candidate_latents, use_cache=False)

            candidate_embeddings = self._dino_frame_embeddings(candidate_pixels)
            del candidate_pixels

            subject_cost = torch.tensor(0.0, device=candidate_embeddings.device)
            if context_pixels is not None and context_pixels.shape[1] > 0:
                context_embeddings = self._dino_frame_embeddings(context_pixels)
                del context_pixels
                reference_embedding = torch.nn.functional.normalize(
                    context_embeddings.mean(dim=0, keepdim=True),
                    dim=-1,
                    eps=1e-6,
                )
                subject_cost = 1.0 - (
                    candidate_embeddings * reference_embedding
                ).sum(dim=-1).mean()
                temporal_embeddings = torch.cat(
                    [context_embeddings[-1:], candidate_embeddings],
                    dim=0,
                )
            else:
                temporal_embeddings = candidate_embeddings

            temporal_cost = torch.tensor(0.0, device=candidate_embeddings.device)
            if temporal_embeddings.shape[0] > 1:
                temporal_cost = 1.0 - (
                    temporal_embeddings[:-1] * temporal_embeddings[1:]
                ).sum(dim=-1).mean()

        components = {
            "dino_subject": float(subject_cost.detach().cpu().item()),
            "dino_temporal": float(temporal_cost.detach().cpu().item()),
            "drift": float(severity),
        }
        score = (
            self.tokentrim_dino_subject_weight * components["dino_subject"]
            + self.tokentrim_dino_temporal_weight * components["dino_temporal"]
            + self.tokentrim_selector_drift_weight * components["drift"]
        )
        return float(score), components

    def _score_tokentrim_candidate(
            self,
            output,
            denoised_pred,
            current_start_frame,
            current_end_frame,
            severity,
            drift=None,
    ):
        if self.tokentrim_rollback_selector == "drift":
            return float(severity), {"drift": float(severity)}

        if self.tokentrim_rollback_selector == "rate_anomaly":
            if drift is None:
                return float(severity), {"drift": float(severity)}
            rate_score, rate_components = self._score_transition_rate_candidate(drift)
            rate_components["drift"] = float(severity)
            score = (
                self.tokentrim_selector_rate_weight * rate_score
                + self.tokentrim_selector_drift_weight * float(severity)
            )
            return float(score), rate_components

        if self.tokentrim_rollback_selector == "dino":
            return self._score_dino_candidate(
                output=output,
                denoised_pred=denoised_pred,
                current_start_frame=current_start_frame,
                current_end_frame=current_end_frame,
                severity=severity,
            )

        candidate_frames = denoised_pred[:, :current_end_frame - current_start_frame]
        context_start_frame = max(0, current_start_frame - self.tokentrim_selector_context_frames)
        context = output[:, context_start_frame:current_start_frame]

        with torch.no_grad():
            candidate_embeddings = self._latent_frame_embeddings(candidate_frames)[0]
            components = {"drift": float(severity)}

            if context.shape[1] > 0:
                context_embeddings = self._latent_frame_embeddings(context)[0]
                reference_embedding = context_embeddings.mean(dim=0, keepdim=True)
                reference_embedding = torch.nn.functional.normalize(reference_embedding, dim=-1, eps=1e-6)
                subject_cost = 1.0 - (
                    candidate_embeddings * reference_embedding
                ).sum(dim=-1).mean()
                boundary_cost = 1.0 - (
                    context_embeddings[-1] * candidate_embeddings[0]
                ).sum()
                combined_embeddings = torch.cat([context_embeddings[-1:], candidate_embeddings], dim=0)
            else:
                subject_cost = candidate_embeddings.new_tensor(0.0)
                boundary_cost = candidate_embeddings.new_tensor(0.0)
                combined_embeddings = candidate_embeddings

            if combined_embeddings.shape[0] > 1:
                motion_cost = 1.0 - (
                    combined_embeddings[1:] * combined_embeddings[:-1]
                ).sum(dim=-1).mean()
            else:
                motion_cost = candidate_embeddings.new_tensor(0.0)

            components["subject"] = float(subject_cost.item())
            components["boundary"] = float(boundary_cost.item())
            components["motion"] = float(motion_cost.item())

            score = (
                self.tokentrim_selector_subject_weight * components["subject"]
                + self.tokentrim_selector_boundary_weight * components["boundary"]
                + self.tokentrim_selector_motion_weight * components["motion"]
                + self.tokentrim_selector_drift_weight * components["drift"]
            )
            return float(score), components

    def _reset_clean_cache(self, device):
        for layer_cache in self.crossattn_cache:
            layer_cache["is_init"] = False
        for layer_cache in self.kv_cache_clean:
            layer_cache["global_end_index"] = torch.tensor([0], dtype=torch.long, device=device)
            layer_cache["local_end_index"] = torch.tensor([0], dtype=torch.long, device=device)

    def _rebuild_clean_cache_from_output(
            self,
            output,
            conditional_dict,
            window_start_blocks,
            up_to_window_index,
    ):
        self._reset_clean_cache(output.device)
        if up_to_window_index <= 0:
            return

        with torch.no_grad():
            context_timestep = torch.ones(
                [output.shape[0], self.num_frame_per_block],
                device=output.device,
                dtype=torch.float32,
            ) * self.args.context_noise

            for replay_window_index in range(up_to_window_index):
                start_block = window_start_blocks[replay_window_index]
                current_start_frame = start_block * self.num_frame_per_block
                cached_block = output[
                    :, current_start_frame:current_start_frame + self.num_frame_per_block
                ]
                self.generator(
                    noisy_image_or_video=cached_block,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    updating_cache=True,
                )

    def _maybe_tokentrim_reroll(
            self,
            denoised_pred,
            noisy_input,
            conditional_dict,
            current_timestep,
            current_start_frame,
            cache_snapshot
    ):
        if not self.tokentrim_enabled:
            return denoised_pred
        self.tokentrim_last_pruned = False
        self.tokentrim_last_token_indices = None
        self.tokentrim_last_severity = None
        self.tokentrim_last_threshold = None
        self.tokentrim_last_drift = None
        self.tokentrim_last_rate_score = None
        self.tokentrim_last_rate_components = None
        self.tokentrim_last_rate_token_indices = None
        if denoised_pred.shape[0] != 1:
            raise ValueError("The current TokenTrim integration supports batch_size=1 / num_samples=1")

        candidate_block = denoised_pred[:, :self.num_frame_per_block]
        current_summary = self._tokentrim_latent_summary(
            candidate_block,
            patch_size=self.tokentrim_patch_size,
        )

        if self.tokentrim_prev_summary is None:
            self.tokentrim_prev_summary = current_summary.detach()
            self.tokentrim_prev_start_frame = current_start_frame
            return denoised_pred

        if self.tokentrim_prev_start_frame == current_start_frame:
            self.tokentrim_prev_summary = current_summary.detach()
            return denoised_pred

        result = self.tokentrim_state.evaluate_summaries(
            self.tokentrim_prev_summary,
            current_summary,
        )
        if self.tokentrim_debug:
            print(
                "TokenTrim decision:",
                f"decision={result.decision.value}",
                f"severity={result.severity:.4f}",
                f"threshold={result.threshold:.4f}" if result.threshold is not None else "threshold=None",
                f"stats_count={result.stats_count}",
                f"start_frame={current_start_frame}",
            )
        self.tokentrim_last_pruned = result.should_prune
        self.tokentrim_last_token_indices = result.token_indices.detach()
        self.tokentrim_last_severity = result.severity
        self.tokentrim_last_threshold = result.threshold
        self.tokentrim_last_drift = result.drift.detach()
        rate_score, rate_should_prune, rate_components, rate_token_indices = self._transition_rate_signal(result.drift)
        self.tokentrim_last_rate_score = rate_score
        self.tokentrim_last_rate_components = rate_components
        self.tokentrim_last_rate_token_indices = rate_token_indices
        if self.tokentrim_trigger_mode == "rate_anomaly":
            self.tokentrim_last_pruned = rate_should_prune
            if rate_should_prune and rate_token_indices is not None:
                self.tokentrim_last_token_indices = rate_token_indices.detach()
        if self.tokentrim_debug:
            print(
                "TokenTrim rate signal:",
                f"trigger={self.tokentrim_trigger_mode}",
                f"rate_z={rate_components['rate_z']:.4f}",
                f"threshold={self.tokentrim_rate_z_threshold:.4f}",
                f"history={len(self.tokentrim_rate_history)}",
                f"decision={'prune_and_reroll' if self.tokentrim_last_pruned else 'accept'}",
            )

        rerolls_remaining = self.tokentrim_max_rerolls
        while result.should_prune and rerolls_remaining > 0:
            print(
                "TokenTrim pruning:",
                f"severity={result.severity:.4f}",
                f"threshold={result.threshold:.4f}" if result.threshold is not None else "threshold=None",
                f"rerolls_remaining={rerolls_remaining}",
            )
            self._restore_kv_cache(self.kv_cache_clean, cache_snapshot)
            self._tokentrim_suppress_cache(
                self.kv_cache_clean,
                result.token_indices[0],
                frame_seq_length=self.frame_seq_length,
                block_length=self.num_frame_per_block * self.frame_seq_length,
                sink_blocks=self.tokentrim_sink_blocks,
            )
            _, denoised_pred = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=current_timestep,
                kv_cache=self.kv_cache_clean,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length
            )
            candidate_block = denoised_pred[:, :self.num_frame_per_block]
            current_summary = self._tokentrim_latent_summary(
                candidate_block,
                patch_size=self.tokentrim_patch_size,
            )
            result = self.tokentrim_state.evaluate_summaries(
                self.tokentrim_prev_summary,
                current_summary,
            )
            if self.tokentrim_debug:
                print(
                    "TokenTrim reroll decision:",
                    f"decision={result.decision.value}",
                    f"severity={result.severity:.4f}",
                    f"threshold={result.threshold:.4f}" if result.threshold is not None else "threshold=None",
                    f"stats_count={result.stats_count}",
                    f"start_frame={current_start_frame}",
                )
            rerolls_remaining -= 1

        self.tokentrim_last_pruned = result.should_prune
        self.tokentrim_last_token_indices = result.token_indices.detach()
        self.tokentrim_last_severity = result.severity
        self.tokentrim_last_threshold = result.threshold
        self.tokentrim_last_drift = result.drift.detach()
        rate_score, rate_should_prune, rate_components, rate_token_indices = self._transition_rate_signal(result.drift)
        self.tokentrim_last_rate_score = rate_score
        self.tokentrim_last_rate_components = rate_components
        self.tokentrim_last_rate_token_indices = rate_token_indices
        if self.tokentrim_trigger_mode == "rate_anomaly":
            self.tokentrim_last_pruned = rate_should_prune
            if rate_should_prune and rate_token_indices is not None:
                self.tokentrim_last_token_indices = rate_token_indices.detach()
        self.tokentrim_state.accept(result)
        self.tokentrim_prev_summary = current_summary.detach()
        self.tokentrim_prev_start_frame = current_start_frame
        return denoised_pred

    def inference_rolling_forcing(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if self.tokentrim_enabled:
            self.tokentrim_state = self._tokentrim_state_cls(self.tokentrim_config)
            self.tokentrim_rate_history = []
            self.tokentrim_prev_summary = None
            self.tokentrim_prev_start_frame = None

        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_clean is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            # reset cross attn cache
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache_clean)):
                self.kv_cache_clean[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_clean[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # implementing rolling forcing 
        # construct the rolling forcing windows
        num_denoising_steps = len(self.denoising_step_list)
        rolling_window_length_blocks = num_denoising_steps
        window_start_blocks = []
        window_end_blocks = []
        window_num = num_blocks + rolling_window_length_blocks - 1

        for window_index in range(window_num):
            start_block = max(0, window_index - rolling_window_length_blocks + 1)
            end_block = min(num_blocks - 1, window_index)
            window_start_blocks.append(start_block)
            window_end_blocks.append(end_block)

        # init noisy cache
        noisy_cache = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # init denosing timestep, same accross windows
        shared_timestep = torch.ones(
            [batch_size, rolling_window_length_blocks * self.num_frame_per_block],
            device=noise.device,
            dtype=torch.float32)
        
        for index, current_timestep in enumerate(reversed(self.denoising_step_list)): # from clean to noisy 
            shared_timestep[:, index * self.num_frame_per_block:(index + 1) * self.num_frame_per_block] *= current_timestep


        tokentrim_rollback_suppressions = {}
        tokentrim_rollback_rate_normalizations = {}
        tokentrim_rollback_attempts = {}
        tokentrim_rollback_candidates = {}
        tokentrim_rollback_queues = {}
        tokentrim_active_candidate = None
        tokentrim_finalizing_windows = set()
        tokentrim_checkpoints = []
        tokentrim_original_checkpoints = {}
        initial_cpu_rng_state = torch.get_rng_state() if self.tokentrim_rollback_reset_rng else None
        initial_cuda_rng_state = (
            torch.cuda.get_rng_state(noise.device)
            if self.tokentrim_rollback_reset_rng and noise.device.type == "cuda"
            else None
        )

        # Denoising loop with rolling forcing
        window_index = 0
        while window_index < window_num:

            if profile:
                block_start.record()

            print('window_index:', window_index)
            start_block = window_start_blocks[window_index]
            end_block = window_end_blocks[window_index] # include
            print(f"start_block: {start_block}, end_block: {end_block}")

            current_start_frame = start_block * self.num_frame_per_block
            current_end_frame = (end_block + 1) * self.num_frame_per_block # not include
            current_num_frames = current_end_frame - current_start_frame

            # noisy_input: new noise and previous denoised noisy frames, only last block is pure noise
            if current_num_frames == rolling_window_length_blocks * self.num_frame_per_block or current_start_frame == 0:
                noisy_input = torch.cat([
                    noisy_cache[:, current_start_frame : current_end_frame - self.num_frame_per_block],
                    noise[:, current_end_frame - self.num_frame_per_block : current_end_frame ]
                ], dim=1)
            else: # at the end of the video
                noisy_input = noisy_cache[:, current_start_frame:current_end_frame]

            # init denosing timestep
            if current_num_frames == rolling_window_length_blocks * self.num_frame_per_block:
                current_timestep = shared_timestep
            elif current_start_frame == 0:
                current_timestep = shared_timestep[:,-current_num_frames:]
            elif current_end_frame == num_frames:
                current_timestep = shared_timestep[:,:current_num_frames]
            else:
                raise ValueError("current_num_frames should be equal to rolling_window_length_blocks * self.num_frame_per_block, or the first or last window.")


            if self.tokentrim_enabled and window_index in tokentrim_rollback_suppressions:
                suppression_spec = tokentrim_rollback_suppressions[window_index]
                suppression_tokens = suppression_spec["token_indices"]
                suppression_scale = suppression_spec.get("scale")
                suppression_label = suppression_spec.get("intervention", "suppress")
                print(
                    "TokenTrim rollback suppression:",
                    f"window={window_index}",
                    f"tokens={suppression_tokens.numel()}",
                    f"intervention={suppression_label}",
                    f"scale={suppression_scale}",
                )
                self._tokentrim_suppress_cache(
                    self.kv_cache_clean,
                    suppression_tokens,
                    frame_seq_length=self.frame_seq_length,
                    block_length=self.num_frame_per_block * self.frame_seq_length,
                    sink_blocks=self.tokentrim_sink_blocks,
                    scale=suppression_scale,
                )

            # calling DiT
            pregeneration_cpu_rng_state = torch.get_rng_state()
            pregeneration_cuda_rng_state = (
                torch.cuda.get_rng_state(noise.device) if noise.device.type == "cuda" else None
            )
            tokentrim_cache_snapshot = (
                self._clone_kv_cache(self.kv_cache_clean)
                if self.tokentrim_enabled and self.tokentrim_max_rerolls > 0
                else None
            )
            _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=current_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            tokentrim_rate_normalize_components = None
            if (
                    self.tokentrim_enabled
                    and (
                        (
                            tokentrim_active_candidate is not None
                            and tokentrim_active_candidate.get("intervention") == "rate_normalize"
                            and tokentrim_active_candidate.get("target_window_index") == window_index
                        )
                        or window_index in tokentrim_rollback_rate_normalizations
                    )
            ):
                rate_normalize_strength = (
                    tokentrim_active_candidate.get("intervention_strength", 1.0)
                    if tokentrim_active_candidate is not None
                    and tokentrim_active_candidate.get("target_window_index") == window_index
                    else tokentrim_rollback_rate_normalizations[window_index]["strength"]
                )
                rate_normalize_tokens = (
                    tokentrim_active_candidate.get("token_indices")
                    if tokentrim_active_candidate is not None
                    and tokentrim_active_candidate.get("target_window_index") == window_index
                    else tokentrim_rollback_rate_normalizations[window_index].get("token_indices")
                )
                denoised_pred, tokentrim_rate_normalize_components = self._rate_normalize_candidate(
                    denoised_pred=denoised_pred,
                    current_start_frame=current_start_frame,
                    strength=rate_normalize_strength,
                    token_indices=rate_normalize_tokens,
                )
                print(
                    "TokenTrim rollback rate_normalize:",
                    f"window={window_index}",
                    f"strength={rate_normalize_strength}",
                    f"tokens={rate_normalize_tokens.numel() if rate_normalize_tokens is not None else 'all'}",
                    self._format_tokentrim_selector_components(tokentrim_rate_normalize_components),
                )

            denoised_pred = self._maybe_tokentrim_reroll(
                denoised_pred=denoised_pred,
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                current_timestep=current_timestep,
                current_start_frame=current_start_frame,
                cache_snapshot=tokentrim_cache_snapshot,
            )

            tokentrim_candidate_recorded = False
            if (
                    tokentrim_active_candidate is not None
                    and tokentrim_active_candidate["window_index"] == window_index
                    and self.tokentrim_last_severity is not None
            ):
                tokentrim_active_candidate["severity"] = self.tokentrim_last_severity
                tokentrim_active_candidate["pruned"] = self.tokentrim_last_pruned
                selector_score, selector_components = self._score_tokentrim_candidate(
                    output=output,
                    denoised_pred=denoised_pred,
                    current_start_frame=current_start_frame,
                    current_end_frame=current_end_frame,
                    severity=self.tokentrim_last_severity,
                    drift=self.tokentrim_last_drift,
                )
                tokentrim_active_candidate["selector_score"] = selector_score
                tokentrim_active_candidate["selector_components"] = selector_components
                if tokentrim_rate_normalize_components is not None:
                    tokentrim_active_candidate["rate_normalize_components"] = tokentrim_rate_normalize_components
                tokentrim_rollback_candidates.setdefault(window_index, []).append(tokentrim_active_candidate)
                print(
                    "TokenTrim rollback candidate:",
                    f"window={window_index}",
                    f"candidate={len(tokentrim_rollback_candidates[window_index])}",
                    f"depth={tokentrim_active_candidate.get('depth')}",
                    f"intervention={tokentrim_active_candidate.get('intervention')}",
                    f"scale={tokentrim_active_candidate.get('suppress_scale')}",
                    f"strength={tokentrim_active_candidate.get('intervention_strength')}",
                    f"severity={self.tokentrim_last_severity:.4f}",
                    f"selector={self.tokentrim_rollback_selector}",
                    f"selector_score={selector_score:.4f}",
                    self._format_tokentrim_selector_components(selector_components),
                    self._format_tokentrim_selector_components(
                        tokentrim_rate_normalize_components
                    ) if tokentrim_rate_normalize_components is not None else "rate_normalize=none",
                    f"pruned={self.tokentrim_last_pruned}",
                )
                tokentrim_active_candidate = None
                tokentrim_candidate_recorded = True

            if (
                    self.tokentrim_enabled
                    and self.tokentrim_rollback_experimental
                    and self.tokentrim_rollback_windows > 0
                    and self.tokentrim_rollback_max_attempts > 0
                    and tokentrim_active_candidate is None
                    and (
                        self.tokentrim_last_pruned
                        or (
                            tokentrim_candidate_recorded
                            and window_index in tokentrim_rollback_queues
                        )
                    )
            ):
                attempts_used = tokentrim_rollback_attempts.get(window_index, 0)
                if window_index not in tokentrim_rollback_queues:
                    base_specs = self._rollback_candidate_specs(
                        window_index,
                        self.tokentrim_last_token_indices,
                    )
                    tokentrim_rollback_queues[window_index] = [
                        spec.copy()
                        for spec in base_specs
                        for _ in range(max(1, self.tokentrim_rollback_best_of_n))
                    ]
                candidate_queue = tokentrim_rollback_queues[window_index]
                candidates = tokentrim_rollback_candidates.get(window_index, [])
                if self.tokentrim_rollback_include_original and not any(
                        candidate.get("intervention") == "original"
                        for candidate in candidates
                ):
                    if window_index not in tokentrim_original_checkpoints:
                        tokentrim_original_checkpoints[window_index] = self._make_tokentrim_checkpoint(
                            next_window_index=window_index,
                            output=output,
                            noisy_cache=noisy_cache,
                        )
                    original_checkpoint = tokentrim_original_checkpoints[window_index]
                    selector_score, selector_components = self._score_tokentrim_candidate(
                        output=output,
                        denoised_pred=denoised_pred,
                        current_start_frame=current_start_frame,
                        current_end_frame=current_end_frame,
                        severity=self.tokentrim_last_severity,
                        drift=self.tokentrim_last_drift,
                    )
                    original_token_indices = (
                        None
                        if self.tokentrim_last_token_indices is None
                        else self.tokentrim_last_token_indices.detach()
                    )
                    original_fallbacks = []
                    if (
                            original_token_indices is not None
                            and self.tokentrim_rollback_selector != "dino"
                    ):
                        for normalize_strength in self._rate_normalize_strengths():
                            fallback = self._estimate_rate_normalized_candidate(
                                denoised_pred=denoised_pred,
                                current_start_frame=current_start_frame,
                                severity=self.tokentrim_last_severity,
                                token_indices=original_token_indices,
                                strength=normalize_strength,
                            )
                            original_fallbacks.append(fallback)
                    candidates.append({
                        "window_index": window_index,
                        "severity": self.tokentrim_last_severity,
                        "pruned": self.tokentrim_last_pruned,
                        "selector_score": selector_score,
                        "selector_components": selector_components,
                        "cpu_rng_state": pregeneration_cpu_rng_state,
                        "cuda_rng_state": pregeneration_cuda_rng_state,
                        "checkpoint": original_checkpoint,
                        "depth": 0,
                        "intervention": "original",
                        "target_window_index": window_index,
                        "token_indices": original_token_indices,
                        "original_fallbacks": original_fallbacks,
                    })
                    tokentrim_rollback_candidates[window_index] = candidates
                    print(
                        "TokenTrim rollback candidate:",
                        f"window={window_index}",
                        f"candidate={len(candidates)}",
                        "depth=0",
                        "intervention=original",
                        f"severity={self.tokentrim_last_severity:.4f}",
                        f"selector={self.tokentrim_rollback_selector}",
                        f"selector_score={selector_score:.4f}",
                        self._format_tokentrim_selector_components(selector_components),
                        f"pruned={self.tokentrim_last_pruned}",
                    )
                if initial_latent is not None:
                    print(
                        "TokenTrim rollback skipped:",
                        "reason=initial_latent_not_supported",
                        f"from_window={window_index}",
                    )
                elif not candidate_queue:
                    print(
                        "TokenTrim rollback skipped:",
                        "reason=no_candidate_specs",
                        f"from_window={window_index}",
                    )
                elif window_index in tokentrim_finalizing_windows:
                    print(
                        "TokenTrim rollback skipped:",
                        "reason=selected_candidate_replay",
                        f"from_window={window_index}",
                    )
                else:
                    if attempts_used >= len(candidate_queue) or attempts_used >= self.tokentrim_rollback_max_attempts:
                        if not candidates:
                            print(
                                "TokenTrim rollback skipped:",
                                "reason=max_attempts_reached",
                                f"from_window={window_index}",
                                f"attempts={attempts_used}",
                            )
                            output[:, current_start_frame:current_end_frame] = denoised_pred
                        else:
                            failed_window_index = window_index
                            rollback_candidates = [
                                candidate
                                for candidate in candidates
                                if candidate.get("intervention") != "original"
                            ]
                            non_pruned_rollback_candidates = [
                                candidate
                                for candidate in rollback_candidates
                                if not candidate.get("pruned", True)
                            ]
                            original_candidate = next(
                                (
                                    candidate
                                    for candidate in candidates
                                    if candidate.get("intervention") == "original"
                                ),
                                None,
                            )
                            if self.tokentrim_rollback_selector == "dino":
                                best_candidate = min(
                                    candidates,
                                    key=lambda candidate: candidate.get(
                                        "selector_score",
                                        candidate["severity"],
                                    ),
                                )
                                selection_reason = "dino_best_candidate"
                            elif non_pruned_rollback_candidates:
                                best_candidate = min(
                                    non_pruned_rollback_candidates,
                                    key=lambda candidate: candidate.get(
                                        "selector_score",
                                        candidate["severity"],
                                    ),
                                )
                                selection_reason = "best_non_pruned_rollback"
                            elif original_candidate is not None:
                                best_candidate = original_candidate
                                selection_reason = "original_adaptive_fallback"
                            else:
                                best_candidate = min(
                                    candidates,
                                    key=lambda candidate: candidate.get(
                                        "selector_score",
                                        candidate["severity"],
                                    ),
                                )
                                selection_reason = "best_available_no_original"
                            restore_checkpoint = best_candidate.get("checkpoint")
                            best_intervention = best_candidate.get("intervention")
                            best_depth = best_candidate.get("depth")
                            winner_checkpoint = (
                                "original"
                                if best_intervention == "original"
                                else f"rollback_minus_{best_depth}"
                            )
                            print(
                                "TokenTrim rollback select:",
                                f"window={window_index}",
                                f"candidates={len(candidates)}",
                                f"winner={best_intervention}",
                                f"winner_checkpoint={winner_checkpoint}",
                                f"reason={selection_reason}",
                                f"best_severity={best_candidate['severity']:.4f}",
                                f"selector={self.tokentrim_rollback_selector}",
                                f"best_score={best_candidate.get('selector_score', best_candidate['severity']):.4f}",
                                self._format_tokentrim_selector_components(
                                    best_candidate.get("selector_components", {})
                                ),
                                f"depth={best_candidate.get('depth')}",
                                f"intervention={best_candidate.get('intervention')}",
                                f"scale={best_candidate.get('suppress_scale')}",
                                f"strength={best_candidate.get('intervention_strength')}",
                                f"checkpoint_next={restore_checkpoint['next_window_index'] if restore_checkpoint else 0}",
                            )
                            tokentrim_rollback_suppressions.clear()
                            tokentrim_rollback_rate_normalizations.clear()
                            if (
                                    best_intervention == "original"
                                    and selection_reason == "original_adaptive_fallback"
                                    and best_candidate.get("token_indices") is not None
                            ):
                                original_fallbacks = best_candidate.get("original_fallbacks", [])
                                non_pruned_fallbacks = [
                                    fallback
                                    for fallback in original_fallbacks
                                    if not fallback.get("pruned", True)
                                ]
                                if non_pruned_fallbacks:
                                    original_fallback = min(
                                        non_pruned_fallbacks,
                                        key=lambda fallback: fallback["strength"],
                                    )
                                elif original_fallbacks:
                                    original_fallback = min(
                                        original_fallbacks,
                                        key=lambda fallback: fallback.get(
                                            "selector_score",
                                            float("inf"),
                                        ),
                                    )
                                else:
                                    original_fallback = None
                                if original_fallback is not None:
                                    tokentrim_rollback_rate_normalizations[best_candidate["target_window_index"]] = {
                                        "strength": original_fallback["strength"],
                                        "token_indices": best_candidate.get("token_indices"),
                                    }
                                    print(
                                        "TokenTrim rollback original fallback:",
                                        "intervention=rate_normalize",
                                        f"window={best_candidate['target_window_index']}",
                                        f"strength={original_fallback['strength']}",
                                        f"tokens={best_candidate['token_indices'].numel()}",
                                        f"fallback_pruned={original_fallback.get('pruned')}",
                                        f"fallback_score={original_fallback.get('selector_score', 0.0):.4f}",
                                        self._format_tokentrim_selector_components(
                                            original_fallback.get("selector_components", {})
                                        ),
                                    )
                            if (
                                    best_intervention in {"suppress", "soft_suppress"}
                                    and best_candidate.get("token_indices") is not None
                            ):
                                tokentrim_rollback_suppressions[best_candidate["target_window_index"]] = {
                                    "token_indices": best_candidate["token_indices"],
                                    "scale": best_candidate.get("suppress_scale"),
                                    "intervention": best_intervention,
                                }
                            if best_intervention == "rate_normalize":
                                tokentrim_rollback_rate_normalizations[best_candidate["target_window_index"]] = {
                                    "strength": best_candidate.get("intervention_strength", 1.0),
                                    "token_indices": best_candidate.get("token_indices"),
                                }
                            if restore_checkpoint is not None:
                                replay_start_window = restore_checkpoint["next_window_index"]
                                window_index = self._restore_tokentrim_checkpoint(
                                    restore_checkpoint,
                                    output,
                                    noisy_cache,
                                    cpu_rng_state=best_candidate["cpu_rng_state"],
                                    cuda_rng_state=best_candidate["cuda_rng_state"],
                                )
                            else:
                                replay_start_window = 0
                                output.zero_()
                                noisy_cache.zero_()
                                self._reset_clean_cache(noise.device)
                                self.tokentrim_state = self._tokentrim_state_cls(self.tokentrim_config)
                                self.tokentrim_rate_history = []
                                self.tokentrim_prev_summary = None
                                self.tokentrim_prev_start_frame = None
                                torch.set_rng_state(best_candidate["cpu_rng_state"])
                                if best_candidate["cuda_rng_state"] is not None:
                                    torch.cuda.set_rng_state(best_candidate["cuda_rng_state"], noise.device)
                                window_index = 0
                            for finalizing_window in range(replay_start_window, failed_window_index + 1):
                                tokentrim_rollback_attempts.pop(finalizing_window, None)
                                tokentrim_rollback_candidates.pop(finalizing_window, None)
                                tokentrim_rollback_queues.pop(finalizing_window, None)
                                tokentrim_original_checkpoints.pop(finalizing_window, None)
                                tokentrim_finalizing_windows.add(finalizing_window)
                            continue
                    else:
                        candidate_spec = candidate_queue[attempts_used]
                        target_window_index = candidate_spec["target_window_index"]
                        restore_checkpoint = self._select_tokentrim_checkpoint(
                            tokentrim_checkpoints,
                            target_window_index,
                        )
                        cpu_rng_state = torch.get_rng_state()
                        cuda_rng_state = torch.cuda.get_rng_state(noise.device) if noise.device.type == "cuda" else None
                        print(
                            "TokenTrim rollback restart:",
                            f"from_window={window_index}",
                            f"to_window={target_window_index}",
                            f"checkpoint_next={restore_checkpoint['next_window_index'] if restore_checkpoint else 0}",
                            f"attempt={attempts_used + 1}/{self.tokentrim_rollback_max_attempts}",
                            f"candidate={len(candidates) + 1}/{len(candidate_queue)}",
                            f"depth={candidate_spec['depth']}",
                            f"intervention={candidate_spec['intervention']}",
                            f"reset_rng={self.tokentrim_rollback_reset_rng}",
                        )
                        tokentrim_rollback_attempts[window_index] = attempts_used + 1
                        tokentrim_active_candidate = {
                            "window_index": window_index,
                            "cpu_rng_state": cpu_rng_state,
                            "cuda_rng_state": cuda_rng_state,
                            "checkpoint": restore_checkpoint,
                            "depth": candidate_spec["depth"],
                            "intervention": candidate_spec["intervention"],
                            "target_window_index": target_window_index,
                            "token_indices": candidate_spec["token_indices"],
                        }
                        tokentrim_rollback_suppressions.clear()
                        tokentrim_rollback_rate_normalizations.clear()
                        intervention_kind, suppress_scale = self._parse_tokentrim_rollback_intervention(
                            candidate_spec["intervention"]
                        )
                        tokentrim_active_candidate["intervention"] = intervention_kind
                        tokentrim_active_candidate["suppress_scale"] = suppress_scale
                        tokentrim_active_candidate["intervention_strength"] = suppress_scale
                        if intervention_kind in {"suppress", "soft_suppress"} and candidate_spec["token_indices"] is not None:
                            tokentrim_rollback_suppressions[target_window_index] = {
                                "token_indices": candidate_spec["token_indices"],
                                "scale": suppress_scale,
                                "intervention": intervention_kind,
                            }
                        if restore_checkpoint is not None:
                            window_index = self._restore_tokentrim_checkpoint(
                                restore_checkpoint,
                                output,
                                noisy_cache,
                                cpu_rng_state=initial_cpu_rng_state if self.tokentrim_rollback_reset_rng else cpu_rng_state,
                                cuda_rng_state=initial_cuda_rng_state if self.tokentrim_rollback_reset_rng else cuda_rng_state,
                            )
                        else:
                            output.zero_()
                            noisy_cache.zero_()
                            self._reset_clean_cache(noise.device)
                            self.tokentrim_state = self._tokentrim_state_cls(self.tokentrim_config)
                            self.tokentrim_rate_history = []
                            self.tokentrim_prev_summary = None
                            self.tokentrim_prev_start_frame = None
                            if self.tokentrim_rollback_reset_rng and initial_cpu_rng_state is not None:
                                torch.set_rng_state(initial_cpu_rng_state)
                                if initial_cuda_rng_state is not None:
                                    torch.cuda.set_rng_state(initial_cuda_rng_state, noise.device)
                            window_index = 0
                        continue

            output[:, current_start_frame:current_end_frame] = denoised_pred
            if self.tokentrim_enabled:
                self._accept_transition_rate(self.tokentrim_last_drift)
                

            # update noisy_cache, which is detached from the computation graph
            with torch.no_grad():
                for block_idx in range(start_block, end_block + 1):
                    
                    block_time_step = current_timestep[:, 
                                    (block_idx - start_block)*self.num_frame_per_block : 
                                    (block_idx - start_block+1)*self.num_frame_per_block].mean().item()
                    matches = torch.abs(self.denoising_step_list - block_time_step) < 1e-4
                    block_timestep_index = torch.nonzero(matches, as_tuple=True)[0]

                    if block_timestep_index == len(self.denoising_step_list) - 1:
                        continue

                    next_timestep = self.denoising_step_list[block_timestep_index + 1].to(noise.device)

                    noisy_cache[:, block_idx * self.num_frame_per_block:
                                    (block_idx+1) * self.num_frame_per_block] = \
                        self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])[:, (block_idx - start_block)*self.num_frame_per_block:
                                                                    (block_idx - start_block+1)*self.num_frame_per_block]


            # rerun with timestep zero to update the clean cache, which is also detached from the computation graph
            with torch.no_grad():
                context_timestep = torch.ones_like(current_timestep) * self.args.context_noise
                # # add context noise
                # denoised_pred = self.scheduler.add_noise(
                #     denoised_pred.flatten(0, 1),
                #     torch.randn_like(denoised_pred.flatten(0, 1)),
                #     context_timestep * torch.ones(
                #         [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                # ).unflatten(0, denoised_pred.shape[:2])

                # only cache the first block
                denoised_pred = denoised_pred[:,:self.num_frame_per_block]
                context_timestep = context_timestep[:,:self.num_frame_per_block]
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    updating_cache=True,
                )

            if (
                    self.tokentrim_enabled
                    and self.tokentrim_checkpoint_count > 0
                    and tokentrim_active_candidate is None
                    and (window_index + 1) % self.tokentrim_checkpoint_interval == 0
            ):
                checkpoint = self._make_tokentrim_checkpoint(
                    next_window_index=window_index + 1,
                    output=output,
                    noisy_cache=noisy_cache,
                )
                tokentrim_checkpoints.append(checkpoint)
                tokentrim_checkpoints = self._prune_tokentrim_checkpoints(
                    tokentrim_checkpoints,
                    window_index,
                )
                print(
                    "TokenTrim checkpoint:",
                    f"next_window={checkpoint['next_window_index']}",
                    f"stored={len(tokentrim_checkpoints)}/{self.tokentrim_checkpoint_count}",
                    f"device={self.tokentrim_checkpoint_device}",
                )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            window_index += 1


        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output. Long runs are decode-memory bound, so drop
        # rolling inference state before expanding latents into pixel frames.
        del noisy_cache
        self.kv_cache_clean = None
        self.crossattn_cache = None
        self.tokentrim_prev_summary = None
        gc.collect()
        torch.cuda.empty_cache()

        video = self.vae.decode_to_pixel(output, use_cache=False)
        video.mul_(0.5).add_(0.5).clamp_(0, 1)

        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output
        else:
            return video



    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache_clean = []
        # if self.local_attn_size != -1:
        #     # Use the local attention size to compute the KV cache size
        #     kv_cache_size = self.local_attn_size * self.frame_seq_length
        # else:
        #     # Use the default KV cache size
        kv_cache_size = 1560 * 24

        for _ in range(self.num_transformer_blocks):
            kv_cache_clean.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_clean = kv_cache_clean  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
