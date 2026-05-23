from typing import List, Optional
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
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
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
        self.tokentrim_prev_summary = None
        self.tokentrim_prev_start_frame = None
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


        # Denoising loop with rolling forcing
        for window_index in range(window_num):

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


            # calling DiT
            tokentrim_cache_snapshot = self._clone_kv_cache(self.kv_cache_clean) if self.tokentrim_enabled else None
            _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=current_timestep,
                    kv_cache=self.kv_cache_clean,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            denoised_pred = self._maybe_tokentrim_reroll(
                denoised_pred=denoised_pred,
                noisy_input=noisy_input,
                conditional_dict=conditional_dict,
                current_timestep=current_timestep,
                current_start_frame=current_start_frame,
                cache_snapshot=tokentrim_cache_snapshot,
            )

            output[:, current_start_frame:current_end_frame] = denoised_pred
                

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

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)


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
