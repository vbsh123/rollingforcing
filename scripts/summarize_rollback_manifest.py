#!/usr/bin/env python3
import argparse
import collections
import json
import pathlib
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--json-out")
    args = parser.parse_args()

    records = []
    for pattern in args.manifests:
        for path in pathlib.Path().glob(pattern):
            with path.open(encoding="utf-8") as manifest:
                records.extend(json.loads(line) for line in manifest if line.strip())

    by_source = collections.defaultdict(list)
    for record in records:
        by_source[record["event_source"]].append(record)

    summary = {"total_events": len(records), "sources": {}}
    if not records:
        print("total_events=0")
        if args.json_out:
            output_path = pathlib.Path(args.json_out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return

    print(f"total_events={len(records)}")
    for source, source_records in sorted(by_source.items()):
        committed_wins = collections.Counter(record["winner_category"] for record in source_records)
        raw_wins = collections.Counter(record["raw_best_category"] for record in source_records)
        resample_group_wins = collections.Counter()
        for record in source_records:
            resample_categories = {
                category: candidate
                for category, candidate in record["best_by_category"].items()
                if category != "original"
            }
            if resample_categories:
                resample_group_wins[min(
                    resample_categories,
                    key=lambda category: resample_categories[category]["score"],
                )] += 1

        source_summary = {
            "events": len(source_records),
            "committed_wins": dict(committed_wins),
            "raw_overall_wins": dict(raw_wins),
            "equal_resample_group_wins": dict(resample_group_wins),
            "improvements_over_original": {},
        }
        summary["sources"][source] = source_summary

        print(f"{source}: events={len(source_records)}")
        print("  committed_wins:")
        for category, count in sorted(committed_wins.items()):
            print(f"  {category}: {count} ({count / len(source_records):.1%})")
        print("  raw_overall_wins:")
        for category, count in sorted(raw_wins.items()):
            print(f"  {category}: {count} ({count / len(source_records):.1%})")
        print("  equal_resample_group_wins:")
        resample_total = sum(resample_group_wins.values())
        for category, count in sorted(resample_group_wins.items()):
            print(f"  {category}: {count} ({count / resample_total:.1%})")
        for category in ("current_resample", "rollback_minus_1", "rollback_minus_2"):
            improvements = [
                record["original_score"] - record["best_by_category"][category]["score"]
                for record in source_records
                if record.get("original_score") is not None
                and category in record["best_by_category"]
            ]
            if improvements:
                source_summary["improvements_over_original"][category] = {
                    "mean": statistics.fmean(improvements),
                    "median": statistics.median(improvements),
                    "positive_rate": sum(value > 0.0 for value in improvements) / len(improvements),
                }
                print(
                    f"  {category}_improvement:"
                    f" mean={statistics.fmean(improvements):.6f}"
                    f" median={statistics.median(improvements):.6f}"
                    f" positive={sum(value > 0.0 for value in improvements) / len(improvements):.1%}"
                )

    if args.json_out:
        output_path = pathlib.Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
