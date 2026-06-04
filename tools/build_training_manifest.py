from __future__ import annotations

from pathlib import Path

from film_mask_automation.ml.data import PairRecord, discover_local_pairs, write_manifest


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    records = discover_local_pairs(project_root)
    output = project_root / "datasets" / "training_manifest.csv"
    manifest_root = output.parent
    relative_records = [
        PairRecord(
            negative=Path("..") / record.negative.relative_to(project_root),
            target=Path("..") / record.target.relative_to(project_root),
            source=record.source,
        )
        for record in records
    ]
    write_manifest(output, relative_records)
    print(f"wrote {len(records)} pairs to {output}")


if __name__ == "__main__":
    main()
