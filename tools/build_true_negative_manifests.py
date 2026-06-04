from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "datasets" / "true_negative_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [
        {"negative": "../../1-1.jpg", "target": "../../1-2.jpg", "source": "local true negative pair 1"},
        {"negative": "../../2-1.jpg", "target": "../../2-2.jpg", "source": "local true negative pair 2"},
        {
            "negative": "../../samples/public_input/wikimedia_negative_positive_negative.jpg",
            "target": "../../samples/public_reference/wikimedia_negative_positive_reference.jpg",
            "source": "Wikimedia Commons Public Domain true negative/positive",
        },
    ]
    test_rows = [
        {"negative": "../../3-1.jpg", "target": "../../3-2.jpg", "source": "local true negative pair 3 held out"},
    ]

    write_manifest(out_dir / "true_train_manifest.csv", train_rows)
    write_manifest(out_dir / "true_test_manifest.csv", test_rows)
    write_manifest(out_dir / "true_all_manifest.csv", train_rows + test_rows)
    print(out_dir)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["negative", "target", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows: {path}")


if __name__ == "__main__":
    main()
