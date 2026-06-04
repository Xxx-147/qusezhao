from pathlib import Path

from tests.test_processor import make_synthetic_negative


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "samples" / "input"
    output_dir.mkdir(parents=True, exist_ok=True)
    make_synthetic_negative().save(output_dir / "synthetic_negative.jpg")
    print(output_dir / "synthetic_negative.jpg")


if __name__ == "__main__":
    main()
