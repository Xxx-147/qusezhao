from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


SAMPLES = {
    "negative_positive_picture.jpg": "https://upload.wikimedia.org/wikipedia/commons/0/0c/Negative_Positive-Picture.jpg",
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_dir = project_root / "samples" / "public_sources"
    input_dir = project_root / "samples" / "public_input"
    reference_dir = project_root / "samples" / "public_reference"
    source_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    for filename, url in SAMPLES.items():
        target = source_dir / filename
        if not target.exists():
            download(url, target)
        if filename == "negative_positive_picture.jpg":
            split_negative_positive_picture(target, input_dir, reference_dir)
        print(target)


def split_negative_positive_picture(path: Path, input_dir: Path, reference_dir: Path) -> None:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    third = width // 3
    negative = image.crop((0, 0, third, height))
    positive = image.crop((third * 2, 0, width, height))
    negative.save(input_dir / "wikimedia_negative_positive_negative.jpg")
    positive.save(reference_dir / "wikimedia_negative_positive_reference.jpg")


def download(url: str, target: Path) -> None:
    request = Request(
        url,
        headers={
            "User-Agent": "film-mask-automation/0.1 research test sample downloader",
        },
    )
    with urlopen(request, timeout=30) as response:
        target.write_bytes(response.read())


if __name__ == "__main__":
    main()
