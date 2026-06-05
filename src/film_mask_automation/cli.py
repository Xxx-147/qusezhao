from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .profile import apply_color_profile, fit_color_profile, load_color_profile, save_color_profile
from .processor import ConversionParams, convert_file
from .smart import convert_image_smart

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="film-mask-auto",
        description="Remove color negative film orange masks and convert scans to positives.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert_parser = subparsers.add_parser("convert", help="Convert one negative image.")
    convert_parser.add_argument("input", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.add_argument("--profile", type=Path, help="Apply a saved color profile after negative conversion.")
    convert_parser.add_argument("--smart-auto", action="store_true", help="Try multiple conversion presets and pick the most stable one.")
    convert_parser.add_argument("--ai-model", type=Path, help="Use a trained AI model checkpoint for conversion.")
    convert_parser.add_argument("--no-ai-enhance", action="store_true", help="Disable conservative color restoration after AI inference.")
    convert_parser.add_argument("--no-ai-hybrid", action="store_true", help="Disable smart-rule anchoring after AI inference.")
    _add_processing_args(convert_parser)

    batch_parser = subparsers.add_parser("batch", help="Convert all supported images in a folder.")
    batch_parser.add_argument("input_dir", type=Path)
    batch_parser.add_argument("output_dir", type=Path)
    batch_parser.add_argument("--diagnostics-json", type=Path, help="Write batch diagnostics to this JSON file.")
    batch_parser.add_argument("--profile", type=Path, help="Apply a saved color profile after negative conversion.")
    batch_parser.add_argument("--smart-auto", action="store_true", help="Try multiple conversion presets for each image.")
    batch_parser.add_argument("--ai-model", type=Path, help="Use a trained AI model checkpoint for each image.")
    batch_parser.add_argument("--no-ai-enhance", action="store_true", help="Disable conservative color restoration after AI inference.")
    batch_parser.add_argument("--no-ai-hybrid", action="store_true", help="Disable smart-rule anchoring after AI inference.")
    _add_processing_args(batch_parser)

    calibrate_parser = subparsers.add_parser("calibrate", help="Fit a reusable color profile from a negative/reference pair.")
    calibrate_parser.add_argument("negative", type=Path)
    calibrate_parser.add_argument("reference", type=Path)
    calibrate_parser.add_argument("profile_json", type=Path)
    calibrate_parser.add_argument("--preview-output", type=Path)
    _add_processing_args(calibrate_parser)

    args = parser.parse_args()
    params = _params_from_args(args)

    if args.command == "convert":
        if args.ai_model:
            from .ml.inference import convert_with_model

            args.output.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(args.input) as image:
                output = convert_with_model(
                    image,
                    args.ai_model,
                    enhance=not args.no_ai_enhance,
                    hybrid_anchor=not args.no_ai_hybrid,
                )
                output.save(args.output)
                diagnostics = {
                    "ai_model": str(args.ai_model),
                    "ai_enhance": not args.no_ai_enhance,
                    "ai_hybrid_anchor": not args.no_ai_hybrid,
                }
        elif args.smart_auto:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(args.input) as image:
                result = convert_image_smart(image)
                result.image.save(args.output)
                diagnostics = result.diagnostics
        else:
            diagnostics = convert_file(args.input, args.output, params)
        if args.profile:
            profile = load_color_profile(args.profile)
            with Image.open(args.output) as image:
                apply_color_profile(image, profile).save(args.output)
        print(json.dumps({"output": str(args.output), "diagnostics": diagnostics}, indent=2, ensure_ascii=False))
        return

    if args.command == "calibrate":
        payload = _calibrate_profile(args.negative, args.reference, args.profile_json, args.preview_output, params)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    results = _convert_batch(
        args.input_dir,
        args.output_dir,
        params,
        args.profile,
        args.smart_auto,
        args.ai_model,
        ai_enhance=not args.no_ai_enhance,
        ai_hybrid_anchor=not args.no_ai_hybrid,
    )
    payload = {"count": len(results), "files": results}
    if args.diagnostics_json:
        args.diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostics_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _add_processing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mask-source", choices=["auto", "border", "percentile", "manual"], default="auto")
    parser.add_argument("--mask-rgb", help="Manual mask RGB as R,G,B in 0-255, for example 230,145,75.")
    parser.add_argument("--border-fraction", type=float, default=0.06)
    parser.add_argument("--black-percentile", type=float, default=0.5)
    parser.add_argument("--white-percentile", type=float, default=99.5)
    parser.add_argument("--reference-exponent", type=float, default=1.0)
    parser.add_argument("--red-ratio", type=float, default=1.0)
    parser.add_argument("--blue-ratio", type=float, default=1.0)
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("--brightness", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.08)
    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tint", type=float, default=0.0)
    parser.add_argument("--red-gain", type=float, default=1.0)
    parser.add_argument("--green-gain", type=float, default=1.0)
    parser.add_argument("--blue-gain", type=float, default=1.0)
    parser.add_argument("--white-balance", choices=["grayworld", "none"], default="grayworld")
    parser.add_argument("--sharpen", type=float, default=0.0)


def _params_from_args(args: argparse.Namespace) -> ConversionParams:
    manual_mask = None
    if args.mask_rgb:
        parts = [float(part.strip()) for part in args.mask_rgb.split(",")]
        if len(parts) != 3:
            raise ValueError("--mask-rgb must contain exactly three comma-separated numbers")
        manual_mask = (parts[0], parts[1], parts[2])

    return ConversionParams(
        mask_source=args.mask_source,
        manual_mask_rgb=manual_mask,
        border_fraction=args.border_fraction,
        black_percentile=args.black_percentile,
        white_percentile=args.white_percentile,
        reference_exponent=args.reference_exponent,
        red_ratio=args.red_ratio,
        blue_ratio=args.blue_ratio,
        exposure=args.exposure,
        brightness=args.brightness,
        gamma=args.gamma,
        contrast=args.contrast,
        saturation=args.saturation,
        temperature=args.temperature,
        tint=args.tint,
        red_gain=args.red_gain,
        green_gain=args.green_gain,
        blue_gain=args.blue_gain,
        white_balance=args.white_balance,
        sharpen=args.sharpen,
    )


def _convert_batch(
    input_dir: Path,
    output_dir: Path,
    params: ConversionParams,
    profile_path: Path | None = None,
    smart_auto: bool = False,
    ai_model_path: Path | None = None,
    ai_enhance: bool = True,
    ai_hybrid_anchor: bool = True,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = load_color_profile(profile_path) if profile_path else None
    results: list[dict[str, object]] = []
    for input_path in sorted(input_dir.iterdir()):
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        output_path = output_dir / f"{input_path.stem}_positive{input_path.suffix}"
        try:
            if ai_model_path:
                from .ml.inference import convert_with_model

                with Image.open(input_path) as image:
                    output = convert_with_model(
                        image,
                        ai_model_path,
                        enhance=ai_enhance,
                        hybrid_anchor=ai_hybrid_anchor,
                    )
                    output.save(output_path)
                    diagnostics = {
                        "ai_model": str(ai_model_path),
                        "ai_enhance": ai_enhance,
                        "ai_hybrid_anchor": ai_hybrid_anchor,
                    }
            elif smart_auto:
                with Image.open(input_path) as image:
                    result = convert_image_smart(image)
                    result.image.save(output_path)
                    diagnostics = result.diagnostics
            else:
                diagnostics = convert_file(input_path, output_path, params)
            if profile:
                with Image.open(output_path) as image:
                    apply_color_profile(image, profile).save(output_path)
            results.append(
                {
                    "input": str(input_path),
                    "output": str(output_path),
                    "status": "ok",
                    "diagnostics": diagnostics,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "input": str(input_path),
                    "output": str(output_path),
                    "status": "error",
                    "error": str(exc),
                }
            )
    return results


def _calibrate_profile(
    negative_path: Path,
    reference_path: Path,
    profile_path: Path,
    preview_output: Path | None,
    params: ConversionParams,
) -> dict[str, object]:
    preview_path = preview_output or profile_path.with_suffix(".preview.jpg")
    diagnostics = convert_file(negative_path, preview_path, params)
    with Image.open(preview_path) as converted, Image.open(reference_path) as reference:
        profile = fit_color_profile(converted, reference)
        corrected = apply_color_profile(converted, profile)
        corrected.save(preview_path)

    save_color_profile(profile, profile_path)
    return {
        "profile": str(profile_path),
        "preview_output": str(preview_path),
        "conversion_diagnostics": diagnostics,
    }


if __name__ == "__main__":
    main()
