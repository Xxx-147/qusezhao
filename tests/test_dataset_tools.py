from pathlib import Path

from tools.build_mixed_true_negative_dataset import discover_wechat_rows


def test_discover_wechat_rows_pairs_adjacent_images(tmp_path: Path) -> None:
    names = [
        "微信图片_20260605112543_250_3.jpg",
        "微信图片_20260605112545_251_3.jpg",
        "微信图片_20260605112546_252_3.jpg",
        "微信图片_20260605112547_253_3.jpg",
    ]
    for name in names:
        (tmp_path / name).touch()

    rows = discover_wechat_rows(tmp_path)

    assert len(rows) == 2
    assert rows[0]["negative"] == f"../../{names[0]}"
    assert rows[0]["target"] == f"../../{names[1]}"
    assert rows[1]["negative"] == f"../../{names[2]}"
    assert rows[1]["target"] == f"../../{names[3]}"
