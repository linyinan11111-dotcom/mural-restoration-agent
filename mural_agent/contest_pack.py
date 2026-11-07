"""Shipped competition observe pack (annotation only, no generative mural).

`data/` is gitignored, so the desktop loader must also find
`datasets/demo_competition`. When that tree is missing, a small synthetic
pack can be built at runtime for the click-through demo.
"""
from __future__ import annotations

import json
from pathlib import Path

from .human_gate import builtin_pending_document, builtin_reviewed_document


def competition_pack_dirs(root: Path) -> list[Path]:
    root = Path(root)
    return [
        root / "datasets" / "demo_competition",
        root / "data" / "demo_competition",
        root / "demo_competition",
    ]


def iter_competition_pack_dirs(root: Path | None = None) -> list[Path]:
    here = Path(__file__).resolve().parent
    repo = here.parent
    ordered: list[Path] = []
    if root is not None:
        ordered.extend(competition_pack_dirs(root))
    ordered.extend([
        repo / "datasets" / "demo_competition",
        repo / "data" / "demo_competition",
        Path.cwd() / "datasets" / "demo_competition",
        Path.cwd() / "data" / "demo_competition",
    ])
    seen = set()
    unique = []
    for path in ordered:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def pack_is_ready(path: Path) -> bool:
    return (path / "source.jpg").is_file() and (path / "regions.json").is_file()


def resolve_competition_pack(root: Path | None = None, *, create: bool = False) -> Path:
    candidates = iter_competition_pack_dirs(root)
    for path in candidates:
        if pack_is_ready(path):
            return path.resolve()
    if create:
        dest = None
        for path in candidates:
            if path.name == "demo_competition" and path.parent.name == "data":
                dest = path
                break
        dest = dest or (Path.cwd() / "data" / "demo_competition")
        return build_synthetic_pack(dest)
    looked = "、".join(str(path) for path in candidates[:4])
    raise FileNotFoundError(
        "未找到比赛观察包（需要 source.jpg 与 regions.json）。"
        f"已查找 datasets/demo_competition 与 data/demo_competition。候选：{looked}"
    )


def build_synthetic_pack(dest: Path) -> Path:
    """Write a tiny three-color observe pack. No restored mural is produced."""
    import numpy as np
    from PIL import Image, ImageDraw

    from .risk_regions import REGION_KINDS, render_risk_overlay

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    height, width = 320, 480
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[:] = (198, 156, 108)
    rgb[24:296, 18:36] = (92, 64, 46)
    rgb[24:40, 18:462] = (120, 78, 52)
    rgb[280:296, 18:462] = (120, 78, 52)
    for x in range(70, 380, 46):
        rgb[60:260, x:x + 3] = (92, 58, 42)
    kind_masks = {kind: np.zeros((height, width), np.uint8) for kind in REGION_KINDS}
    kind_masks["repairable"][70:110, 80:140] = 255
    kind_masks["uncertainty"][130:230, 180:320] = 255
    kind_masks["unrestorable"][40:280, 400:460] = 255
    overlay = render_risk_overlay(rgb, kind_masks)
    compare = np.concatenate([rgb, overlay], axis=1)
    legend = Image.fromarray(overlay.copy())
    draw = ImageDraw.Draw(legend)
    draw.rectangle((8, 8, 460, 78), fill=(25, 25, 25))
    draw.text((16, 14), "repairable orange / uncertainty red / unrestorable hatch", fill=(255, 236, 214))
    draw.text((16, 36), "observe only — no generative mural", fill=(255, 196, 186))
    draw.text((16, 54), "restored_image=false", fill=(210, 210, 210))

    Image.fromarray(rgb).save(dest / "source.jpg", quality=90)
    Image.fromarray(overlay).save(dest / "overlay_risk.png")
    Image.fromarray(compare).save(dest / "compare_source_overlay.png")
    legend.save(dest / "legend.png")
    regions = [
        {
            "id": "r1",
            "region_kind": "repairable",
            "class": "repairable",
            "class_alias": "repairable",
            "label": "可保守填的露底",
            "repair_allowed": True,
            "bbox_xywh": [80, 70, 60, 40],
            "evidence": "小块可见露底，周边有色块延续；可候选保守记录，禁止生成纹样。",
            "status": "proposed",
        },
        {
            "id": "r2",
            "region_kind": "uncertainty",
            "class": "uncertainty",
            "class_alias": "uncertainty",
            "label": "高不确定（须标红）",
            "repair_allowed": False,
            "bbox_xywh": [180, 130, 140, 100],
            "evidence": "大面积露底且周边无线稿延续；高不确定，默认不补绘。",
            "status": "proposed",
        },
        {
            "id": "r3",
            "region_kind": "unrestorable",
            "class": "unrestorable",
            "class_alias": "unrestorable",
            "label": "不可复原主张区（禁止补绘）",
            "repair_allowed": False,
            "bbox_xywh": [400, 40, 60, 240],
            "evidence": "残片外缘紧贴现代托底，无颜料依据；禁止想象补全。",
            "status": "proposed",
        },
    ]
    payload = {
        "policy": "competition_observe_only",
        "region_kind": "unrestorable",
        "blocking": True,
        "legend": {
            "repairable": "orange translucent",
            "uncertainty": "red translucent",
            "unrestorable": "red hatch",
        },
        "regions": regions,
        "notice": "观察标注包：橙=repairable，红半透明=uncertainty，红斜线=unrestorable。禁止生成纹样/图像。",
    }
    (dest / "regions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "README.md").write_text(
        "# demo_competition · 风险标注包\n\n"
        "正赛交付：**标注层 + 复核说明**，禁止生成壁画成图。\n\n"
        "- `source.jpg` 原图\n"
        "- `overlay_risk.png` 三色风险叠加\n"
        "- `compare_source_overlay.png` 原图与标注对照\n"
        "- `legend.png` 图例\n"
        "- `regions.json` 分区列表\n\n"
        "本目录没有修复成图。\n",
        encoding="utf-8",
    )
    return dest.resolve()


def write_gate_samples(dest: Path) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    pending = builtin_pending_document()
    reviewed = builtin_reviewed_document()
    (dest / "sample-from-shibi-pending.json").write_text(
        json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "sample-from-shibi-reviewed.json").write_text(
        json.dumps(reviewed, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest.resolve()
