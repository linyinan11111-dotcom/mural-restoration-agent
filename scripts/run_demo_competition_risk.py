# -*- coding: utf-8 -*-
"""Competition annotation export only — NO generative mural image."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(r"D:\lyn\壁画修复agent")
if not (ROOT / "mural_agent").exists():
    ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo_competition"
DOCS = Path(os.path.expanduser(r"~\Documents\demo_competition"))
BOX = Path("/workspace/demo_competition")
SRC_CANDIDATES = [
    ROOT / "data" / "demo_step" / "source.jpg",
    Path(r"C:\Users\lin\Documents\demo_step\source.jpg"),
    BOX / "source.jpg",
]

sys.path.insert(0, str(ROOT))


def find_source() -> Path:
    for p in SRC_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError("source.jpg missing")


def main():
    if "--synthetic" in sys.argv or not any(path.is_file() for path in SRC_CANDIDATES):
        from mural_agent.contest_pack import build_synthetic_pack, write_gate_samples
        pack = build_synthetic_pack(ROOT / "datasets" / "demo_competition")
        gates = write_gate_samples(ROOT / "datasets" / "human-gate")
        print("synthetic pack", pack)
        print("gate samples", gates)
        return
    from mural_agent.risk_regions import (
        classify_mask_regions, render_risk_overlay, detect_mount_mask, REGION_LABELS,
    )
    src = find_source()
    OUT.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(Image.open(src).convert("RGB"))
    Image.fromarray(rgb).save(OUT / "source.jpg", quality=95)

    mask_path = ROOT / "data" / "demo_step" / "mask.png"
    if mask_path.is_file():
        lacuna = np.asarray(Image.open(mask_path).convert("L"))
        if lacuna.shape[:2] != rgb.shape[:2]:
            lacuna = np.asarray(
                Image.fromarray(lacuna).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST))
    else:
        import cv2
        lab = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)
        L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
        mount = detect_mount_mask(rgb) > 0
        lacuna = ((L > 120) & (L < 175) & (B > 138) & ~mount).astype(np.uint8) * 255
        lacuna = cv2.morphologyEx(lacuna, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        lacuna = cv2.morphologyEx(lacuna, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    risk = classify_mask_regions(rgb, lacuna)
    overlay = render_risk_overlay(rgb, risk["kind_masks"])
    legend = Image.fromarray(overlay.copy())
    draw = ImageDraw.Draw(legend)
    draw.rectangle((8, 8, 520, 92), fill=(25, 25, 25))
    draw.text((16, 12), "REGION_KINDS risk overlay (no generative mural)", fill=(255, 240, 230))
    draw.text((16, 34), "orange=repairable  red-fill=uncertainty  red-hatch=unrestorable", fill=(255, 210, 180))
    draw.text((16, 56), "Competition export: annotation + review only", fill=(200, 200, 200))
    draw.text((16, 74), "禁止生成纹样/图像", fill=(255, 180, 180))
    legend.save(OUT / "overlay_risk.png")

    regions = []
    for r in risk["regions"]:
        regions.append({
            "id": r["id"],
            "region_kind": r["region_kind"],
            "label": r["label"],
            "area_px": r["area_px"],
            "area_frac": r["area_frac"],
            "bbox_xywh": r["bbox_xywh"],
            "mount_touch": r["mount_touch"],
            "line_score": r["line_score"],
            "evidence": r["evidence"],
            "status": r["status"],
            "repair_allowed": r["repair_allowed"],
        })
    payload = {
        "policy": "正赛禁止生成壁画；本目录仅标注层+说明，无修复成图。",
        "region_kind": risk["region_kind"],
        "blocking": risk["blocking"],
        "mount_fraction": risk["mount_fraction"],
        "lacuna_fraction": risk["lacuna_fraction"],
        "legend": {
            "repairable": "orange translucent",
            "uncertainty": "red translucent",
            "unrestorable": "red hatch",
        },
        "regions": regions,
        "notice": risk["notice"],
        "source": str(src),
    }
    (OUT / "regions.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = """# demo_competition · 风险标注包

正赛交付：**标注层 + 复核说明**，禁止生成壁画成图。

## 文件
- `overlay_risk.png` — 风险叠加
  - **橙**：`repairable` 可保守填露底
  - **红半透明**：`uncertainty` 高不确定
  - **红斜线**：`unrestorable` 不可复原主张区
- `regions.json` — 分区列表与依据
- `source.jpg` — 原图副本
- `README.md` — 本说明

## 门禁摘要
`uncertainty` / `unrestorable` 默认禁止 create_job / enhanced / speculative；可确认可填、拒答跳过，或专家强制放行并记档。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    Image.fromarray((lacuna > 0).astype(np.uint8) * 255).save(OUT / "lacuna_mask.png")

    for dest in (DOCS, BOX):
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("overlay_risk.png", "regions.json", "README.md", "source.jpg", "lacuna_mask.png"):
                shutil.copy2(OUT / name, dest / name)
            print("copied ->", dest)
        except Exception as exc:
            print("copy skip", dest, exc)
    print("OK", OUT, "kind=", risk["region_kind"], "n=", len(regions))


if __name__ == "__main__":
    main()
