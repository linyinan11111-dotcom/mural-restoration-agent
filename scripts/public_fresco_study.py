"""Paired inpainting study on openly licensed mural photographs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mural_agent.config import Settings
from mural_agent.imaging import composite, image_array, opencv_inpaint, sha256
from mural_agent.models import LamaBackend
from mural_agent.quality import paired_metrics
from mural_agent.datasets import synthetic_damage


def source_records():
    records = []
    smoke = ROOT / "datasets/public-fresco-smoke"
    for entry in (json.loads(line) for line in (smoke / "images.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()):
        data = json.loads((smoke / (Path(entry["image"]).stem + ".metadata.json")).read_text(encoding="utf-8-sig"))
        technique = data.get("technique", "")
        culture = data.get("culture", "")
        creation_date = data.get("creation_date", "")
        entry.update(path=str(smoke / entry["image"]), sha256=sha256(smoke / entry["image"]),
                     date=creation_date, culture=culture, medium=technique,
                     citation=entry["source"],
                     observed_damage="博物馆藏品残片照片；可见表面磨蚀/缺失，未作专业分级。")
        records.append(entry)

    folder = ROOT / "artifacts/mural-study/images"
    extras = [
        {"image": "ajanta-jataka-cave17.jpg", "title": "Ajanta Cave 17 Jataka mural", "date": "约5世纪", "culture": "印度阿旃陀佛教石窟", "subject": "本生故事群像", "source": "https://commons.wikimedia.org/wiki/File:Ajanta_murals_depicting_Jatakas_in_Cave_17.jpg", "license": "CC0", "observed_damage": "画面整体尚连续，局部褪色、斑驳与细节磨蚀；目视低至中等，非专家分级。"},
        {"image": "egypt-nebamun-hunt.jpg", "title": "Tomb of Nebamun: hunting in the marshes", "date": "约公元前1350年", "culture": "古埃及第十八王朝", "subject": "墓葬沼泽狩猎", "source": "https://commons.wikimedia.org/wiki/File:Tomb_of_Nebamun.jpg", "license": "Public domain", "observed_damage": "博物馆保存的墓画残片，边缘和若干内部区域缺失；目视高等，非专家分级。"},
        {"image": "dunhuang-dancer.jpg", "title": "Dunhuang Mogao Caves dancer", "date": "唐代（文件页未给出窟号/精确年代）", "culture": "中国敦煌莫高窟", "subject": "舞者与装饰图案", "source": "https://commons.wikimedia.org/wiki/File:Dunhuang_Mogao_Caves_Dancer.jpg", "license": "Public domain", "observed_damage": "主体轮廓大体连续，颜料褪色/磨蚀和局部裂损可见；目视中等，非专家分级。"},
    ]
    for record in extras:
        path = folder / record["image"]
        record.update(path=str(path), sha256=sha256(path), category="real_damage")
        records.append(record)
    return records


def paired_row(target, output, mask, perceptual):
    return paired_metrics(target, output, mask, perceptual)


def save_comparison(target, damaged, mask, telea, lama, label, output_path):
    width, height = 2040, 410
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    panels = [(target, "Original"), (damaged, "Damaged input"), (mask, "Combined mask"),
              (telea, "OpenCV Telea"), (lama, "LaMa")]
    delta = np.abs(target.astype(np.int16) - lama.astype(np.int16)).max(axis=2)
    heat = np.zeros_like(target)
    heat[..., 0] = np.clip(delta * 4, 0, 255).astype(np.uint8)
    heat[..., 1] = np.clip(delta, 0, 100).astype(np.uint8)
    heat[mask == 0] = (238, 238, 238)
    panels.append((heat, "Difference from original (masked area)"))
    draw.text((8, 7), label, fill="black")
    for index, (image, title) in enumerate(panels):
        draw.text((index * 340 + 8, 29), title, fill="black")
        if title == "Combined mask":
            image = Image.fromarray(np.asarray(image.convert("L"))).convert("RGB")
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        else:
            image = image.copy()
        image.thumbnail((330, 350), Image.Resampling.LANCZOS)
        canvas.paste(image, (index * 340 + 5, 52))
    canvas.save(output_path, quality=94)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 runs all images")
    parser.add_argument("--lpips", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / "artifacts/mural-study"
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    records = source_records()
    if args.limit:
        records = records[:args.limit]
    settings = Settings.from_env()
    settings.data_dir = output_dir / "isolated-workspace"
    settings.device = "cuda"
    backend = LamaBackend(settings)
    if not backend.available():
        raise RuntimeError("LaMa model or PyTorch unavailable")
    results = []
    severities = [("light", 0.02), ("moderate", 0.10)]
    kinds = ("crack", "hole", "loss")
    for image_index, record in enumerate(records):
        target = image_array(record["path"])
        for severity_index, (severity, fraction) in enumerate(severities):
            for kind_index, kind in enumerate(kinds):
                seed = 20261031 + image_index * 100 + severity_index * 10 + kind_index
                damaged, mask, parameters = synthetic_damage(target, seed=seed, kind=kind, fraction=fraction)
                case_id = f"{image_index+1:02}-{severity}-{kind}"
                case_dir = cases_dir / case_id
                case_dir.mkdir(exist_ok=True)
                Image.fromarray(mask).save(case_dir / "mask.png", optimize=True)
                Image.fromarray(damaged).save(case_dir / "damaged.png", optimize=True)
                for method in ("opencv_telea", "lama"):
                    started = time.perf_counter()
                    if method == "opencv_telea":
                        prediction = opencv_inpaint(damaged, mask, method="telea", radius=3)
                    else:
                        prediction = backend.predict(damaged, mask)
                    restored = composite(damaged, prediction, mask)
                    elapsed = time.perf_counter() - started
                    Image.fromarray(restored).save(case_dir / f"{method}.png", optimize=True)
                    metrics = paired_row(target, restored, mask, args.lpips)
                    changed_outside = bool(np.any(restored[mask == 0] != target[mask == 0]))
                    results.append({
                        "case_id": case_id, "image_id": image_index + 1,
                        "title": record["title"], "source": record["source"],
                        "source_sha256": record["sha256"], "license": record["license"],
                        "date": record.get("date", ""), "culture": record.get("culture", ""),
                        "subject": record.get("subject", record.get("title", "")),
                        "observed_damage": record.get("observed_damage", ""),
                        "mask_kind": kind, "severity": severity,
                        "requested_fraction": fraction, "actual_fraction": parameters["actual_fraction"],
                        "seed": seed, "method": method, "elapsed_seconds": round(elapsed, 3),
                        "outside_pixels_changed": changed_outside, **metrics,
                    })
                    print(f"{len(results)}/{len(records)*len(severities)*len(kinds)*2} {case_id} {method} {elapsed:.2f}s", flush=True)
                    (output_dir / "results.json").write_text(json.dumps({
                        "protocol": "same source pixels, fixed synthetic mask and seed per method; metrics use only masked MAE/PSNR and context ROI SSIM/optional LPIPS",
                        "limitations": ["Synthetic masks do not model all natural deterioration.", "Metrics measure recovery of deliberately hidden photographed pixels, not original historical appearance.", "No conservator ratings or expert damage labels were collected.", "Source photographs differ in resolution, framing and photographic conditions."],
                        "records": records, "rows": results,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = list(results[0]) if results else []
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Add a single mural with three simultaneous, spatially separated damage types.
    mixed_record = next(record for record in records if record["title"] == "Tomb of Nebamun: hunting in the marshes")
    target = image_array(mixed_record["path"])
    height, width = target.shape[:2]
    mask_image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_image)
    draw.line([(int(width*x), int(height*y)) for x, y in
               [(0.15, 0.12), (0.22, 0.25), (0.19, 0.40), (0.26, 0.55), (0.23, 0.72), (0.31, 0.88)]],
              fill=255, width=max(7, round(min(width, height)*0.014)), joint="curve")
    draw.ellipse((int(width*0.70), int(height*0.13), int(width*0.85), int(height*0.30)), fill=255)
    draw.polygon([(int(width*0.64), int(height*0.62)), (int(width*0.82), int(height*0.59)),
                  (int(width*0.86), int(height*0.78)), (int(width*0.73), int(height*0.84)),
                  (int(width*0.62), int(height*0.75))], fill=255)
    mixed_mask = np.asarray(mask_image).copy()
    mixed_damaged = target.copy()
    mixed_damaged[mixed_mask > 0] = [224, 211, 179]
    mixed_dir = cases_dir / "mixed-nebamun-three-damage-types"
    mixed_dir.mkdir(exist_ok=True)
    Image.fromarray(target).save(mixed_dir / "original.png", optimize=True)
    Image.fromarray(mixed_mask).save(mixed_dir / "mask.png", optimize=True)
    Image.fromarray(mixed_damaged).save(mixed_dir / "damaged.png", optimize=True)
    mixed_results = []
    outputs = {}
    for method in ("opencv_telea", "lama"):
        started = time.perf_counter()
        prediction = (opencv_inpaint(mixed_damaged, mixed_mask, method="telea", radius=3)
                      if method == "opencv_telea" else backend.predict(mixed_damaged, mixed_mask))
        restored = composite(mixed_damaged, prediction, mixed_mask)
        elapsed = time.perf_counter() - started
        outputs[method] = restored
        Image.fromarray(restored).save(mixed_dir / f"{method}.png", optimize=True)
        metrics = paired_row(target, restored, mixed_mask, args.lpips)
        mixed_results.append({
            "case_id": "mixed-nebamun-three-damage-types", "title": mixed_record["title"],
            "source": mixed_record["source"], "source_sha256": mixed_record["sha256"],
            "license": mixed_record["license"], "mask_types": ["crack", "hole", "loss"],
            "actual_fraction": float((mixed_mask > 0).mean()), "method": method,
            "elapsed_seconds": round(elapsed, 3),
            "outside_pixels_changed": bool(np.any(restored[mixed_mask == 0] != target[mixed_mask == 0])),
            **metrics,
        })
    (output_dir / "multiple-damage.json").write_text(json.dumps({
        "protocol": "One source image with three separate synthetic damage regions in the same mask: a winding crack, an elliptical hole, and an irregular loss polygon. Same mask is restored by both methods.",
        "source_record": mixed_record, "rows": mixed_results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    save_comparison(target, mixed_damaged, Image.fromarray(mixed_mask), outputs["opencv_telea"],
                    outputs["lama"], "Nebamun mural: crack + hole + flaking/loss on one image",
                    output_dir / "multi-damage-comparison.jpg")
    # A second direct before/after comparison on the earlier Kizil example.
    single_dir = cases_dir / "03-moderate-loss"
    single_target = image_array(records[2]["path"])
    Image.fromarray(single_target).save(single_dir / "original.png", optimize=True)
    single_damaged = image_array(single_dir / "damaged.png")
    single_mask = Image.open(single_dir / "mask.png").convert("L")
    save_comparison(single_target, single_damaged, single_mask,
                    image_array(single_dir / "opencv_telea.png"), image_array(single_dir / "lama.png"),
                    "Kizil bodhisattva: original vs one synthetic loss repair",
                    output_dir / "original-restoration-comparison.jpg")
    (output_dir / "environment.json").write_text(json.dumps({
        "python": sys.version, "platform": sys.platform,
        "numpy": np.__version__, "device": str(backend.device),
        "model_path": str(settings.lama_model), "model_sha256": sha256(settings.lama_model),
        "source_image_count": len(records), "case_count": len(records)*len(severities)*len(kinds),
        "method_count": 2, "perceptual_lpips": args.lpips,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
