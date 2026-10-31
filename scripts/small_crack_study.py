"""Compare local repair methods on short, thin synthetic mural cracks."""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mural_agent.config import Settings
from mural_agent.imaging import composite, image_array, opencv_inpaint, sha256
from mural_agent.models import LamaBackend
from public_fresco_study import source_records


def small_crack_metrics(target, output, mask):
    active = mask > 0
    delta = target.astype(np.float32) - output.astype(np.float32)
    selected = delta[active]
    mse = float(np.mean(selected**2))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(active.astype(np.uint8), 8)
    ssim_values = []
    height, width = active.shape
    from skimage.metrics import structural_similarity
    for component_id in range(1, component_count):
        x, y, box_w, box_h, area = stats[component_id]
        if area < 2:
            continue
        pad = 24
        x0, y0 = max(0, x-pad), max(0, y-pad)
        x1, y1 = min(width, x+box_w+pad), min(height, y+box_h+pad)
        # Keep tiny crack crops large enough for a stable local SSIM window.
        if x1-x0 < 64:
            extra = 64-(x1-x0)
            x0, x1 = max(0, x0-extra//2), min(width, x1+(extra+1)//2)
        if y1-y0 < 64:
            extra = 64-(y1-y0)
            y0, y1 = max(0, y0-extra//2), min(height, y1+(extra+1)//2)
        truth_crop, output_crop = target[y0:y1, x0:x1], output[y0:y1, x0:x1]
        window = min(7, min(truth_crop.shape[:2]))
        window -= window % 2 == 0
        if window >= 3:
            ssim_values.append(float(structural_similarity(truth_crop, output_crop,
                                                            channel_axis=2, data_range=255,
                                                            win_size=window)))
    return {"mask_mae": float(np.mean(np.abs(selected))),
            "mask_psnr": 10*np.log10(255**2/mse) if mse else None,
            "perfect_match": mse == 0,
            "roi_ssim": float(np.mean(ssim_values)) if ssim_values else None,
            "ssim_protocol": "unweighted mean of per-connected-crack crop SSIM; each crop includes up to 24px context and is expanded to 64px minimum",
            "roi_lpips": None}


def make_short_cracks(shape, seed):
    height, width = shape
    rng = random.Random(seed)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    diagonal = (width**2 + height**2) ** 0.5
    crack_count = rng.randint(7, 11)
    widths = []
    lengths = []
    for _ in range(crack_count):
        x = rng.randint(max(0, width // 20), max(width // 20, width - width // 20))
        y = rng.randint(max(0, height // 20), max(height // 20, height - height // 20))
        length = rng.uniform(0.025, 0.06) * diagonal
        angle = rng.uniform(0, 2 * np.pi)
        points = [(x, y)]
        segments = rng.randint(3, 5)
        for part in range(1, segments + 1):
            angle += rng.uniform(-0.58, 0.58)
            step = length / segments * rng.uniform(0.75, 1.25)
            x = int(np.clip(x + np.cos(angle) * step, 0, width - 1))
            y = int(np.clip(y + np.sin(angle) * step, 0, height - 1))
            points.append((x, y))
        line_width = 1 if min(width, height) < 1100 or rng.random() < 0.7 else 2
        draw.line(points, fill=255, width=line_width, joint="curve")
        widths.append(line_width)
        lengths.append(round(length, 2))
        if rng.random() < 0.35:
            branch_start = points[rng.randint(1, len(points) - 2)]
            bx, by = branch_start
            branch_angle = angle + rng.choice((-1, 1)) * rng.uniform(0.55, 1.15)
            branch_length = length * rng.uniform(0.18, 0.35)
            end = (int(np.clip(bx + np.cos(branch_angle) * branch_length, 0, width - 1)),
                   int(np.clip(by + np.sin(branch_angle) * branch_length, 0, height - 1)))
            draw.line([branch_start, end], fill=255, width=line_width, joint="curve")
    array = np.asarray(mask).copy()
    return array, {"seed": seed, "crack_count": crack_count, "widths_px": widths,
                   "lengths_px": lengths, "actual_fraction": float((array > 0).mean()),
                   "damage_color_rgb": [92, 78, 63]}


def save_panel(record, case_dir, output_path):
    target = Image.open(record["path"]).convert("RGB")
    damaged = Image.open(case_dir / "damaged.png").convert("RGB")
    mask = Image.open(case_dir / "mask.png").convert("L")
    panels = [(target, "Original"), (damaged, "Short cracks"), (mask, "Mask"),
              (Image.open(case_dir / "opencv_telea.png").convert("RGB"), "Telea"),
              (Image.open(case_dir / "opencv_ns.png").convert("RGB"), "Navier-Stokes"),
              (Image.open(case_dir / "lama.png").convert("RGB"), "LaMa")]
    scale_size = (240, 300)
    canvas = Image.new("RGB", (6 * 250, 345), "white")
    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), f"{record['title']} | original, synthetic thin cracks and restorations", fill="black")
    for index, (image, title) in enumerate(panels):
        draw.text((index * 250 + 6, 26), title, fill="black")
        if title == "Mask":
            image = image.convert("RGB")
        image.thumbnail(scale_size, Image.Resampling.LANCZOS)
        canvas.paste(image, (index * 250 + 4, 46))
    canvas.save(output_path, quality=94)


def save_zoom_panel(record, case_dir, output_path):
    original = image_array(case_dir / "original.png")
    damaged = image_array(case_dir / "damaged.png")
    mask_image = Image.open(case_dir / "mask.png").convert("L")
    mask = np.asarray(mask_image)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    components = sorted(stats[1:component_count], key=lambda item: item[cv2.CC_STAT_AREA], reverse=True)[:3]
    methods = ["opencv_telea", "opencv_ns", "lama"]
    output_images = {method: image_array(case_dir / f"{method}.png") for method in methods}
    canvas = Image.new("RGB", (6*330, 3*280), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ["Original", "Crack input", "Mask", "Telea", "Navier-Stokes", "LaMa"]
    for column, title in enumerate(titles):
        draw.text((column*330+6, 4), title, fill="black")
    height, width = mask.shape
    for row, stats_row in enumerate(components):
        x, y, box_w, box_h = (int(stats_row[i]) for i in range(4))
        pad_x, pad_y = max(70, box_w), max(70, box_h)
        x0, x1 = max(0, x-pad_x), min(width, x+box_w+pad_x)
        y0, y1 = max(0, y-pad_y), min(height, y+box_h+pad_y)
        crop_mask = mask_image.crop((x0, y0, x1, y1)).resize((320, 225), Image.Resampling.NEAREST)
        arrays = [original[y0:y1, x0:x1], damaged[y0:y1, x0:x1], np.asarray(crop_mask.convert("RGB"))]
        arrays.extend(output_images[method][y0:y1, x0:x1] for method in methods)
        for column, array in enumerate(arrays):
            image = Image.fromarray(array.astype(np.uint8))
            image.thumbnail((320, 225), Image.Resampling.LANCZOS)
            canvas.paste(image, (column*330+4, row*280+24))
    draw.text((8, 3*280-20), f"{record['title']} | close-up of three generated short cracks", fill="black")
    canvas.save(output_path, quality=95)


def main():
    output_dir = ROOT / "artifacts/mural-study/small-cracks"
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    records = source_records()
    settings = Settings.from_env()
    settings.device = "cuda"
    settings.data_dir = output_dir / "isolated-workspace"
    backend = LamaBackend(settings)
    if not backend.available():
        raise RuntimeError("LaMa model or PyTorch unavailable")
    rows = []
    seeds = (17, 43, 91)
    total = len(records) * len(seeds) * 3
    index = 0
    for image_index, record in enumerate(records):
        original = image_array(record["path"])
        exemplar_case = None
        for seed_index, seed_base in enumerate(seeds):
            seed = seed_base + image_index * 1000
            mask, parameters = make_short_cracks(original.shape[:2], seed)
            damaged = original.copy()
            damaged[mask > 0] = parameters["damage_color_rgb"]
            case_id = f"{image_index+1:02}-seed-{seed}"
            case_dir = cases_dir / case_id
            case_dir.mkdir(exist_ok=True)
            Image.fromarray(original).save(case_dir / "original.png", optimize=True)
            Image.fromarray(damaged).save(case_dir / "damaged.png", optimize=True)
            Image.fromarray(mask).save(case_dir / "mask.png", optimize=True)
            if seed_index == 0:
                exemplar_case = case_dir
            for method in ("opencv_telea", "opencv_ns", "lama"):
                started = time.perf_counter()
                if method == "lama":
                    prediction = backend.predict(damaged, mask)
                else:
                    prediction = opencv_inpaint(original=damaged, mask=mask,
                                                method="telea" if method.endswith("telea") else "ns", radius=3)
                restored = composite(damaged, prediction, mask)
                elapsed = time.perf_counter() - started
                Image.fromarray(restored).save(case_dir / f"{method}.png", optimize=True)
                metrics = small_crack_metrics(original, restored, mask)
                row = {"case_id": case_id, "image_index": image_index + 1,
                       "title": record["title"], "source": record["source"],
                       "source_sha256": record["sha256"], "license": record["license"],
                       "date": record.get("date", ""), "culture": record.get("culture", ""),
                       "seed": seed, **parameters, "method": method,
                       "elapsed_seconds": round(elapsed, 4),
                       "outside_pixels_changed": bool(np.any(restored[mask == 0] != original[mask == 0])),
                       **metrics}
                rows.append(row)
                index += 1
                (output_dir / "results.json").write_text(json.dumps({
                    "protocol": "7-11 randomly placed short thin cracks per image, three fixed seed variants per source image, same mask shared by LaMa/Telea/Navier-Stokes; original photograph is paired pixel target",
                    "limitations": ["Synthetic crack paths are not expert damage annotations.", "Pixel metrics measure recovery of hidden photograph pixels, not historical authenticity.", "No conservator ratings were collected."],
                    "rows": rows,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"{index}/{total} {case_id} {method} {elapsed:.2f}s", flush=True)
        save_panel(record, exemplar_case, output_dir / f"example-{image_index+1:02}.jpg")
        save_zoom_panel(record, exemplar_case, output_dir / f"example-{image_index+1:02}-zoom.jpg")

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import statistics
    summary = {}
    for method in ("opencv_telea", "opencv_ns", "lama"):
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            "n": len(subset),
            "mask_mae_mean": round(statistics.mean(row["mask_mae"] for row in subset), 4),
            "mask_psnr_mean_db": round(statistics.mean(row["mask_psnr"] for row in subset), 4),
            "roi_ssim_mean": round(statistics.mean(row["roi_ssim"] for row in subset), 5),
            "mean_seconds": round(statistics.mean(row["elapsed_seconds"] for row in subset), 4),
            "mean_mask_percent": round(100 * statistics.mean(row["actual_fraction"] for row in subset), 5),
            "outside_changed": sum(row["outside_pixels_changed"] for row in subset),
        }
    summary["case_count"] = len({row["case_id"] for row in rows})
    summary["source_count"] = len(records)
    summary["seeds_per_source"] = len(seeds)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps({
        "device": str(backend.device), "model_sha256": sha256(settings.lama_model),
        "image_count": len(records), "seed_count": len(seeds), "method_count": 3,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
