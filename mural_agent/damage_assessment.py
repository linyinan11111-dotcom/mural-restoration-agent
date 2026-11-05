"""Local vision-model proposals for visible mural damage regions."""
from pathlib import Path
import json
import subprocess
import sys
import time

from PIL import Image


TYPES = {"crack", "paint_loss", "plaster_loss", "flaking", "stain", "other"}
SIZES = {"small", "medium", "large", "uncertain"}

# Conservative routing suggestion, not a conservation decision. Mirrors the
# small-crack study (Navier-Stokes best) and the paired study (LaMa better on
# medium/large losses, Telea better on fine cracks).
def recommendation_for(item):
    kind = item.get("type", "other")
    size = item.get("size", "uncertain")
    if kind == "stain":
        return {"backend": "color", "tool": "color", "method": "局部色调匹配"}
    if kind == "crack":
        if size == "large":
            return {"backend": "lama", "tool": "lama", "method": "LaMa 候选；长裂缝需人工复核"}
        return {"backend": "ns", "tool": "ns", "method": "Navier–Stokes 细线修补"}
    if kind in {"paint_loss", "plaster_loss"} and size == "large":
        return {"backend": "lama", "tool": "lama", "method": "LaMa 纹理补全"}
    if kind in {"paint_loss", "plaster_loss", "flaking"}:
        return {"backend": "telea", "tool": "telea", "method": "Telea 局部纹理传播"}
    return {"backend": "opencv", "tool": "telea", "method": "先人工确认，再选择保守修补"}


def _area(bbox):
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _overlap(a, b):
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, width) * max(0, height)


def merge_adjacent_candidates(items, gap=10):
    """Merge duplicate/near-duplicate boxes and join adjacent fragments.

    A local VLM often repeats the same candidate several times and also reports
    two parallel cracks that partially overlap. Identical/contained boxes are the
    same region; adjacent fragments (disjoint, gap <= `gap` px) belong together;
    partially overlapping but clearly different boxes stay separate.
    """
    merged = []
    for source in items:
        if not isinstance(source, dict) or source.get("type") not in TYPES:
            continue
        try:
            bbox = [max(0, min(1000, int(value))) for value in source["bbox"]]
            if len(bbox) != 4 or bbox[2]-bbox[0] < 2 or bbox[3]-bbox[1] < 2:
                continue
            item = {"type": source["type"], "bbox": bbox,
                    "size": source.get("size", "uncertain") if source.get("size") in SIZES else "uncertain",
                    "confidence": max(0., min(1., float(source.get("confidence", 0)))),
                    "evidence": str(source.get("evidence", ""))[:300]}
        except (TypeError, ValueError, KeyError):
            continue
        x0, y0, x1, y1 = item["bbox"]
        found = None
        for existing in merged:
            if existing["type"] != item["type"]:
                continue
            a0, b0, a1, b1 = existing["bbox"]
            overlap = _overlap(existing["bbox"], item["bbox"])
            if overlap:
                # Same region when one box mostly covers the other; otherwise keep
                # separate (two parallel cracks can partially overlap).
                if overlap / max(1, min(_area(existing["bbox"]), _area(item["bbox"]))) >= 0.6:
                    found = existing
                    break
                continue
            dx = max(0, a0-x1, x0-a1)
            dy = max(0, b0-y1, y0-b1)
            if dx <= gap and dy <= gap:
                found = existing
                break
        if found is None:
            merged.append(item)
            continue
        a0, b0, a1, b1 = found["bbox"]
        found["bbox"] = [min(a0, x0), min(b0, y0), max(a1, x1), max(b1, y1)]
        found["confidence"] = max(found["confidence"], item["confidence"])
        if item["size"] in {"medium", "large"}:
            found["size"] = item["size"]
        if item.get("evidence") and item["evidence"] not in found["evidence"]:
            found["evidence"] = (found["evidence"] + "；" + item["evidence"])[:300]
    for index, item in enumerate(merged, 1):
        item["index"] = index
    return merged


def parse_model_output(text):
    start = text.find("{")
    if start < 0:
        raise ValueError("视觉模型没有返回 JSON 对象。")
    decoder = json.JSONDecoder()
    try:
        data, _ = decoder.raw_decode(text[start:])
        if isinstance(data, dict):
            # Merge duplicates before capping: a VLM may repeat the same candidate
            # several times and fill the eight slots with duplicates.
            data["items"] = merge_adjacent_candidates(data.get("items", []))[:8]
            return data
    except json.JSONDecodeError:
        pass
    # Preserve only complete candidate objects when generation ends mid-JSON.
    items_at = text.find('"items"', start)
    array_start = text.find("[", items_at)
    if items_at < 0 or array_start < 0:
        raise ValueError("视觉模型返回的缺损清单无法解析。")
    items, offset = [], array_start + 1
    while offset < len(text):
        while offset < len(text) and text[offset] in " \t\r\n,":
            offset += 1
        if offset >= len(text) or text[offset] == "]":
            break
        if text[offset] != "{":
            break
        try:
            item, consumed = decoder.raw_decode(text[offset:])
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            items.append(item)
        offset += consumed
    if not items:
        raise ValueError("视觉模型输出被截断，未得到完整缺损候选。")
    return {"items": merge_adjacent_candidates(items)[:8],
            "limitations": "模型输出被截断；仅保留已完整解析的候选，需人工检查。"}


def assess_image(engine, image_record):
    image_path = engine.store.folder(image_record["id"]) / "image.png"
    if not (engine.settings.vision_model / "provenance.json").exists():
        raise ValueError("本地 Qwen 视觉模型未安装。")
    folder = engine.store.folder(image_record["id"])
    preview_path = folder / "damage-assessment-input.jpg"
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1024, 1024))
    image.save(preview_path, quality=90)
    request_path = folder / "damage-assessment-request.json"
    output_path = folder / "damage-assessment-output.txt"
    request_path.write_text(json.dumps({
        "model": str(engine.settings.vision_model),
        "device": engine.settings.device,
        "image": str(preview_path),
        "output": str(output_path),
        "prompt": (
            "Inspect this mural photograph for visible physical deterioration and cautiously describe "
            "its visible cultural context. "
            "Do not label intentional painted outlines, figures, ornament, pigment boundaries, "
            "or the normal edge of a detached fragment as cracks. Report distinct visible candidates "
            "such as cracks, paint loss, plaster loss, flaking, or stains. Use a tight bounding box "
            "for each candidate, normalized to integers from 0 to 1000 in [x0,y0,x1,y1] order. "
            "Estimate size as small, medium, large, or uncertain based on the visible affected region; "
            "estimate visual confidence from 0.0 to 1.0 and give one short visual evidence sentence. "
            "This is only a visual proposal, not a conservation diagnosis. It is acceptable to return "
            "no candidates when uncertain. Consolidate continuous or nearby parts of one crack into one item. "
            "Do not repeat the same crack with adjacent boxes; report at most 8 spatially distinct candidates. "
            "For cultural context, describe only visible subject matter, symbols, clothing, "
            "inscriptions, pigments, and composition. A possible period or tradition is a hypothesis, "
            "not a fact: give confidence, list visual clues, and say what needs source verification. "
            "Never invent an artifact identity, exact date, religion, historical event, or citation. "
            "State preservation constraints that follow from visible evidence, especially elements "
            "that must not be erased or reconstructed without references. "
            "Return only JSON exactly shaped as: "
            '{"items":[{"type":"crack|paint_loss|plaster_loss|flaking|stain|other",'
            '"bbox":[0,0,10,10],"size":"small|medium|large|uncertain",'
            '"confidence":0.0,"evidence":"..."}],'
            '"cultural_context":{"visible_subject":"...","possible_period_or_tradition":"...",'
            '"confidence":"low|medium|high","visual_clues":["..."],'
            '"verification_needed":["..."],"preservation_constraints":["..."]},'
            '"limitations":"..."}. '
            "Never invent damage hidden by the photograph."
        ),
    }, ensure_ascii=False), encoding="utf-8")
    engine.lama.unload()
    engine.sam.unload()
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with (folder / "damage-assessment.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [engine.settings.worker_python or sys.executable, "-m", "mural_agent.damage_assessment", str(request_path)],
            stdout=log, stderr=log, creationflags=flags)
        started = time.monotonic()
        try:
            while process.poll() is None:
                if time.monotonic() - started > 240:
                    process.terminate()
                    raise RuntimeError("壁画缺损识别超过 240 秒，已停止。")
                time.sleep(.1)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    if process.returncode:
        raise RuntimeError("本地视觉模型缺损识别失败；诊断信息保存在工作区记录中。")
    text = output_path.read_text(encoding="utf-8")
    data = parse_model_output(text)
    items = []
    for index, candidate in enumerate(data.get("items", [])[:8]):
        try:
            kind = str(candidate.get("type", "other")).strip().lower()
            kind = kind if kind in TYPES else "other"
            size = str(candidate.get("size", "uncertain")).strip().lower()
            size = size if size in SIZES else "uncertain"
            bbox = [max(0, min(1000, int(value))) for value in candidate["bbox"]]
            if len(bbox) != 4:
                continue
            x0, y0, x1, y1 = bbox
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            confidence = max(0., min(1., float(candidate.get("confidence", 0))))
            items.append({"index": index + 1, "type": kind, "bbox": bbox, "size": size,
                          "confidence": confidence, "evidence": str(candidate.get("evidence", ""))[:300],
                          "confirmed": False})
        except (TypeError, ValueError, KeyError):
            continue
    items = merge_adjacent_candidates(items)
    context = data.get("cultural_context") if isinstance(data.get("cultural_context"), dict) else {}
    def context_list(key):
        value = context.get(key, [])
        if not isinstance(value, list):
            return []
        return [str(entry)[:240] for entry in value[:8] if str(entry).strip()]
    cultural_context = {
        "visible_subject": str(context.get("visible_subject", "未能可靠辨认"))[:500],
        "possible_period_or_tradition": str(context.get("possible_period_or_tradition", "未判断"))[:500],
        "confidence": str(context.get("confidence", "low"))[:20],
        "visual_clues": context_list("visual_clues"),
        "verification_needed": context_list("verification_needed"),
        "preservation_constraints": context_list("preservation_constraints"),
        "verified": False,
    }
    return {"items": items, "cultural_context": cultural_context,
            "limitations": str(data.get("limitations", "模型判断不等于病害鉴定。"))[:1000],
            "model": "Qwen3-VL local", "confirmed": False}


def main():
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    request_path = Path(sys.argv[1]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    device = "cuda" if request["device"] != "cpu" and torch.cuda.is_available() else "cpu"
    if request["device"] == "cuda" and device != "cuda":
        raise RuntimeError("请求 CUDA，但 GPU 当前不可用。")
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        request["model"], local_files_only=True, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    processor = AutoProcessor.from_pretrained(request["model"], local_files_only=True)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": Image.open(request["image"]).convert("RGB")},
        {"type": "text", "text": request["prompt"]},
    ]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(device)
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=1600, do_sample=False)
    text = processor.batch_decode(result[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    Path(request["output"]).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
