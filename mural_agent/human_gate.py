"""Human-gate helpers for contest observe mode (no generative mural).

Mirrors deepseek validate_gate.evaluate and maps 拾壁 UI decisions onto
gate candidate decisions (approve/correct/reject/abstain).
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

BANNED_KINDS = {"lacuna_fill", "style_hint"}
ABSTAIN_OK = {"reject", "abstain"}
UI_TO_GATE = {
    "adopt": "approve",
    "approve": "approve",
    "modify": "correct",
    "correct": "correct",
    "reject": "reject",
    "abstain": "abstain",
}
GATE_TO_UI = {
    "approve": "adopt",
    "correct": "modify",
    "reject": "reject",
    "abstain": "abstain",
    "pending": "pending",
}
SHANGHAI = timezone(timedelta(hours=8))


def evaluate(doc: dict) -> dict:
    """Return gate + export_allowed for a review document."""
    reasons = []
    for c in doc.get("candidates", []):
        cid = c.get("id")
        decision = c.get("decision")
        if decision == "pending":
            reasons.append(f"候选 {cid} 仍为 pending")
        if (c.get("uncertainty") == "irrecoverable" or c.get("region_kind") == "unrestorable") and decision not in ABSTAIN_OK:
            reasons.append(f"不可复原区 {cid} 未 reject/abstain")
        if c.get("region_kind") == "uncertainty" and decision == "approve":
            reasons.append(f"高不确定区 {cid} 不可直接 approve 为可修，请 correct 后降级或 abstain/reject")
        if c.get("kind") in BANNED_KINDS and decision not in ABSTAIN_OK:
            reasons.append(f"禁生成：候选 {cid} kind={c.get('kind')} 须 reject/abstain")
        if decision == "correct" and not str(c.get("correction_note") or "").strip():
            reasons.append(f"候选 {cid} 为 correct 但缺少 correction_note")
        if decision == "reject" and not str(c.get("reject_reason") or "").strip():
            reasons.append(f"候选 {cid} 为 reject 但缺少 reject_reason")
        if decision == "abstain" and not str(c.get("reject_reason") or c.get("correction_note") or "").strip():
            reasons.append(f"候选 {cid} 为 abstain 但缺少拒答说明")
        if c.get("kind") == "literature_claim" and decision == "approve":
            evidence = str(c.get("evidence") or "")
            if not evidence.strip() or (
                "出处" not in evidence
                and "页" not in evidence
                and "ISBN" not in evidence
                and "著录" not in evidence
            ):
                reasons.append(f"文献候选 {cid} 批准前需可核对出处")
    flags = doc.get("request_flags") or {}
    if flags.get("wants_generative_restore"):
        reasons.append("禁生成：检测到生成/补绘修复请求，已拒答")
    if doc.get("stage") in {"enhanced_draft", "finalize"}:
        reasons.append(f"禁生成：stage={doc.get('stage')} 不允许进入正赛导出")
    if doc.get("stage") == "enhanced_draft":
        if not doc.get("gate", {}).get("require_risk_banner"):
            reasons.append("加强档必须 require_risk_banner=true")
        if not doc.get("reviewer", {}).get("confirmed"):
            reasons.append("加强档必须人工 confirmed")

    blocked = bool(reasons)
    confirmed = bool(doc.get("reviewer", {}).get("confirmed"))
    return {
        "blocked": blocked,
        "block_reasons": reasons,
        "next_stage_allowed": (not blocked) and confirmed and doc.get("stage") == "observe",
        "require_risk_banner": doc.get("stage") in {"enhanced_draft", "observe"} and any(
            c.get("uncertainty") in {"high", "irrecoverable"} for c in doc.get("candidates", [])
        ),
        "export_allowed": {
            "annotation_layer": (not blocked) and confirmed,
            "review_record": (not blocked) and confirmed,
            "restored_image": False,
        },
    }


def apply_evaluate(doc: dict) -> dict:
    """Mutate/return doc with gate + export_allowed filled from evaluate()."""
    gate = evaluate(doc)
    doc = copy.deepcopy(doc)
    doc["gate"] = {
        **doc.get("gate", {}),
        **{k: gate[k] for k in ("blocked", "block_reasons", "next_stage_allowed", "require_risk_banner")},
    }
    doc["export_allowed"] = gate["export_allowed"]
    return doc


def human_gate_dirs(root: Path) -> list[Path]:
    root = Path(root)
    return [
        root / "datasets" / "human-gate",
        root / "data" / "human-gate",
        root / "human-gate",
    ]


def resolve_gate_dir(root: Path | None = None) -> Path:
    candidates = []
    here = Path(__file__).resolve().parent
    repo = here.parent
    if root is not None:
        candidates.extend(human_gate_dirs(root))
    candidates.extend([
        repo / "datasets" / "human-gate",
        repo / "data" / "human-gate",
        repo / "human-gate",
        Path.cwd() / "datasets" / "human-gate",
        Path.cwd() / "data" / "human-gate",
    ])
    seen = set()
    for path in candidates:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_dir() and (
            (path / "sample-from-shibi-pending.evaluated.json").exists()
            or (path / "validate_gate.py").exists()
            or (path / "sample-from-shibi-pending.json").exists()
        ):
            return path.resolve()
    raise FileNotFoundError("未找到 data/human-gate 样例包。")


def load_sample(name: str, root: Path | None = None) -> dict:
    """Load a named sample JSON (with or without .evaluated / .json suffix)."""
    gate_dir = resolve_gate_dir(root)
    stem = name
    for suffix in (".evaluated.json", ".json", ""):
        if stem.endswith(suffix) and suffix:
            stem = stem[: -len(suffix)]
            break
    candidates = [
        gate_dir / f"{stem}.evaluated.json",
        gate_dir / f"{stem}.json",
        gate_dir / name,
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"未找到门禁样例：{name}（目录 {gate_dir}）")


def banner_text(doc: dict) -> str:
    gate = doc.get("gate") or {}
    export = doc.get("export_allowed") or {}
    if gate.get("blocked"):
        reasons = gate.get("block_reasons") or ["人工复核未完成"]
        short = "；".join(reasons[:3])
        more = f" 等{len(reasons)}项" if len(reasons) > 3 else ""
        return f"门禁阻断：{short}{more}。禁止导出修复成图。"
    flags = []
    if export.get("annotation_layer"):
        flags.append("标注层")
    if export.get("review_record"):
        flags.append("复核记录")
    allowed = "、".join(flags) if flags else "无"
    return (
        f"门禁已放行 · 可导出：{allowed}；"
        f"restored_image=false（禁止修复成图）。"
    )


def map_ui_decision(value: str | None) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    if not key:
        return None
    if key not in UI_TO_GATE:
        raise ValueError("复核决定须为 采纳/修改后采纳/驳回/拒答（adopt/modify/reject/abstain）。")
    return UI_TO_GATE[key]


def default_note_for(decision: str, region_kind: str | None, note: str) -> str:
    text = (note or "").strip()
    if text:
        return text
    if decision == "abstain":
        if region_kind == "uncertainty":
            return "依据不足：高不确定区默认拒答补绘，仅保留标注"
        if region_kind == "unrestorable":
            return "不可复原区：禁修禁生成，拒答补绘"
        return "依据不足 / 涉及补绘生成 → 已拒答，仅保留原状记录。"
    if decision == "reject":
        return "人工驳回该标注候选。"
    if decision == "correct":
        return "人工改正标注后采纳（观察层，不成图）。"
    return ""


def apply_bulk_decision(
    doc: dict,
    *,
    ui_decision: str,
    note: str = "",
    checklist: dict | None = None,
    reviewer_name: str = "",
    confirm: bool = True,
    candidate_ids: list | None = None,
) -> dict:
    """Apply one UI decision across candidates (or selected ids) and re-evaluate."""
    decision = map_ui_decision(ui_decision)
    if not decision:
        raise ValueError("请选择复核决定：采纳 / 修改后采纳 / 驳回 / 拒答。")
    out = copy.deepcopy(doc)
    out.setdefault("stage", "observe")
    out.setdefault("request_flags", {"wants_generative_restore": False})
    targets = set(str(x) for x in candidate_ids) if candidate_ids else None
    for c in out.get("candidates", []):
        if targets is not None and str(c.get("id")) not in targets:
            continue
        kind = c.get("region_kind")
        # Hard rule: unrestorable / high-uncertainty cannot be plain approve
        final = decision
        if kind == "unrestorable" and decision in {"approve", "correct"}:
            final = "abstain"
        if kind == "uncertainty" and decision == "approve":
            final = "abstain"
        c["decision"] = final
        note_text = default_note_for(final, kind, note)
        if final in {"reject", "abstain"}:
            c["reject_reason"] = note_text
            if final == "abstain" and not c.get("correction_note"):
                c["correction_note"] = ""
        elif final == "correct":
            c["correction_note"] = note_text or "人工改正标注后采纳"
            c["reject_reason"] = ""
        else:
            c["correction_note"] = ""
            c["reject_reason"] = ""
    checklist = checklist or {}
    out["checklist"] = {
        "repairable_checked": bool(checklist.get("repairable_checked")),
        "uncertainty_checked": bool(checklist.get("uncertainty_checked")),
        "unrestorable_checked": bool(checklist.get("unrestorable_checked")),
        "literature_checked": bool(checklist.get("literature_checked")),
    }
    reviewer = dict(out.get("reviewer") or {})
    reviewer.setdefault("role", "文物修复人员")
    if reviewer_name:
        reviewer["name"] = reviewer_name
    reviewer["confirmed"] = bool(confirm)
    if confirm:
        reviewer["confirmed_at"] = datetime.now(SHANGHAI).isoformat(timespec="seconds")
    out["reviewer"] = reviewer
    out = apply_evaluate(out)
    # Extra soft block: required checklist items for observe export
    missing = []
    if not out["checklist"]["repairable_checked"]:
        missing.append("repairable 未勾选已核")
    if not out["checklist"]["uncertainty_checked"]:
        missing.append("uncertainty 未勾选已核")
    if not out["checklist"]["unrestorable_checked"]:
        missing.append("unrestorable 未勾选已核")
    if not out["checklist"]["literature_checked"]:
        missing.append("文献条目未勾选已核/待查")
    if missing and not out["gate"]["blocked"]:
        # Only soft-block when otherwise clear — checklist is contest UI hard gate
        out["gate"]["blocked"] = True
        out["gate"]["block_reasons"] = list(out["gate"].get("block_reasons") or []) + missing
        out["gate"]["next_stage_allowed"] = False
        out["export_allowed"] = {
            "annotation_layer": False,
            "review_record": False,
            "restored_image": False,
        }
    return out


def response_payload(doc: dict) -> dict:
    return {
        "gate": doc.get("gate") or {},
        "export_allowed": doc.get("export_allowed") or {
            "annotation_layer": False,
            "review_record": False,
            "restored_image": False,
        },
        "candidates": doc.get("candidates") or [],
        "checklist": doc.get("checklist") or {},
        "reviewer": doc.get("reviewer") or {},
        "banner": banner_text(doc),
        "gate_review": doc,
        "restored_image": False,
    }


# Shipped click-through samples. Files under datasets/human-gate mirror these
# so a fresh clone does not depend on the gitignored data/ directory.
DEMO_CHECKLIST_CLEAR = {
    "repairable_checked": True,
    "uncertainty_checked": True,
    "unrestorable_checked": True,
    "literature_checked": True,
}


def builtin_pending_document() -> dict:
    """Unreviewed observe sample: every candidate is still pending."""
    return {
        "run_id": "sample-from-shibi-pending",
        "stage": "observe",
        "request_flags": {"wants_generative_restore": False},
        "candidates": [
            {
                "id": "r1",
                "kind": "damage_annotation",
                "region_kind": "repairable",
                "bbox_xywh": [80, 70, 60, 40],
                "evidence": "小块可见露底，周边有色块延续；可候选保守记录，禁止生成纹样。",
                "uncertainty": "medium",
                "decision": "pending",
                "correction_note": "",
                "reject_reason": "",
            },
            {
                "id": "r2",
                "kind": "damage_annotation",
                "region_kind": "uncertainty",
                "bbox_xywh": [180, 130, 140, 100],
                "evidence": "大面积露底且周边无线稿延续；高不确定，默认不补绘。",
                "uncertainty": "high",
                "decision": "pending",
                "correction_note": "",
                "reject_reason": "",
            },
            {
                "id": "r3",
                "kind": "damage_annotation",
                "region_kind": "unrestorable",
                "bbox_xywh": [400, 40, 60, 240],
                "evidence": "残片外缘紧贴现代托底，无颜料依据；禁止想象补全。",
                "uncertainty": "irrecoverable",
                "decision": "pending",
                "correction_note": "",
                "reject_reason": "",
            },
            {
                "id": "lit1",
                "kind": "literature_claim",
                "region_kind": "repairable",
                "bbox_xywh": [],
                "evidence": "出处：示范著录摘录，供人工核对，不构成病害鉴定。",
                "uncertainty": "medium",
                "decision": "pending",
                "correction_note": "",
                "reject_reason": "",
            },
        ],
        "checklist": {
            "repairable_checked": False,
            "uncertainty_checked": False,
            "unrestorable_checked": False,
            "literature_checked": False,
        },
        "reviewer": {
            "role": "文物修复人员",
            "name": "",
            "confirmed": False,
            "confirmed_at": "",
        },
    }


def builtin_reviewed_document() -> dict:
    """Reviewed observe sample: export annotation + review record, never a mural."""
    doc = apply_bulk_decision(
        builtin_pending_document(),
        ui_decision="adopt",
        note="演示复核：三色标注已核对，不生成、不补绘。",
        checklist=DEMO_CHECKLIST_CLEAR,
        reviewer_name="演示复核",
        confirm=True,
    )
    reviewer = dict(doc.get("reviewer") or {})
    reviewer["confirmed"] = True
    reviewer["confirmed_at"] = "2026-11-06T10:00:00+08:00"
    reviewer["name"] = reviewer.get("name") or "演示复核"
    doc["reviewer"] = reviewer
    doc["run_id"] = "sample-from-shibi-reviewed"
    return apply_evaluate(doc)


def candidates_from_regions(regions: list, *, run_id: str = "observe") -> dict:
    """Build a pending gate document from a competition regions.json list."""
    doc = {
        "run_id": run_id,
        "stage": "observe",
        "request_flags": {"wants_generative_restore": False},
        "candidates": [],
        "reviewer": {
            "role": "文物修复人员",
            "name": "",
            "confirmed": False,
            "confirmed_at": "",
        },
    }
    for region in regions or []:
        klass = (
            region.get("region_kind")
            or region.get("class")
            or region.get("region_class")
            or region.get("class_alias")
            or "uncertainty"
        )
        try:
            from .risk_regions import normalize_region_kind
            klass = normalize_region_kind(klass)
        except ValueError:
            klass = "uncertainty"
        unc = {"repairable": "medium", "uncertainty": "high", "unrestorable": "irrecoverable"}.get(klass, "high")
        doc["candidates"].append({
            "id": region.get("id"),
            "kind": "damage_annotation",
            "region_kind": klass,
            "bbox_xywh": region.get("bbox_xywh") or region.get("bbox") or [],
            "evidence": region.get("evidence") or "",
            "uncertainty": unc,
            "decision": "pending",
            "correction_note": "",
            "reject_reason": "",
        })
    return apply_evaluate(doc)


def load_builtin_sample(name: str) -> dict:
    stem = (name or "").lower()
    if "pending" in stem:
        return apply_evaluate(builtin_pending_document())
    if "generative" in stem or "restore" in stem:
        doc = builtin_reviewed_document()
        doc["request_flags"] = {"wants_generative_restore": True}
        doc["stage"] = "enhanced_draft"
        return apply_evaluate(doc)
    return builtin_reviewed_document()


OBSERVE_ZIP_FILES = (
    "overlay_risk.png",
    "compare_source_overlay.png",
    "legend.png",
    "regions.json",
)
_FORBIDDEN_EXPORT_NAMES = {
    "result.png",
    "raw.png",
    "difference.png",
    "restored.png",
    "restored_image.png",
    "generative.png",
    "final.png",
    "source.bin",
}


def forbidden_export_name(name: str) -> bool:
    base = Path(str(name)).name.lower()
    if base in _FORBIDDEN_EXPORT_NAMES:
        return True
    if "restored" in base or base.startswith("generative"):
        return True
    return False


def observe_blocks_restored_export(doc: dict | None) -> bool:
    """Observe / generative sessions must not ship a restored mural."""
    if not doc:
        return False
    export = doc.get("export_allowed") or {}
    if export.get("restored_image"):
        return True
    stage = doc.get("stage") or "observe"
    if stage in {"observe", "enhanced_draft", "finalize"}:
        return True
    flags = doc.get("request_flags") or {}
    return bool(flags.get("wants_generative_restore"))


def require_observe_export(doc: dict, kind: str) -> None:
    """Raise unless this observe document may export annotation or review only."""
    export = doc.get("export_allowed") or {}
    gate = doc.get("gate") or {}
    if export.get("restored_image"):
        raise ValueError("比赛禁止导出修复成图。")
    if doc.get("stage") in {"enhanced_draft", "finalize"}:
        raise ValueError(f"禁生成：stage={doc.get('stage')} 不允许进入正赛导出")
    flags = doc.get("request_flags") or {}
    if flags.get("wants_generative_restore"):
        raise ValueError("禁生成：检测到生成/补绘修复请求，已拒答")
    if gate.get("blocked") or not export.get(kind):
        if kind == "annotation_layer":
            raise ValueError("门禁未放行：不能导出标注层。请完成人工复核（pending/不可复原未处理会阻断）。")
        raise ValueError("门禁未放行：不能导出复核记录。")


def annotation_export_source(doc: dict, asset_dir: Path) -> Path:
    require_observe_export(doc, "annotation_layer")
    folder = Path(asset_dir)
    for name in ("overlay_risk.png", "compare_source_overlay.png", "legend.png"):
        src = folder / name
        if src.is_file():
            return src
    raise ValueError("未找到观察标注图，请先载入比赛观察包。")


def write_observe_zip(dest: Path, doc: dict, asset_dir: Path | None = None) -> dict:
    """Write an observe archive that cannot contain a generative mural."""
    import zipfile
    require_observe_export(doc, "review_record")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    readme = (
        "拾壁比赛观察导出\n"
        "- 仅含标注层 + 复核记录 + 说明\n"
        "- restored_image=false：不含修复成图\n"
        "- AI 标注非病害鉴定结论\n"
    )
    written = []
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("gate_review.json", json.dumps(doc, ensure_ascii=False, indent=2))
        written.append("gate_review.json")
        zipped.writestr("README.txt", readme)
        written.append("README.txt")
        if asset_dir is not None:
            folder = Path(asset_dir)
            for name in OBSERVE_ZIP_FILES:
                src = folder / name
                if src.is_file() and not forbidden_export_name(name):
                    zipped.write(src, arcname=name)
                    written.append(name)
    with zipfile.ZipFile(dest) as zipped:
        leaked = [name for name in zipped.namelist() if forbidden_export_name(name)]
        if leaked:
            dest.unlink(missing_ok=True)
            raise ValueError("导出含修复成图，已阻断：" + ",".join(leaked))
    return {"saved": True, "restored_image": False, "files": written}
