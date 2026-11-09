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


def _kind(value):
    """Normalize legacy/current region names to the canonical risk vocabulary."""
    try:
        from .risk_regions import normalize_region_kind
        return normalize_region_kind(value or "uncertain")
    except (ValueError, TypeError):
        return "uncertain"


def evaluate(doc: dict) -> dict:
    """Return gate + export_allowed for a review document."""
    reasons = []
    for c in doc.get("candidates", []):
        cid = c.get("id")
        decision = c.get("decision")
        kind = _kind(c.get("region_kind"))
        if decision == "pending":
            reasons.append(f"候选 {cid} 仍为 pending")
        if (c.get("uncertainty") == "irrecoverable" or kind == "unrestorable") and decision not in ABSTAIN_OK:
            reasons.append(f"不可复原区 {cid} 未 reject/abstain")
        if kind == "uncertain" and decision == "approve":
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
        root / "data" / "human-gate",
        root / "human-gate",
        root,
    ]


def resolve_gate_dir(root: Path | None = None) -> Path:
    candidates = []
    if root is not None:
        candidates.extend(human_gate_dirs(root))
    here = Path(__file__).resolve().parent
    candidates.extend([
        here.parent / "data" / "human-gate",
        here / "data" / "human-gate",
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
    kind = _kind(region_kind)
    if decision == "abstain":
        if kind == "uncertain":
            return "依据不足：高不确定区默认拒答补绘，仅保留标注"
        if kind == "unrestorable":
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
        kind = _kind(c.get("region_kind"))
        # Hard rule: unrestorable / high-uncertainty cannot be plain approve
        final = decision
        if kind == "unrestorable" and decision in {"approve", "correct"}:
            final = "abstain"
        if kind == "uncertain" and decision == "approve":
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
