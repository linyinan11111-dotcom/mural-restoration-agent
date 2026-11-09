"""Pluggable cloud image edit for the diffusion tool.

The default wire format is OpenAI-compatible JSON Images, aimed at xAI Grok
Imagine (`https://api.x.ai/v1`, `grok-imagine-image-2.0`). A different vendor
that speaks the same JSON edits API is selected by base URL and model name.
`openai_multipart` is the official OpenAI form upload, for hosts that do not
accept the JSON body.

These endpoints edit a whole picture; they are not a strict mask inpaint.
The masked pixels are blanked in the request, and the returned image is
resized and composited back so pixels outside the mask stay the source.
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import httpx
import numpy as np
from PIL import Image

from .imaging import composite

DEFAULT_PROVIDER = "openai_json"
DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-imagine-image-2.0"
MAX_SIDE = 1024

FULL_REDRAW_PROMPT = (
    "Reconstruct this damaged ancient mural photograph as a complete coherent mural painting. "
    "Keep the original subject, composition, visible painted lines and mineral pigments. "
    "Repair missing plaster and paint without inventing extra figures or symbols. "
    "Do not add text, watermark, frame, or modern objects. "
    "This is a hypothetical digital reconstruction, not a historical photograph."
)

PROVIDERS = {
    "openai_json": {
        "label": "OpenAI 兼容 JSON Images",
        "default_base_url": DEFAULT_BASE_URL,
        "default_model": DEFAULT_MODEL,
    },
    "openai_multipart": {
        "label": "OpenAI multipart Images",
        "default_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-image-1",
    },
}

_ALIASES = {
    "xai": "openai_json",
    "grok": "openai_json",
    "grok_imagine": "openai_json",
    "openai_images": "openai_json",
    "openai": "openai_multipart",
    "multipart": "openai_multipart",
}


def normalize_provider(value: str | None) -> str:
    key = (value or DEFAULT_PROVIDER).strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in PROVIDERS:
        known = "、".join(PROVIDERS)
        raise ValueError(f"云端生图协议须为 {known}。")
    return key


def validate_base_url(base: str | None) -> str:
    text = (base or "").strip().rstrip("/")
    if not text.startswith(("https://", "http://127.0.0.1", "http://localhost")):
        raise ValueError("云端生图请填写 HTTPS 服务根地址，或本机服务地址。")
    if " " in text or "\n" in text or "\r" in text:
        raise ValueError("云端生图服务根地址无效。")
    return text


def provider_defaults(provider: str | None) -> dict:
    spec = PROVIDERS[normalize_provider(provider)]
    return {"base_url": spec["default_base_url"], "model": spec["default_model"]}


@dataclass
class ImagineConfig:
    provider: str = DEFAULT_PROVIDER
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    timeout: float = 120

    @classmethod
    def from_settings(cls, settings) -> "ImagineConfig":
        provider = normalize_provider(getattr(settings, "imagine_provider", DEFAULT_PROVIDER))
        defaults = provider_defaults(provider)
        base = getattr(settings, "imagine_base_url", "") or defaults["base_url"]
        model = (getattr(settings, "imagine_model", "") or defaults["model"]).strip()
        return cls(
            provider=provider,
            base_url=validate_base_url(base),
            model=model,
            api_key=(getattr(settings, "imagine_api_key", "") or "").strip(),
            timeout=float(getattr(settings, "imagine_timeout", 120) or 120),
        )

    @property
    def ready(self) -> bool:
        return bool(self.api_key and self.base_url and self.model)


def _png_bytes(array: np.ndarray, mode: str = "RGB") -> bytes:
    image = Image.fromarray(np.asarray(array).astype(np.uint8), mode=mode)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def prepare_edit_inputs(original: np.ndarray, mask: np.ndarray, max_side: int = MAX_SIDE):
    """Blank the mask so the edit request shows a hole, and cap the upload size."""
    if mask.ndim != 2:
        raise ValueError("掩膜尺寸不正确。")
    if not (mask > 0).any():
        raise ValueError("请先圈选需要云端生图填补的区域。")
    marked = np.array(original, copy=True)
    marked[mask > 0] = (214, 206, 190)
    height, width = marked.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    scaled_mask = mask
    if scale < 1:
        resized = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        marked = np.asarray(Image.fromarray(marked).resize(resized, Image.Resampling.LANCZOS))
        scaled_mask = np.asarray(
            Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(resized, Image.Resampling.NEAREST))
    return marked, scaled_mask


def edit_prompt(user_prompt: str, mask: np.ndarray) -> str:
    ys, xs = np.where(mask > 0)
    height, width = mask.shape[:2]
    box = f"x {int(xs.min())}-{int(xs.max())}, y {int(ys.min())}-{int(ys.max())}, image {width}x{height}"
    note = (user_prompt or "").strip()
    text = (
        "Edit this aged mural photograph. A flat beige blank is the only damaged area "
        f"({box}). Reconstruct only that blank by continuing adjacent mineral pigment, "
        "weathered texture, and existing lines. Do not invent figures, symbols, inscriptions, "
        "or new ornaments. Do not change paint outside the blank. "
        "This is a hypothetical digital fill, not a historical reconstruction."
    )
    if note:
        text += " Conservator note: " + note[:800]
    return text[:2000]


def _openai_mask_png(mask: np.ndarray) -> bytes:
    """OpenAI edits: transparent pixels are the area to change."""
    height, width = mask.shape[:2]
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[mask == 0, 3] = 255
    return _png_bytes(rgba, mode="RGBA")


def _decode_image_bytes(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as image:
        return np.asarray(image.convert("RGB"))


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "***")
    return text


def _image_from_payload(payload: dict, fetch_url) -> np.ndarray:
    data = payload.get("data") or []
    if not data:
        message = ""
        error = payload.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or "")
        elif error:
            message = str(error)
        raise RuntimeError("云端生图没有返回图像。" + (f" {message[:200]}" if message else ""))
    item = data[0] or {}
    if item.get("b64_json"):
        return _decode_image_bytes(base64.b64decode(item["b64_json"]))
    url = item.get("url")
    if url:
        return _decode_image_bytes(fetch_url(url))
    raise RuntimeError("云端生图响应缺少 b64_json 或 url。")


def _resize(image: np.ndarray, shape) -> np.ndarray:
    height, width = shape[:2]
    if image.shape[0] == height and image.shape[1] == width:
        return image
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.LANCZOS))


class _HttpResult:
    def __init__(self, status_code: int, payload: dict | None = None, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.content = content
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("响应不是 JSON。")
        return self._payload


class ImagineClient:
    """Talks to `{base_url}/images/edits`. `transport` replaces HTTP in tests."""

    def __init__(self, config: ImagineConfig, transport=None):
        self.config = config
        self.transport = transport

    def _missing_key(self):
        if not self.config.api_key:
            raise ValueError("云端生图尚未配置 API Key。请在设置中填写服务根地址、模型名与密钥。")
        if not self.config.model:
            raise ValueError("云端生图尚未填写模型名。")

    def _request(self, method: str, url: str, *, json_body=None, data=None, files=None):
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        timeout = self.config.timeout
        if self.transport is not None:
            response = self.transport(method, url, headers=headers, json=json_body, data=data, files=files, timeout=timeout)
        else:
            with httpx.Client(timeout=timeout) as client:
                if method == "GET":
                    response = client.get(url, headers=headers)
                elif files is not None:
                    response = client.post(url, headers=headers, data=data, files=files)
                else:
                    response = client.post(url, headers={**headers, "Content-Type": "application/json"}, json=json_body)
        if getattr(response, "status_code", 200) >= 400:
            detail = ""
            try:
                body = response.json()
                error = body.get("error") if isinstance(body, dict) else None
                if isinstance(error, dict):
                    detail = str(error.get("message") or "")
                elif error:
                    detail = str(error)
            except Exception:
                detail = getattr(response, "text", "") or ""
            detail = _redact(detail[:240], self.config.api_key)
            raise RuntimeError(f"云端生图请求失败（HTTP {response.status_code}）。{detail}".rstrip())
        return response

    def _fetch_url(self, url: str) -> bytes:
        response = self._request("GET", url)
        return response.content

    def edit(self, original: np.ndarray, mask: np.ndarray, prompt: str = "") -> tuple[np.ndarray, dict]:
        """Return the provider image resized to the source. Does not composite."""
        self._missing_key()
        marked, scaled_mask = prepare_edit_inputs(original, mask)
        text = edit_prompt(prompt, scaled_mask)
        png = _png_bytes(marked)
        provider = normalize_provider(self.config.provider)
        url = self.config.base_url.rstrip("/") + "/images/edits"
        if provider == "openai_multipart":
            files = {
                "image": ("image.png", png, "image/png"),
                "mask": ("mask.png", _openai_mask_png(scaled_mask), "image/png"),
            }
            data = {"model": self.config.model, "prompt": text, "n": "1", "response_format": "b64_json"}
            response = self._request("POST", url, data=data, files=files)
        else:
            body = {
                "model": self.config.model,
                "prompt": text,
                "n": 1,
                "response_format": "b64_json",
                "image": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                    "type": "image_url",
                },
            }
            response = self._request("POST", url, json_body=body)
        generated = _resize(_image_from_payload(response.json(), self._fetch_url), original.shape)
        meta = {
            "provider": provider,
            "base_url": self.config.base_url,
            "model": self.config.model,
            "endpoint": "/images/edits",
            "device": "cloud",
            "note": "云端生图按整图编辑，不是严格掩膜 inpaint；调用方须把结果合成回掩膜内。",
        }
        return generated, meta

    def redraw(self, original: np.ndarray, prompt: str = "") -> tuple[np.ndarray, dict]:
        """Full-image reinterpretation from the intact original (no blank mask)."""
        self._missing_key()
        height, width = original.shape[:2]
        scale = min(1.0, MAX_SIDE / max(height, width))
        upload = original
        if scale < 1:
            upload = np.asarray(Image.fromarray(original).resize(
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                Image.Resampling.LANCZOS))
        png = _png_bytes(upload)
        text = (prompt or "").strip() or FULL_REDRAW_PROMPT
        provider = normalize_provider(self.config.provider)
        url = self.config.base_url.rstrip("/") + "/images/edits"
        if provider == "openai_multipart":
            files = {"image": ("image.png", png, "image/png")}
            data = {"model": self.config.model, "prompt": text[:2000],
                    "n": "1", "response_format": "b64_json"}
            response = self._request("POST", url, data=data, files=files)
        else:
            body = {
                "model": self.config.model,
                "prompt": text[:2000],
                "n": 1,
                "response_format": "b64_json",
                "image": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii"),
                    "type": "image_url",
                },
            }
            response = self._request("POST", url, json_body=body)
        generated = _resize(_image_from_payload(response.json(), self._fetch_url), original.shape)
        meta = {
            "provider": provider,
            "base_url": self.config.base_url,
            "model": self.config.model,
            "endpoint": "/images/edits",
            "device": "cloud",
            "mode": "full_redraw",
            "note": "云端按完整原图重绘整幅；结果为推测性重构，不是历史复原。",
        }
        return generated, meta


def redraw_with_imagine(settings, original, prompt: str = "", transport=None):
    """Full-image cloud redraw from the intact original; no mask, no compositing."""
    config = ImagineConfig.from_settings(settings)
    generated, meta = ImagineClient(config, transport=transport).redraw(original, prompt)
    return generated, meta


def restore_with_imagine(settings, original, mask, prompt: str = "", transport=None):
    """Edit in the cloud and composite the result into the mask only."""
    config = ImagineConfig.from_settings(settings)
    generated, meta = ImagineClient(config, transport=transport).edit(original, mask, prompt)
    filled = composite(np.asarray(original), generated, np.asarray(mask))
    meta = {**meta, "composited_into_mask": True}
    return filled, meta


def imagine_env_updates(values: dict) -> dict:
    """Env assignments for the cloud image settings. Blank key keeps the stored key."""
    provider = normalize_provider(values.get("provider"))
    defaults = provider_defaults(provider)
    base = validate_base_url(values.get("base_url") or defaults["base_url"])
    model = (values.get("model") or defaults["model"]).strip()
    if not model or any(char in model for char in "\r\n"):
        raise ValueError("请填写云端生图模型名。")
    fields = {
        "MURAL_IMAGINE_PROVIDER": provider,
        "MURAL_IMAGINE_BASE_URL": base,
        "MURAL_IMAGINE_MODEL": model,
    }
    key = values.get("key") or ""
    if any(char in key for char in "\r\n"):
        raise ValueError("配置不能包含换行。")
    if key:
        fields["MURAL_IMAGINE_API_KEY"] = key
    if any("\n" in value or "\r" in value for value in fields.values()):
        raise ValueError("配置不能包含换行。")
    return fields
