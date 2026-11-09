from io import BytesIO
from pathlib import Path
import hashlib
import warnings
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 20_000_000
MAX_BYTES = 35 * 1024 * 1024


def decode_image(payload: bytes, mask=False) -> Image.Image:
    if not payload or len(payload) > MAX_BYTES:
        raise ValueError('图片不能为空，且须小于 35 MB。')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError('图片最多支持 2000 万像素，请先裁剪。')
                if image.format not in {'PNG', 'JPEG', 'WEBP', 'TIFF', 'BMP'}:
                    raise ValueError('请选择 PNG、JPEG、WebP、TIFF 或 BMP 图片。')
                image.load()
                image = ImageOps.exif_transpose(image)
                if mask and 'A' in image.getbands():
                    background = Image.new('RGBA', image.size, (0, 0, 0, 255))
                    image = Image.alpha_composite(background, image.convert('RGBA'))
                return image.convert('L' if mask else 'RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError('无法读取图片，文件损坏或尺寸过大。') from exc


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).copy()


def mask_array(path):
    with Image.open(path) as image:
        return (np.asarray(image.convert('L')) > 127).astype(np.uint8) * 255


def composite(original, prediction, mask):
    if original.shape != prediction.shape or mask.shape != original.shape[:2]:
        raise ValueError('图像与掩膜尺寸不一致。')
    if np.array_equal(original, prediction):
        return original.copy()
    active = mask > 0
    if not active.any():
        return original.copy()
    output = original.copy()
    distance = cv2.distanceTransform(active.astype(np.uint8), cv2.DIST_L2, 3)
    # Thin masks (hairline cracks) must be replaced at full strength; feathering
    # them would erase most of the repair. Thicker masks blend only in a narrow
    # boundary band and keep the prediction untouched inside.
    if float(distance[active].max()) < 4.0:
        output[active] = np.clip(prediction[active], 0, 255).astype(np.uint8)
        return output
    alpha = np.clip((distance - 1.0) / 3.0, 0, 1)[..., None]
    blended = original.astype(np.float32) * (1 - alpha) + np.asarray(prediction, dtype=np.float32) * alpha
    output[active] = np.clip(blended[active], 0, 255).astype(np.uint8)
    return output


def evaluate(original, output, mask):
    edited = mask > 0
    delta = np.abs(original.astype(np.int16) - output.astype(np.int16))
    outside = delta[~edited]
    ring = cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0
    ring &= ~edited
    inside_boundary = edited & (cv2.erode(mask, np.ones((3, 3), np.uint8)) == 0)
    # A heuristic for sorting candidates, not an estimate of historic authenticity.
    if ring.any() and inside_boundary.any():
        seam = float(np.linalg.norm(output[inside_boundary].mean(0) - output[ring].mean(0)))
    else:
        seam = 0.0
    return {
        'outside_max_difference': int(outside.max()) if outside.size else 0,
        'outside_unchanged': bool(not outside.size or outside.max() == 0),
        'masked_fraction': round(float(edited.mean()), 6),
        'changed_pixels': int(np.any(delta > 0, axis=2).sum()),
        'boundary_color_gap': round(seam, 3),
        'note': '边界色差仅作启发式参考，不代表历史真实性。',
    }


def make_difference(original, output):
    delta = np.abs(original.astype(np.int16) - output.astype(np.int16)).max(axis=2)
    diff = np.zeros_like(original)
    diff[..., 0] = np.clip(delta * 4, 0, 255)
    diff[..., 1] = np.clip(delta, 0, 100)
    return diff


def opencv_inpaint(original, mask, method='telea', radius=3):
    # OpenCV receives BGR, while the rest of this application uses RGB.
    bgr = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    flags = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    repaired = cv2.inpaint(bgr, mask, radius, flags)
    return cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)

