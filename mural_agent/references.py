"""Local provenance-aware retrieval and conservative geometric registration."""
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
from .workspace import add_image
from .storage import new_id, timestamp
from .imaging import image_array, sha256


def add_reference(store, payload, name, metadata):
    required = ('source', 'license', 'relation', 'group')
    if any(not str(metadata.get(k, '')).strip() for k in required):
        raise ValueError('参考图必须填写来源、许可、对应关系和壁画分组。')
    if metadata['relation'] not in {'same_mural', 'tracing', 'related'}:
        raise ValueError('不支持的参考关系。')
    image = add_image(store, payload, name)
    return store.put('reference', {'id': new_id(), 'image_id': image['id'], 'name': name,
                                 'metadata': {k: str(v)[:2000] for k, v in metadata.items()}, 'created_at': timestamp()})


def thumbnail(array, limit=1200):
    scale = min(1., limit / max(array.shape[:2]))
    return cv2.resize(array, None, fx=scale, fy=scale) if scale < 1 else array, scale


def retrieve(store, image, query='', group=''):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    ranked = []
    terms = set(query.lower().split())
    for ref in store.list('reference'):
        meta = ref['metadata']
        if group and meta['group'] != group:
            continue
        array = image_array(store.folder(ref['image_id']) / 'image.png')
        rh = cv2.calcHist([cv2.cvtColor(array, cv2.COLOR_RGB2HSV)], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(rh, rh)
        score = float(cv2.compareHist(hist, rh, cv2.HISTCMP_CORREL))
        text = (' '.join(meta.values()) + ' ' + ref['name']).lower()
        score += .2 * sum(term in text for term in terms)
        ranked.append(dict(ref, retrieval_score=round(score, 4)))
    return sorted(ranked, key=lambda r: r['retrieval_score'], reverse=True)


def register(store, image_record, reference, damage_mask=None):
    if reference['metadata']['relation'] == 'related':
        raise ValueError('同题材参考不作为直接重建证据，请选择同一壁画或可信临摹。')
    target = image_array(store.folder(image_record['id']) / 'image.png')
    source = image_array(store.folder(reference['image_id']) / 'image.png')
    a, sa = thumbnail(source)
    b, sb = thumbnail(target)
    sift = cv2.SIFT_create(nfeatures=6000)
    kp1, d1 = sift.detectAndCompute(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), None)
    allowed = None
    if damage_mask is not None:
        allowed = cv2.resize((damage_mask == 0).astype(np.uint8)*255, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_NEAREST)
    kp2, d2 = sift.detectAndCompute(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), allowed)
    if d1 is None or d2 is None:
        raise ValueError('参考图可用特征不足，无法配准。')
    matches = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance]
    if len(good) < 12:
        raise ValueError('可靠匹配不足 12 个，拒绝直接重建。')
    p1 = np.float32([kp1[m.queryIdx].pt for m in good]) / sa
    p2 = np.float32([kp2[m.trainIdx].pt for m in good]) / sb
    homography, inliers = cv2.findHomography(p1, p2, cv2.RANSAC, 4./sb)
    if homography is None or inliers is None:
        raise ValueError('配准变换求解失败。')
    valid = inliers.ravel().astype(bool)
    transformed = cv2.perspectiveTransform(p1[:, None, :], homography)[:, 0]
    error = float(np.median(np.linalg.norm(transformed[valid] - p2[valid], axis=1)))
    hull = cv2.convexHull(p2[valid])
    spread = cv2.contourArea(hull) / (target.shape[0]*target.shape[1])
    if valid.sum() < 12 or valid.mean() < .35 or error > 4/sb or spread < .01:
        raise ValueError('匹配内点数量、空间分布或重投影误差不满足要求。')
    h, w = target.shape[:2]
    corners = np.float32([[0, 0], [source.shape[1], 0], [source.shape[1], source.shape[0]], [0, source.shape[0]]])
    warped_corners = cv2.perspectiveTransform(corners[None], homography)[0]
    area = abs(cv2.contourArea(warped_corners)) / (w*h)
    if not np.isfinite(warped_corners).all() or not .02 < area < 50 or not cv2.isContourConvex(warped_corners):
        raise ValueError('配准变换退化或范围异常。')
    aligned = cv2.warpPerspective(source, homography, (w, h))
    coverage = cv2.warpPerspective(np.full(source.shape[:2], 255, np.uint8), homography, (w, h), flags=cv2.INTER_NEAREST)
    coverage = cv2.erode(coverage, np.ones((5, 5), np.uint8))
    if damage_mask is not None and np.count_nonzero((damage_mask > 0) & (coverage > 0)) == 0:
        raise ValueError('参考图未覆盖圈选缺损。')
    identifier = new_id()
    folder = store.folder(identifier)
    Image.fromarray(aligned).save(folder / 'aligned.png')
    Image.fromarray(coverage).save(folder / 'coverage.png')
    blend = cv2.addWeighted(target, .5, aligned, .5, 0)
    Image.fromarray(blend).save(folder / 'overlay.png')
    return store.put('registration', {'id': identifier, 'image_id': image_record['id'],
        'reference_id': reference['id'], 'homography': homography.tolist(), 'inliers': int(valid.sum()),
        'inlier_ratio': float(valid.mean()), 'median_error_px': error, 'feature_coverage': float(spread),
        'approved': False, 'created_at': timestamp(), 'aligned_sha256': sha256(folder / 'aligned.png')})


def approve_registration(store, registration_id, payload, note):
    registration = store.get('registration', registration_id)
    if not registration:
        raise ValueError('配准记录不存在。')
    if not note.strip():
        raise ValueError('请填写核验意见。')
    from .imaging import decode_image, mask_array
    array = np.asarray(decode_image(payload, mask=True)) > 127
    coverage = mask_array(store.folder(registration_id) / 'coverage.png') > 0
    if array.shape != coverage.shape or not array.any() or (array & ~coverage).any():
        raise ValueError('核验区域必须非空且完全位于参考覆盖内。')
    Image.fromarray(array.astype(np.uint8)*255).save(store.folder(registration_id) / 'approved.png')
    return store.update('registration', registration_id, approved=True, review_note=note[:2000],
                        approved_mask_sha256=sha256(store.folder(registration_id) / 'approved.png'), approved_at=timestamp())
