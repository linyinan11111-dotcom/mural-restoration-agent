"""Engineering checks and explicitly scoped, paired-image metrics."""
import cv2
import numpy as np
from .imaging import evaluate


def assess(original, output, mask, thresholds=None):
    metrics = evaluate(original, output, mask)
    active = mask > 0
    distances = []
    gradients = []
    gray = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY).astype(np.float32)
    for axis in (0, 1):
        # Only pairs sharing an edge across the edit boundary are compared.
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        crossing = active[left] != active[right]
        diff = output[left].astype(float) - output[right].astype(float)
        distances.extend(np.linalg.norm(diff[crossing], axis=-1).tolist())
        gradients.extend(np.abs(gray[left] - gray[right])[crossing].tolist())
    metrics['boundary_pair_delta'] = round(float(np.mean(distances)), 3) if distances else 0.
    metrics['boundary_gradient'] = round(float(np.mean(gradients)), 3) if gradients else 0.
    metrics['finite'] = bool(np.isfinite(output).all())
    limits = {'boundary_pair_delta': 65., 'boundary_gradient': 45.}
    limits.update(thresholds or {})
    issues = []
    if not metrics['outside_unchanged']:
        issues.append('outside_changed')
    if not metrics['finite']:
        issues.append('nonfinite')
    if metrics['changed_pixels'] == 0:
        issues.append('no_change')
    for name, limit in limits.items():
        if name in metrics and metrics[name] > limit:
            issues.append(name)
    metrics.update(passed=not issues, issues=issues, thresholds=limits,
                   score=round(metrics['boundary_pair_delta'] + .25 * metrics['boundary_gradient'], 3),
                   note='边界统计是工程启发式，天然画线可能触发；须人工复核，不衡量历史真实性。')
    return metrics


def paired_metrics(target, output, mask, perceptual=False):
    """Mask MAE/PSNR plus SSIM/LPIPS on the mask bounding box + 16px context."""
    if target.shape != output.shape or mask.shape != target.shape[:2] or not (mask > 0).any():
        raise ValueError('配对目标、输出、非空掩膜必须对齐且尺寸一致。')
    active = mask > 0
    difference = target.astype(float) - output.astype(float)
    mse = float(np.mean(difference[active] ** 2))
    ys, xs = np.where(active)
    y0, y1 = max(0, ys.min()-16), min(target.shape[0], ys.max()+17)
    x0, x1 = max(0, xs.min()-16), min(target.shape[1], xs.max()+17)
    a, b = target[y0:y1, x0:x1], output[y0:y1, x0:x1]
    from skimage.metrics import structural_similarity
    window = min(7, min(a.shape[:2]))
    window -= (window % 2 == 0)
    ssim = float(structural_similarity(a, b, channel_axis=2, data_range=255, win_size=window)) if window >= 3 else None
    result = {'mask_mae': float(np.mean(np.abs(difference[active]))),
              'mask_psnr': 10 * np.log10(255**2/mse) if mse else None,
              'perfect_match': mse == 0, 'roi_ssim': ssim, 'roi_xyxy': [int(x0), int(y0), int(x1), int(y1)],
              'protocol': 'mask-only MAE/PSNR; SSIM/LPIPS mask bounding box +16px; data_range=255',
              'roi_lpips': None}
    if perceptual:
        import torch
        import lpips
        # Pretrained AlexNet weights must be cached; no random untrained metric.
        from pathlib import Path
        from .config import ROOT
        project_checkpoint = ROOT / 'models/lpips/alexnet-owt-7be5be79.pth'
        checkpoint = Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'
        if project_checkpoint.exists() and not checkpoint.exists():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(project_checkpoint, checkpoint)
        if not checkpoint.exists():
            raise ValueError('LPIPS 需要预先下载官方 AlexNet 权重；请运行模型下载脚本 --lpips。')
        network = lpips.LPIPS(net='alex', verbose=False).eval()
        def tensor(image):
            if min(image.shape[:2]) < 64:
                image = cv2.resize(image, (max(64, image.shape[1]), max(64, image.shape[0])))
            return torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
        with torch.inference_mode():
            result['roi_lpips'] = float(network(tensor(a), tensor(b)).item())
    return result
