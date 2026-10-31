"""Prompted SAM 2.1 segmentation; heuristic damage proposals stay unconfirmed."""
from pathlib import Path
from threading import Lock
import cv2
import numpy as np


def damage_proposal(image, sensitivity=85):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    response = np.zeros_like(gray)
    for size in (5, 11, 21):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        for operation in (cv2.MORPH_BLACKHAT, cv2.MORPH_TOPHAT):
            response = np.maximum(response, cv2.morphologyEx(gray, operation, kernel))
    threshold = max(18, float(np.percentile(response, max(70, min(99, sensitivity)))))
    mask = (response > threshold).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    accepted = np.zeros_like(mask)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if 8 <= area <= image.shape[0]*image.shape[1]*.03 and max(w, h) >= 8:
            accepted[components == index] = 255
    return accepted, {'method': 'multiscale_morphology', 'threshold': threshold,
                      'confirmed': False, 'notice': '高对比细线候选，会误选原画线条；并非病害分类模型。'}


def crack_mask_proposal(image, box, padding_fraction=.015):
    """Extract thin dark-line candidates inside one vision-model crack proposal."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    if x1 <= x0 or y1 <= y0:
        raise ValueError('裂缝候选框无效。')
    padding = max(4, round(min(width, height)*padding_fraction))
    x0, y0 = max(0, x0-padding), max(0, y0-padding)
    x1, y1 = min(width, x1+padding), min(height, y1+padding)
    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    response = np.zeros_like(gray)
    for size in (3, 5, 7, 9):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        response = np.maximum(response, cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel))
    threshold = max(8., float(np.percentile(response, 97)))
    local = (response >= threshold).astype(np.uint8)
    local = cv2.morphologyEx(local, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(local, 8)
    selected = np.zeros_like(local)
    for index in range(1, count):
        x, y, component_width, component_height, area = stats[index]
        long_side = max(component_width, component_height)
        short_side = max(1, min(component_width, component_height))
        if area >= 3 and long_side >= 5 and long_side/short_side >= 1.4:
            selected[labels == index] = 1
    selected = cv2.dilate(selected, np.ones((3, 3), np.uint8), iterations=1)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y0:y1, x0:x1] = selected*255
    return mask, {'method': 'qwen_box_multiscale_blackhat', 'threshold': threshold,
                  'confirmed': False, 'box': [x0, y0, x1, y1],
                  'fraction': float((mask > 0).mean()),
                  'notice': '裂缝线像素候选，黑色原画线条也可能被选中；请人工检查和擦除误选区域。'}


class SamSegmenter:
    def __init__(self, settings):
        self.settings = settings
        self.model = self.processor = None
        self.lock = Lock()

    def available(self):
        return (Path(self.settings.sam_model) / 'model.safetensors').exists()

    def unload(self):
        self.model = self.processor = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, image, points=None, labels=None, box=None):
        if not self.available():
            raise ValueError('SAM 2.1 权重尚未安装。')
        if not points and box is None:
            raise ValueError('请提供至少一个点或矩形框。')
        import torch
        from transformers import Sam2Model, Sam2Processor
        with self.lock:
            if self.model is None:
                device = 'cuda' if self.settings.device != 'cpu' and torch.cuda.is_available() else 'cpu'
                self.processor = Sam2Processor.from_pretrained(str(self.settings.sam_model), local_files_only=True)
                self.model = Sam2Model.from_pretrained(str(self.settings.sam_model), local_files_only=True).to(device).eval()
            kwargs = {}
            if points:
                kwargs['input_points'] = [[points]]
                kwargs['input_labels'] = [[labels or [1]*len(points)]]
            if box is not None:
                kwargs['input_boxes'] = [[box]]
            inputs = self.processor(images=image, return_tensors='pt', **kwargs).to(self.model.device)
            with torch.inference_mode():
                outputs = self.model(**inputs, multimask_output=True)
            masks = self.processor.post_process_masks(outputs.pred_masks.cpu(), inputs['original_sizes'].cpu())[0]
            scores = outputs.iou_scores.detach().cpu().numpy().reshape(-1)
            arrays = masks.numpy().reshape(-1, image.shape[0], image.shape[1])
            selected = int(np.argmax(scores))
            return arrays[selected].astype(np.uint8)*255, {'method': 'sam2.1-hiera-tiny',
                   'predicted_iou': float(scores[selected]), 'confirmed': False,
                   'points': points, 'labels': labels, 'box': box,
                   'notice': '分割置信分不是病害概率，需人工修改并确认。'}
