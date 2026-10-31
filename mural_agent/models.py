from pathlib import Path
from threading import Lock
import importlib.util
import numpy as np
from PIL import Image
from .config import Settings


class LamaBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = Lock()
        self.model = None
        self.device = None

    def available(self):
        return self.settings.lama_model.is_file() and importlib.util.find_spec('torch') is not None

    def info(self):
        return {'available': self.available(), 'loaded': self.model is not None,
                'device': str(self.device) if self.device else self.settings.device,
                'max_side': self.settings.lama_max_side}

    def unload(self):
        with self.lock:
            self.model = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self):
        if self.model is not None:
            return
        if not self.available():
            raise RuntimeError('LaMa 权重或 PyTorch 未安装，请按 README 运行安装步骤。')
        import torch
        requested = self.settings.device
        cuda_usable = requested != 'cpu' and torch.cuda.is_available()
        if cuda_usable:
            major, minor = torch.cuda.get_device_capability(0)
            architectures = torch.cuda.get_arch_list()
            cuda_usable = f'sm_{major}{minor}' in architectures
        if requested == 'cuda' and not cuda_usable:
            raise RuntimeError('当前 PyTorch 不支持这张 GPU，请安装支持 sm_120 的 CUDA 12.8 或更新版本。')
        self.device = torch.device('cuda' if requested != 'cpu' and cuda_usable else 'cpu')
        # Torch's C++ path reader can reject non-ASCII Windows paths. Python's
        # binary stream handles the user's Chinese workspace path correctly.
        with self.settings.lama_model.open('rb') as checkpoint:
            self.model = torch.jit.load(checkpoint, map_location=self.device).eval()
        # Keep CPU inference from monopolizing the notebook's full CPU.
        if self.device.type == 'cpu':
            torch.set_num_threads(min(8, torch.get_num_threads()))

    def predict(self, original, mask):
        import torch
        with self.lock:
            self.load()
            ys, xs = np.where(mask > 0)
            pad = 160
            x0, x1 = max(0, int(xs.min()) - pad), min(original.shape[1], int(xs.max()) + pad + 1)
            y0, y1 = max(0, int(ys.min()) - pad), min(original.shape[0], int(ys.max()) + pad + 1)
            crop = original[y0:y1, x0:x1]
            crop_mask = mask[y0:y1, x0:x1]
            height, width = crop.shape[:2]
            scale = min(1, self.settings.lama_max_side / max(height, width))
            size = (max(8, round(width * scale)), max(8, round(height * scale)))
            image = np.asarray(Image.fromarray(crop).resize(size, Image.Resampling.LANCZOS)).copy()
            # NEAREST downscaling can drop a one-pixel crack entirely. Dilate the
            # mask by the inverse scale so the model still sees the editable region;
            # the final composite uses the original mask, so extra context is safe.
            model_mask = crop_mask
            if scale < 1:
                import cv2
                radius = int(np.ceil(1 / scale)) + 1
                model_mask = cv2.dilate(crop_mask, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
            mask_small = np.asarray(Image.fromarray(model_mask).resize(size, Image.Resampling.NEAREST)).copy()
            if not (mask_small > 127).any():
                raise ValueError('掩膜在推理缩放下丢失；请扩大圈选范围或提高推理分辨率。')
            h, w = image.shape[:2]
            image = np.pad(image, ((0, (-h) % 8), (0, (-w) % 8), (0, 0)), mode='symmetric')
            mask_small = np.pad(mask_small, ((0, (-h) % 8), (0, (-w) % 8)), mode='constant')
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()[None].to(self.device) / 255
            mask_tensor = torch.from_numpy((mask_small > 127).astype(np.float32))[None, None].to(self.device)
            with torch.inference_mode():
                result = self.model(tensor, mask_tensor)
            if isinstance(result, dict):
                result = result.get('inpainted', result.get('output'))
            if isinstance(result, (tuple, list)):
                result = result[0]
            output = result[0].detach().float().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            output = np.clip(output * 255, 0, 255).astype(np.uint8)
            output = np.asarray(Image.fromarray(output).resize((width, height), Image.Resampling.LANCZOS))
            prediction = original.copy()
            prediction[y0:y1, x0:x1] = output
            return prediction

