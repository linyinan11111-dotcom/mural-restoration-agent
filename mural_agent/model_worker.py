"""One-shot worker. May run under a separate MURAL_WORKER_PYTHON environment."""
from pathlib import Path
import json
import sys
import subprocess
from PIL import Image
from .imaging import sha256


def execute(request):
    output = Path(request['output'])
    if request['tool'] == 'research':
        adapter = json.loads(Path(request['research_adapter']).read_text(encoding='utf-8'))
        if not all(adapter.get(k) for k in ('python', 'script', 'repository_commit', 'weight_sha256', 'license')):
            raise ValueError('研究模型 adapter 需要独立 Python、脚本、commit、权重 hash 与许可记录。')
        # User-installed adapter implements the same JSON file protocol; no shell execution.
        subprocess.run([adapter['python'], adapter['script'], sys.argv[1]], check=True, timeout=adapter.get('timeout', 240))
        return {'device': request['device'], 'adapter': adapter}
    import torch
    from diffusers import StableDiffusionInpaintPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel
    device = 'cuda' if request['device'] != 'cpu' and torch.cuda.is_available() else 'cpu'
    if request['device'] == 'cuda' and device != 'cuda':
        raise RuntimeError('请求 CUDA，但设备不可用。')
    dtype = torch.float16 if device == 'cuda' else torch.float32
    step = request['step']
    kwargs = dict(torch_dtype=dtype, local_files_only=True, use_safetensors=True, variant='fp16')
    if step.get('structure_control'):
        control = ControlNetModel.from_pretrained(request['controlnet'], local_files_only=True, torch_dtype=dtype)
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(request['model'], controlnet=control, **kwargs)
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(request['model'], **kwargs)
    if device == 'cuda':
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cpu')
    pipe.enable_attention_slicing()
    image = Image.open(request['image']).convert('RGB')
    mask = Image.open(request['mask']).convert('L')
    # Maintain aspect ratio by pad, then undo pad. Inference limited to 512px on laptop VRAM.
    original_size = image.size
    scale = min(1., 512/max(original_size))
    size = (max(8, round(image.width*scale)), max(8, round(image.height*scale)))
    image = image.resize(size, Image.Resampling.LANCZOS)
    mask = mask.resize(size, Image.Resampling.NEAREST)
    padded_size = (((size[0]+7)//8)*8, ((size[1]+7)//8)*8)
    canvas = Image.new('RGB', padded_size)
    canvas.paste(image, (0, 0))
    mask_canvas = Image.new('L', padded_size)
    mask_canvas.paste(mask, (0, 0))
    def callback(pipeline, index, timestep, callback_kwargs):
        if Path(request['cancel_file']).exists():
            raise RuntimeError('Cancelled')
        return callback_kwargs
    args = dict(prompt=step.get('prompt') or 'conservative reconstruction of aged mural, preserve existing painted lines, muted mineral pigments, matching surrounding texture',
                negative_prompt='modern objects, text, watermark, vivid neon colors, photorealistic face',
                image=canvas, mask_image=mask_canvas, height=padded_size[1], width=padded_size[0],
                num_inference_steps=25, strength=step.get('strength', .8), guidance_scale=6.5,
                generator=torch.Generator('cpu').manual_seed(step['seed']), callback_on_step_end=callback)
    if step.get('structure_control'):
        import numpy as np
        control_array = np.asarray(canvas).copy()
        control_array[np.asarray(mask_canvas) > 127] = 255
        args['control_image'] = Image.fromarray(control_array)
    result = pipe(**args)
    if result.nsfw_content_detected and any(result.nsfw_content_detected):
        raise RuntimeError('模型安全检查未通过，未保存生成结果。')
    result.images[0].crop((0, 0, *size)).resize(original_size, Image.Resampling.LANCZOS).save(output)
    def provenance(path):
        ledger = Path(path) / 'provenance.json'
        return json.loads(ledger.read_text(encoding='utf-8')) if ledger.exists() else None
    return {'device': device, 'torch': torch.__version__, 'seed': step['seed'],
            'model': request['model'], 'model_index_sha256': sha256(Path(request['model'])/'model_index.json'),
            'model_provenance': provenance(request['model']),
            'controlnet_provenance': provenance(request['controlnet']) if step.get('structure_control') else None,
            'inference_steps': 25, 'structure_control': step.get('structure_control', False)}


def main():
    path = Path(sys.argv[1]).resolve()
    request = json.loads(path.read_text(encoding='utf-8'))
    if request.get('protocol') != 1 or request['tool'] not in {'diffusion', 'research'}:
        raise ValueError('Unsupported worker request')
    metadata = execute(request)
    path.with_name('worker-result.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
