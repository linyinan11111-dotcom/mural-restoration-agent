from dataclasses import dataclass
from pathlib import Path
import os

import sys
ROOT = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent.parent


def load_env(path: Path = ROOT / '.env') -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        os.environ.setdefault(key.strip(), value.strip().strip('\"\''))


@dataclass
class Settings:
    data_dir: Path
    lama_model: Path
    device: str = 'auto'
    lama_max_side: int = 1024
    api_kind: str = 'responses'
    api_base_url: str = 'https://api.openai.com/v1'
    api_model: str = ''
    api_key: str = ''
    api_timeout: float = 90
    api_json_mode: bool = True
    sam_model: Path = ROOT / 'models/sam2'
    diffusion_model: Path = ROOT / 'models/sd-inpainting'
    controlnet_model: Path = ROOT / 'models/controlnet-inpaint'
    worker_python: str = ''
    research_adapter: Path = ROOT / 'models/research-adapter.json'
    vision_model: Path = ROOT / 'models/qwen-vl'
    imagine_provider: str = 'openai_json'
    imagine_base_url: str = 'https://api.x.ai/v1'
    imagine_model: str = 'grok-imagine-image-2.0'
    imagine_api_key: str = ''
    imagine_timeout: float = 180

    @property
    def api_ready(self) -> bool:
        return bool(self.api_key and self.api_model)

    @property
    def imagine_ready(self) -> bool:
        return bool(self.imagine_api_key.strip() and self.imagine_base_url.strip() and self.imagine_model.strip())

    @classmethod
    def from_env(cls):
        load_env()
        return cls(
            data_dir=Path(os.getenv('MURAL_DATA_DIR', str(ROOT / 'data'))).resolve(),
            lama_model=Path(os.getenv('MURAL_LAMA_MODEL', str(ROOT / 'models/big-lama.pt'))).resolve(),
            device=os.getenv('MURAL_DEVICE', 'auto'),
            lama_max_side=max(128, min(2048, int(os.getenv('MURAL_LAMA_MAX_SIDE', '1024')))),
            api_kind=os.getenv('MURAL_API_KIND', 'responses'),
            api_base_url=os.getenv('MURAL_API_BASE_URL', 'https://api.openai.com/v1').rstrip('/'),
            api_model=os.getenv('MURAL_API_MODEL', ''),
            api_key=os.getenv('MURAL_API_KEY', ''),
            api_timeout=float(os.getenv('MURAL_API_TIMEOUT', '90')),
            api_json_mode=os.getenv('MURAL_API_JSON_MODE', 'true').lower() == 'true',
            sam_model=Path(os.getenv('MURAL_SAM_MODEL', str(ROOT / 'models/sam2'))),
            diffusion_model=Path(os.getenv('MURAL_DIFFUSION_MODEL', str(ROOT / 'models/sd-inpainting'))),
            controlnet_model=Path(os.getenv('MURAL_CONTROLNET_MODEL', str(ROOT / 'models/controlnet-inpaint'))),
            worker_python=os.getenv('MURAL_WORKER_PYTHON', ''),
            research_adapter=Path(os.getenv('MURAL_RESEARCH_ADAPTER', str(ROOT / 'models/research-adapter.json'))),
            vision_model=Path(os.getenv('MURAL_VISION_MODEL', str(ROOT / 'models/qwen-vl'))),
            imagine_provider=os.getenv('MURAL_IMAGINE_PROVIDER', 'openai_json'),
            imagine_base_url=os.getenv('MURAL_IMAGINE_BASE_URL', 'https://api.x.ai/v1').rstrip('/'),
            imagine_model=os.getenv('MURAL_IMAGINE_MODEL', 'grok-imagine-image-2.0'),
            imagine_api_key=os.getenv('MURAL_IMAGINE_API_KEY', ''),
            imagine_timeout=float(os.getenv('MURAL_IMAGINE_TIMEOUT', '180')),
        )

