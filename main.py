import os
from pathlib import Path

import torch
import hydra
from pipelines.pipeline import InferencePipeline
from chaplin import Chaplin


def _load_env_files():
    """Populate os.environ with values from .env/.envrc if present."""
    base_dir = Path(__file__).resolve().parent
    for filename in (".env", ".envrc"):
        path = base_dir / filename
        if not path.exists():
            continue
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip()
            value = value.strip()
            if not key or key.startswith("#"):
                continue
            if value and value[0] in ("'", '"') and value[-1] == value[0]:
                value = value[1:-1]
            os.environ.setdefault(key, value)


_load_env_files()


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    if "output" in cfg and cfg.output.mode:
        os.environ.setdefault("CHAPLIN_OUTPUT_MODE", cfg.output.mode)
    llm_cfg = cfg.llm if "llm" in cfg else None
    camera_idx = getattr(cfg, "camera_index", 0)
    output_cfg = cfg.output if "output" in cfg else None
    chaplin = Chaplin(
        llm_config=llm_cfg,
        output_config=output_cfg,
        camera_index=camera_idx,
    )

    # load the model
    chaplin.vsr_model = InferencePipeline(
        cfg.config_filename, device=torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available(
        ) and cfg.gpu_idx >= 0 else "cpu"), detector=cfg.detector, face_track=True)

    print("\n\033[48;5;22m\033[97m\033[1m MODEL LOADED SUCCESSFULLY! \033[0m\n")

    # start the webcam video capture
    chaplin.start_webcam()


if __name__ == '__main__':
    main()
