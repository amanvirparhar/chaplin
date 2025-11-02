import torch
import hydra
from pipelines.pipeline import InferencePipeline
from chaplin import Chaplin


@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
def main(cfg):
    chaplin = Chaplin()

    # load the model
    chaplin.vsr_model = InferencePipeline(
        cfg.config_filename, device=torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available(
        ) and cfg.gpu_idx >= 0 else "cpu"), detector=cfg.detector, face_track=True)

    print("\n\033[48;5;22m\033[97m\033[1m MODEL LOADED SUCCESSFULLY! \033[0m\n")

    # start the webcam video capture
    chaplin.start_webcam()


if __name__ == '__main__':
    main()
