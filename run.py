import hydra
from omegaconf import DictConfig
from src.train import train


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Run train logic with the configuration object
    train(config)


if __name__ == '__main__':
    main()
