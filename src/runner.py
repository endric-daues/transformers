from jsonargparse import ArgumentParser

from trainer import Trainer

parser = ArgumentParser()
parser.add_class_arguments(Trainer, "trainer")
parser.add_argument("--action", required=True)

if __name__ == "__main__":
    cfg = parser.parse_args()
    trainer = (
        Trainer()
    )  # TODO: move this one layer down and add a sampler and predictor class.
    if cfg.action == "train":
        trainer.train()
    if cfg.action == "sample":
        trainer.sample()
