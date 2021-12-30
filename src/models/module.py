from torch.optim import Optimizer, optimizer
import pytorch_lightning as pl


class PLModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def training_step(self, batch):
        loss = self._calculate_loss('train', batch)
        return loss

    def validation_step(self, batch, *args):
        loss = self._calculate_loss('valid', batch)
        return loss

    def _calculate_loss(self, phase, batch):
        embeddings, captions, _ = batch
        loss = self.model.compute_loss(embeddings, captions)
        self.model.log(f'{phase}_loss', loss)
        return loss

    def configure_optimizers(self):
        return optimizer
