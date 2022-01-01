import torch

from torch.optim import Optimizer
from sklearn.metrics import classification_report
import torch.nn.functional as F
import pytorch_lightning as pl


class PLModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        return self.model(images)

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def training_step(self, batch):
        loss = self._calculate_loss('train', batch)
        return loss

    def validation_step(self, batch, *args):
        loss = self._calculate_loss('valid', batch)
        return loss

    def _calculate_loss(self, phase, batch):
        images, labels = batch
        outputs = self(images)

        # Log more metrics
        self._log_metrics(phase, outputs, labels)

        # Calculate loss
        loss = F.cross_entropy(outputs, labels)
        self.log(f'{phase}/loss', loss)

        return loss

    def _log_metrics(self, phase, outputs, labels):
        predictions = outputs.argmax(dim=1).cpu().numpy()
        labels = labels.cpu().numpy()

        # Generate a classification report
        report = classification_report(
            y_true=labels, y_pred=predictions, output_dict=True, zero_division=0)

        metric_names = ['precision', 'recall', 'f1-score']
        for metric_name in metric_names:
            metric = report['weighted avg'][metric_name]
            self.log(f'{phase}/{metric_name}', metric)

    def configure_optimizers(self):
        return self.optimizer
