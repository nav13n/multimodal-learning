import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from pytorch_lightning import LightningModule
from transformers import BertModel

from uniter.model import UniterModel

class HatefulMemesUniter(nn.Module):

    def __init__(self,
                 model: UniterModel,
                 hidden_size: int,
                 n_classes: int):
        super().__init__()
        self.model = model
        self.n_classes = n_classes
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, **kwargs):
        out = self.model(**kwargs)
        out = self.model.pooler(out)
        out = self.linear(out)
        return out

class HatefulMemesUniterModule(LightningModule):
    def __init__(
        self,
        embedding_dim,
        backbone_output_dim,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
        model_type="concat",
        num_classes=2,
        lr=0.0003,
        weight_decay=0.00005,
    ):
        super().__init__()

        assert model_type in ["concat", "concat_bert"]

        self.save_hyperparameters()

        if model_type == "concat":
            self.model = ConcatModel(
                embedding_dim,
                backbone_output_dim,
                language_feature_dim,
                vision_feature_dim,
                fusion_output_size,
                dropout_p,
                num_classes=num_classes,
            )
        else:
            self.model = ConcatBert(
                embedding_dim,
                backbone_output_dim,
                language_feature_dim,
                vision_feature_dim,
                fusion_output_size,
                dropout_p,
                num_classes=num_classes,
            )

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.train_auroc = torchmetrics.AUROC(2)
        self.val_auroc = torchmetrics.AUROC(2)

    def forward(self, text, image, label=None):
        return self.model(text, image, label)

    def training_step(self, batch, batch_idx):

        image, text, label = batch
        logits, pred = self.model(text, image)

        loss = self.loss_fn(pred, label)
        acc = self.train_accuracy(pred, label)
        auroc = self.train_auroc(pred, label)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, text, label = batch
        logits, pred = self.model(text, image)

        loss = self.loss_fn(pred, label)
        acc = self.val_accuracy(pred, label)
        auroc = self.val_auroc(pred, label)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
