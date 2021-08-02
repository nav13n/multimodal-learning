import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule

from .uniter.model import UniterModel
from .uniter.pretrain import UniterForPretraining

IMG_DIM = 2048
IMG_LABEL_DIM = 1601

class HatefulMemesUniterModel(nn.Module):

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

class HatefulMemesUniter(LightningModule):
    def __init__(
        self,
        pretrained_model_file,
        pretrained_model_config,
        num_classes=2,
        dropout=0.1,
        lr=0.0003,
        weight_decay=0.00005,
    ):
        super().__init__()

        self.save_hyperparameters()

        # Initialise HatefulMemes Uniter Model
        checkpoint = torch.load(pretrained_model_file)
        base_model = UniterForPretraining.from_pretrained(pretrained_model_config,
                                                            state_dict=checkpoint,
                                                            img_dim=IMG_DIM,
                                                            img_label_dim=IMG_LABEL_DIM)
        self.model = HatefulMemesUniterModel(uniter_model=base_model.uniter,
                                hidden_size=base_model.uniter.config.hidden_size,
                                n_classes=num_classes)       


        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.train_auroc = torchmetrics.AUROC(2)
        self.val_auroc = torchmetrics.AUROC(2)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_idx):

        pred = self.model(img_feat=batch['img_feat'], img_pos_feat=batch['img_pos_feat'], input_ids=batch['input_ids'],
                            position_ids=batch['position_ids'], attention_mask=batch['attn_mask'], gather_index=batch['gather_index'],
                            output_all_encoded_layers=False)

        label = batch['labels']
        loss = self.loss_fn(pred, label)
        acc = self.train_accuracy(pred, label)
        auroc = self.train_auroc(pred, label)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.model(img_feat=batch['img_feat'], img_pos_feat=batch['img_pos_feat'], input_ids=batch['input_ids'],
                            position_ids=batch['position_ids'], attention_mask=batch['attn_mask'], gather_index=batch['gather_index'],
                            output_all_encoded_layers=False)

        label = batch['labels']
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

