import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from pytorch_lightning import LightningModule
from transformers import BertModel


PRE_TRAINED_BERT = "bert-base-uncased"


class LanguageModule(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_BERT)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        text_features = self.bert(text)
        return F.relu(self.fc(text_features.pooler_output))


class VisionModule(nn.Module):
    def __init__(self, backbone_output_dim, output_dim):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(backbone_output_dim, output_dim)

    def forward(self, image):
        return F.relu(self.fc(self.backbone(image)))


class ConcatModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        backbone_output_dim,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
        num_classes=2,
    ):
        super().__init__()

        self.language_module = LanguageModule(embedding_dim, language_feature_dim)
        self.vision_module = VisionModule(backbone_output_dim, vision_feature_dim)

        self.fusion = nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=fusion_output_size,
        )
        self.fc = nn.Linear(in_features=fusion_output_size, out_features=num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text, image):
        text_features = self.language_module(text)
        image_features = self.vision_module(image)

        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        logits = self.fc(fused)
        pred = F.softmax(logits)

        return pred


class ConcatBert(nn.Module):
    def __init__(
        self,
        embedding_dim,
        backbone_output_dim,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
        num_classes=2,
    ):
        super().__init__()

        self.language_module = LanguageModule(embedding_dim, language_feature_dim)
        self.vision_module = VisionModule(backbone_output_dim, vision_feature_dim)

        self.fusion = nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=fusion_output_size,
        )
        self.fc = nn.Linear(in_features=fusion_output_size, out_features=num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text, image):
        text_features = self.language_module(text)
        text_features = F.relu(text_features)
        image_features = F.relu(self.vision_module(image))

        combined = torch.cat([text_features, image_features], dim=1)
        fused = F.relu(self.fusion(combined))
        logits = self.fc(fused)
        pred = F.softmax(logits)

        return pred


class LanguageAndVisionConcat(LightningModule):
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

    def forward(self, text, image, label=None):
        return self.model(text, image, label)

    def training_step(self, batch, batch_idx):

        image, text, label = batch
        pred = self.model(text, image)

        loss = self.loss_fn(pred, label)
        acc = self.train_accuracy(pred, label)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        image, text, label = batch
        pred = self.model(text, image)

        loss = self.loss_fn(pred, label)
        acc = self.val_accuracy(pred, label)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
