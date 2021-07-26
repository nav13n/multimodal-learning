import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule


class LanguageModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        return F.relu(self.fc(self.embed(text)))


class VisionModule(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, image):
        return F.relu(self.fc(self.backbone(image)))


class ConcatModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
        num_classes=2,
    ):
        super().__init__()

        self.language_module = LanguageModule(
            vocab_size, embedding_dim, language_feature_dim
        )
        self.vision_module = VisionModule(vision_feature_dim)

        self.fusion = nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=fusion_output_size,
        )
        self.fc = nn.Linear(in_features=fusion_output_size, out_features=num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text, image):
        text_features = F.relu(self.language_module(text))
        image_features = F.relu(self.vision_module(image))

        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        logits = self.fc(fused)
        pred = F.softmax(logits)

        return pred


class LanguageAndVisionConcat(LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
        num_classes=2,
        lr=0.0003,
        weight_decay=0.00005,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConcatModel(
            vocab_size,
            embedding_dim,
            language_feature_dim,
            vision_feature_dim,
            fusion_output_size,
            dropout_p,
            num_classes=num_classes,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, text, image, label=None):
        return self.model(text, image, label)

    def training_step(self, batch, batch_idx):
        text, image, label = batch
        pred = self.model(text, image, label)
        loss = self.loss_fn(pred, label)
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        text, image, label = batch
        pred = self.model(text, image, label)
        loss = self.loss_fn(pred, label)
        self.log(
            "val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(tuple(output for output in outputs)).mean()
        self.log("val_loss", avg_loss)
        return avg_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
