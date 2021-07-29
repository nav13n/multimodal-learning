import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from pytorch_lightning import LightningModule


class LanguageModule(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        return F.relu(self.fc(text))


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
        text_features = F.relu(self.language_module(text))
        image_features = F.relu(self.vision_module(image))

        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        logits = self.fc(fused)
      
        return logits


class SemiLanguageAndVisionConcat(LightningModule):
    def __init__(
        self,
        embedding_dim,
        backbone_output_dim,
        language_feature_dim,
        vision_feature_dim,
        fusion_output_size,
        dropout_p,
        num_classes=2,
        lr=0.0003,
        weight_decay=0.00005,
        T=1,
        threshold=0.95,
        lambda_s = 1
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ConcatModel(
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
        self.T = T
        self.threshold = threshold
        self.lambda_s = lambda_s


    def forward(self, text, image, label=None):
        return self.model(text, image, label)

    def training_step(self, batch, batch_idx):
        labeled, unlabeled = batch

        image, text, label = labeled
        img_tensor_w, img_tensor_s, text_tensor_w, text_tensor_s = unlabeled
       
        # Actual Image Loss 
        logits = self.model(text, image)
        pred = F.softmax(logits)
        loss = F.cross_entropy(logits, label, reduction='mean')
     
        acc = self.train_accuracy(pred, label)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        #TODO All three can be inferred at once with some tensor magic

        # Weak Aug Loss 
        logits_w = self.model(text_tensor_w, img_tensor_w)

        loss_w = self.loss_fn(pred, label)

        pseudo_label = torch.softmax(logits_w.detach()/self.T, dim=-1)
        max_probs, label_s = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        # Strong Aug Loss
        logits_s = self.model(text_tensor_s, img_tensor_s)
        loss_s = (F.cross_entropy(logits_s, label_s,
                                  reduction='none') * mask).mean()

        self.log("train/loss_s", loss, on_step=False, on_epoch=True, prog_bar=False)
        
        total_loss = loss + self.lambda_s * loss_s

        return total_loss

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
