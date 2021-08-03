from pandas.core.base import NoNewAttributesMixin
from src.models.modules.uniter import model
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import torchmetrics
from pytorch_lightning import LightningModule
from transformers import BertModel


PRE_TRAINED_BERT = "bert-base-uncased"


class LanguageModule(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text):
        return self.dropout(F.relu(self.fc(text)))


class VisionModule(nn.Module):
    def __init__(self, backbone_output_dim, output_dim):
        super().__init__()
        self.backbone = torchvision.models.resnet101(pretrained=True)
        self.fc = nn.Linear(backbone_output_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, image):
        return self.dropout(F.relu(self.fc(self.backbone(image))))


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
        self.dropout1 = nn.Dropout(dropout_p)

    def forward(self, text, image):
        text_features = self.language_module(text)
        image_features = self.vision_module(image)

        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(torch.nn.functional.relu(self.fusion(combined)))
        logits = self.fc(fused)
        pred = F.softmax(logits)

        return logits, pred


class UnimodalImage(nn.Module):
    def __init__(
        self,
        backbone_output_dim,
        vision_feature_dim,
        dropout_p,
        num_classes=2,
    ):
        super().__init__()

        self.vision_module = VisionModule(backbone_output_dim, vision_feature_dim)
        self.fc = nn.Linear(in_features=vision_feature_dim, out_features=num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text, image):
        image_features = self.vision_module(image)
        logits = self.fc(image_features)
        pred = F.softmax(logits)

        return logits, pred


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

        self.bert = BertModel.from_pretrained(PRE_TRAINED_BERT)

        self.language_module = LanguageModule(embedding_dim, language_feature_dim)
        self.vision_module = VisionModule(backbone_output_dim, vision_feature_dim)

        self.fusion = nn.Linear(
            in_features=(language_feature_dim + vision_feature_dim),
            out_features=fusion_output_size,
        )
        self.fc = nn.Linear(in_features=fusion_output_size, out_features=num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text, image):
        text_features = self.bert(text)
        text_features = self.language_module(text_features.pooler_output)
        text_features = text_features
        image_features = self.vision_module(image)

        combined = torch.cat([text_features, image_features], dim=1)
        fused = self.dropout(F.relu(self.fusion(combined)))
        logits = self.fc(fused)
        pred = F.softmax(logits)

        return logits, pred


class UnimodalFasttext(nn.Module):
    def __init__(
        self,
        embedding_dim,
        language_feature_dim,
        dropout_p,
        num_classes=2,
    ):
        super().__init__()

        self.language_module = LanguageModule(embedding_dim, language_feature_dim)
        self.fc = nn.Linear(in_features=language_feature_dim, out_features=num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text, image):
        text_features = self.language_module(text)
        logits = self.fc(text_features)
        pred = F.softmax(logits)

        return logits, pred


class UnimodalBert(nn.Module):
    def __init__(
        self,
        embedding_dim,
        language_feature_dim,
        dropout_p,
        num_classes=2,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(PRE_TRAINED_BERT)

        self.language_module = LanguageModule(embedding_dim, language_feature_dim)
        self.fc = nn.Linear(in_features=language_feature_dim, out_features=num_classes)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, text, image):
        text_features = self.bert(text)
        text_features = self.language_module(text_features.pooler_output)
        text_features = text_features
        logits = self.fc(text_features)
        pred = F.softmax(logits)
        return logits, pred


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

        if model_type == "unimodel_fasttext":

            self.model = UnimodalFasttext(
                embedding_dim,
                language_feature_dim,
                dropout_p,
                num_classes=num_classes,
            )

        if model_type == "unimodel_bert":

            self.model = UnimodalBert(
                embedding_dim,
                language_feature_dim,
                dropout_p,
                num_classes=num_classes,
            )

        if model_type == "unimodel_bert":

            self.model = UnimodalBert(
                embedding_dim,
                language_feature_dim,
                dropout_p,
                num_classes=num_classes,
            )

        if model_type == "unimodel_image":
            self.model = UnimodalImage(
                backbone_output_dim,
                vision_feature_dim,
                dropout_p,
                num_classes=num_classes,
            )

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


class SemiLanguageAndVisionConcat(LanguageAndVisionConcat):
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
        T=1,
        threshold=0.95,
        lambda_s=1,
    ):
        super().__init__(
            embedding_dim,
            backbone_output_dim,
            language_feature_dim,
            vision_feature_dim,
            fusion_output_size,
            dropout_p,
            model_type,
            num_classes,
            lr,
            weight_decay,
        )
        self.T = T
        self.threshold = threshold
        self.lambda_s = lambda_s

    def training_step(self, batch, batch_idx):
        labeled, unlabeled = batch

        image, text, label = labeled
        img_tensor_w, img_tensor_s, text_tensor_w, text_tensor_s = unlabeled

        batch_size = image.shape[0]

        # stack all inputs together in a single batch
        # this would help call model.forward() only once

        img_inputs = torch.cat((image, img_tensor_w, img_tensor_s))

        # text from labeled and unlabeled datasets can vary in length
        # hence pad them
        # output shape: (3, max_length, batch_size)
        text_inputs = pad_sequence(
            (
                text.transpose(0, 1),
                text_tensor_w.transpose(0, 1),
                text_tensor_s.transpose(0, 1),
            ),
            batch_first=True,
        )

        # output shape: (3, batch_size, max_length)
        text_inputs = text_inputs.transpose(1, 2)

        # output shape: (3 * batch_size, max_length)
        text_inputs = text_inputs.reshape(-1, text_inputs.shape[-1])

        logits, pred = self.model(text_inputs, img_inputs)

        logits_x = logits[:batch_size]
        logits_w = logits[batch_size : 2 * batch_size]
        logits_s = logits[2 * batch_size : 3 * batch_size]

        del logits

        pred_x = pred[:batch_size]
        pred_w = pred[batch_size : 2 * batch_size]
        pred_s = pred[2 * batch_size : 3 * batch_size]

        loss = F.cross_entropy(logits_x, label, reduction="mean")

        acc = self.train_accuracy(pred_x, label)
        auroc = self.train_auroc(pred_x, label)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/auroc", auroc, on_step=False, on_epoch=True, prog_bar=True)

        # TODO All three can be inferred at once with some tensor magic

        # Weak Aug Loss
        # logits_w, pred_w = self.model(text_tensor_w, img_tensor_w)

        pseudo_label = torch.softmax(logits_w.detach() / self.T, dim=-1)
        max_probs, label_s = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        # Strong Aug Loss
        # logits_s, pred_s = self.model(text_tensor_s, img_tensor_s)
        loss_s = (F.cross_entropy(logits_s, label_s, reduction="none") * mask).mean()

        self.log("train/loss_s", loss_s, on_step=True, on_epoch=True, prog_bar=True)
        print(f"loss lab: {loss} | loss strong: {loss_s}")
        total_loss = loss + self.lambda_s * loss_s
        self.log(
            "train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
        )

        return total_loss
