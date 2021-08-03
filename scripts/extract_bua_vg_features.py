# coding=utf-8
# Copyleft 2019 Project LXRT

import argparse
import base64
import csv
import json
import math
import os
import random
import sys
import time
csv.field_size_limit(sys.maxsize)

# import some common libraries
import cv2
import numpy as np
import torch
import tqdm

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.utils.visualizer import _create_text_labels_attr, _create_text_labels


D2_ROOT = os.path.dirname(os.path.dirname(detectron2.__file__)) # Root of detectron2

parser = argparse.ArgumentParser()
parser.add_argument('--split', default='dev', help='train,dev,test')
parser.add_argument('--data_path', default='', help='path to the train/dev/test split images folders')
parser.add_argument('--output_path', default='', help='path to the output for the extracted features')
parser.add_argument('--min_boxes', default=36, type=int, help='minimum number of detected boxes')
parser.add_argument('--max_boxes', default=36, type=int, help='maximum number of detected boxes')
parser.add_argument('--score_thresh', default=0.2, type=float, help='roi score threshold')
parser.add_argument('--batchsize', default=4, type=int, help='batch_size')
parser.add_argument('--model', default='res5', type=str, help='options: "res4", "res5"; features come from)')
parser.add_argument('--weight', default='vg', type=str, 
        help='option: mask, obj, vg. mask:mask_rcnn on COCO, obj: faster_rcnn on COCO, vg: faster_rcnn on Visual Genome')

args = parser.parse_args()

from torchvision.ops import nms
from detectron2.structures import Boxes, Instances
def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    
    # Select max scores
    max_scores, max_classes = scores.max(1)       # R x C --> R
    num_objs = boxes.size(0)
    boxes = boxes.view(-1, 4)
    idxs = torch.arange(num_objs).cuda() * num_bbox_reg_classes + max_classes
    max_boxes = boxes[idxs]     # Select max boxes according to the max scores.
    
    # Apply NMS
    keep = nms(max_boxes, max_scores, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores = max_boxes[keep], max_scores[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = max_classes[keep]
    
    return result, keep

def get_vg_classes():
    # Load VG Classes
    data_path = 'demo/data/genome/1600-400-20'

    vg_classes = []
    with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            vg_classes.append(object.split(',')[0].lower().strip())
            
    vg_attrs = []
    with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
        for object in f.readlines():
            vg_attrs.append(object.split(',')[0].lower().strip())

    return vg_classes, vg_attrs

def doit(detector, raw_images):
    with torch.no_grad():
        # Preprocessing
        inputs = []
        for raw_image in raw_images:
            image = detector.transform_gen.get_transform(raw_image).apply_image(raw_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append({"image": image, "height": raw_image.shape[0], "width": raw_image.shape[1]})
        images = detector.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = detector.model.backbone(images.tensor)
        
        # Generate proposals with RPN
        proposals, _ = detector.model.proposal_generator(images, features, None)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in detector.model.roi_heads.in_features]
        box_features = detector.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # (sum_proposals, 2048), pooled to 1x1
        
        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = detector.model.roi_heads.box_predictor(
                feature_pooled)
        rcnn_outputs = FastRCNNOutputs(
            detector.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            detector.model.roi_heads.smooth_l1_beta,
        )

        # Fixed-number NMS
        instances_list, ids_list = [], []
        probs_list = rcnn_outputs.predict_probs()
        boxes_list = rcnn_outputs.predict_boxes()
        for probs, boxes, image_size in zip(probs_list, boxes_list, images.image_sizes):
            for nms_thresh in np.arange(0.3, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image_size, 
                    score_thresh=args.score_thresh, nms_thresh=nms_thresh, topk_per_image=args.max_boxes
                )
                if len(ids) >= args.min_boxes:
                    break

            instances_list.append(instances)
            ids_list.append(ids)
        
        # Post processing for features
        features_list = feature_pooled.split(rcnn_outputs.num_preds_per_image) # (sum_proposals, 2048) --> [(p1, 2048), (p2, 2048), ..., (pn, 2048)]
        roi_features_list = []
        for ids, features in zip(ids_list, features_list):
            roi_features_list.append(features[ids].detach())
        
        # Post processing for bounding boxes (rescale to raw_image)
        raw_instances_list = []
        for instances, input_per_image, image_size in zip(
                instances_list, inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                raw_instances = detector_postprocess(instances, height, width)
                raw_instances_list.append(raw_instances)
        
        return raw_instances_list, roi_features_list

def get_labels(classes, class_names):
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
        return labels

def dump_features(output_path, detector, pathXid):
    img_paths, img_ids = zip(*pathXid)
    imgs = [cv2.imread(img_path) for img_path in img_paths]
    instances_list, features_list = doit(detector, imgs)

    for img, img_id, instances, features in zip(imgs, img_ids, instances_list, features_list):
        instances = instances.to('cpu')
        features = features.to('cpu')

        num_objects = len(instances)
        vg_classes, _ = get_vg_classes()
        item = {
            "img_id": img_id,
            "img_h": img.shape[0],
            "img_w": img.shape[1],
            "objects_id": instances.pred_classes.numpy().tolist(),  # int64
            "objects_conf": instances.scores.numpy().tolist(),  # float32
            "num_boxes": num_objects,
            "objects": get_labels(instances.pred_classes, vg_classes),
            "boxes": base64.b64encode(instances.pred_boxes.tensor.numpy()).decode(),  # float32
            "features": base64.b64encode(features.numpy()).decode()  # float32
        }
  
        with open(os.path.join(output_path, img_id + '.json'), 'w') as jsonfile:
            json.dump(item, jsonfile)

def _create_text_labels(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels

FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
def extract_feat(output_path, detector, pathXid):

    for start in tqdm.tqdm(range(0, len(pathXid), args.batchsize)):
        pathXid_trunk = pathXid[start: start + args.batchsize]
        dump_features(output_path, detector, pathXid_trunk)


def load_image_ids(img_root, split_dir):
    """images in the same directory are in the same split"""
    pathXid = []
    img_root = os.path.join(img_root, split_dir)
    for name in os.listdir(img_root):
        idx = name.split(".")[0]
        pathXid.append(
                (
                    os.path.join(img_root, name),
                    idx))
    if split_dir == 'val2014':
        print("Place the features of minival in the front of val2014 tsv.")
        # Put the features of 5000 minival images in front.
        minival_img_ids = set(json.load(open('data/mscoco_imgfeat/coco_minival_img_ids.json')))
        a, b = [], []
        for item in pathXid:
            img_id = item[1]
            if img_id in minival_img_ids:
                a.append(item)
            else:
                b.append(item)
        assert len(a) == 5000
        assert len(a) + len(b) == len(pathXid)
        pathXid = a + b
        assert len(pathXid) == 40504
    return pathXid

def build_model():
    # Build model and load weights.
    if args.weight == 'mask':
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(
            D2_ROOT, "configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        print("Load the Mask RCNN weight for ResNet101, pretrained on MS COCO segmentation. ")
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl"
    elif args.weight == 'obj':
        print("Load the Faster RCNN weight for ResNet101, pretrained on MS COCO detection.")
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join(
            D2_ROOT, "configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl"
    elif args.weight == 'vg':
        cfg = get_cfg() # Renew the cfg file
        cfg.merge_from_file(os.path.join(
            D2_ROOT, "configs/VG-Detection/faster_rcnn_R_101_C4_caffemaxpool.yaml"))
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
        cfg.INPUT.MIN_SIZE_TEST = 600
        cfg.INPUT.MAX_SIZE_TEST = 1000
        cfg.MODEL.RPN.NMS_THRESH = 0.7
        cfg.MODEL.WEIGHTS = "http://nlp.cs.unc.edu/models/faster_rcnn_from_caffe_attr_original.pkl"

    else:
        assert False, "no this weight"
    detector = DefaultPredictor(cfg)
    return detector


if __name__ == "__main__":
    pathXid = load_image_ids(args.data_path, args.split)     # Get paths and ids
    detector = build_model()
    output_path = os.path.join(args.output_path, 'bua_vg_frcnn_resnet101')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    extract_feat(output_path, detector, pathXid)