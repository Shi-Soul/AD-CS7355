import torch
import torchvision
from torchvision import tv_tensors
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.transforms import v2, ToTensor
from torch.utils.data import DataLoader, Dataset
from pprint import pprint
from torchmetrics.detection import MeanAveragePrecision
import os
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys 
sys.path.append(os.path.dirname(__file__))
from cfg import *

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the training and validation datasets
test_dataset = YOLODataset(root="../carla_object_detection_dataset3/valid",transforms=transforms_val)
# test_dataset = YOLODataset(root="../carla_object_detection_dataset2/test",transforms=transforms_val)

# Define the data loaders
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Define the model
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 7  # 6 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.to(device)
model.load_state_dict(torch.load("my_weight49.pth"))
model.eval()
# 加载测试Metric
map_metric = MeanAveragePrecision(box_format='xyxy',iou_type='bbox',class_metrics=True)

with torch.no_grad():
    for images, targets in test_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        prediction = model(images)
        mAP=map_metric(prediction,targets)

# 计算总mAP
total_mAP=map_metric.compute()
print("mAP:",total_mAP['map'].item())
# pprint(total_mAP) #可以看到更详细的mAP指标，如map_50,map_75,map_class等
