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

# Define the dataset class
class YOLODataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])

        img = Image.open(img_path).convert("RGB")
        
        w, h = img.size
        img = ToTensor()(img)
        img = tv_tensors.Image(img)
        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f.readlines():
                cls, x_center, y_center, width, height = map(float, line.split())
                x_min = w * (x_center - width / 2)
                y_min = h * (y_center - height / 2)
                x_max = w * (x_center + width / 2)
                y_max = h * (y_center + height / 2)
                boxes.append([x_min, y_min, x_max, y_max])
                cls=int(cls)
                # 目标检测中，标签（label）0通常被保留用于表示背景类。为了适配该数据集格式，把0（bike）安排为6
                if cls==0:
                    cls=6
                labels.append(cls)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])

        # 如果boxes为空，则提供一个虚拟的边界框和标签   
        if boxes.size(0) == 0:
            # 如果为空，则设置area为一个空的张量
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)  # 虚拟的边界框
            labels = torch.tensor([0], dtype=torch.int64)  # 虚拟的标签，假设0是背景类
            area = torch.tensor([1], dtype=torch.float32)  # 虚拟的面积
        else:
            # 如果不为空，则正常计算area
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        # 将Boundingbox转换为tv_tensors
        boxes = tv_tensors.BoundingBoxes(boxes, format='XYXY',canvas_size=(h, w)) #type:ignore

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # 进行图像及BoundingBox变换
        if self.transforms is not None:
            img,target = self.transforms(img,target)  

        return img, target

    def __len__(self):
        return len(self.imgs)

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

# 将模型移动到指定设备
model.to(device)

# 加载之前训练的模型权重
model.load_state_dict(torch.load("my_weight5.pth"))
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
