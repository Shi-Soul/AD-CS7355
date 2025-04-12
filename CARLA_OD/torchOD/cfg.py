import os
import numpy as np
from PIL import Image
from pprint import pprint
import torch
import torchvision
from torchvision import tv_tensors
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.transforms import v2, ToTensor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision



# 定义检测类别


CARLA_CLASSES = ['background', 'motorbike', 'pedestrian', 'traffic_light', 'traffic_sign', 'vehicle', 'bike']

TRAIN_BATCHSIZE = 4

transforms = v2.Compose([
    v2.Resize((640,640)),
    # v2.RandomResizedCrop(size=(640,640), antialias=True),
    # TODO:
    # v2.RandomHorizontalFlip(p=0.5),
    # v2.RandomPhotometricDistort(p=0.5),
    # v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
    v2.RandomIoUCrop(),


    # 
    v2.SanitizeBoundingBoxes(), #移除退化的boundingbox
])


# 验证集的transform仅进行resize即可
transforms_val = v2.Compose([
    v2.Resize((640,640)),
    v2.SanitizeBoundingBoxes(), #移除退化的boundingbox
])



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
        # 转为tv_tensors便于后续Transform
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
                # 目标检测中，标签（label）0通常被保留用于表示背景类。为了适配该数据集格式，把原数据集中的0（bike）安排为6
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
