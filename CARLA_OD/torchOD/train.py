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

import sys 
sys.path.append(os.path.dirname(__file__))
from cfg import *

CARLA_CLASSES = ['background', 'motorbike', 'pedestrian', 'traffic_light', 'traffic_sign', 'vehicle', 'bike']


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


# 请尝试加入不同的transform来进行数据增强
# reference: 
# https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html#sphx-glr-auto-examples-transforms-plot-transforms-e2e-py
# https://pytorch.org/vision/stable/transforms.html

# 检查CUDA是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the training and validation datasets 
train_dataset = YOLODataset(root="../carla_object_detection_dataset3/train", transforms=transforms)
val_dataset = YOLODataset(root="../carla_object_detection_dataset3/valid", transforms=transforms_val)

# Define the data loaders
# 可手动调整batch_size大小，如果爆显存了，请适当调小batch_size
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# 使用采用ResNet50作为backbone的FasterRCNN模型
model = fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 7  # 6 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Define the optimizer
# 你可以尝试其他的学习率等参数
# 你也可以尝试其他的optimizer，比如Adam，RMSprop等
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005) #type:ignore

# 将模型移动到指定设备
model.to(device)

# 加载之前训练的模型权重，若没有则注释该句，系统会自动下载在COCO数据集预训练的权重
# model.load_state_dict(torch.load("my_weights.pth"))

# Training and validation loop
begin_epoch = 0
num_epochs = 5
train_losses = []

# 定义评测指标
# 关于mAP：
# https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
# 关于torchmetrics中的mAP：
# https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
map_metric = MeanAveragePrecision(box_format='xyxy',iou_type='bbox',class_metrics=True)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+begin_epoch}/{num_epochs+begin_epoch}")
    model.train()
    train_loss = 0
    # 一次取一个batch进行训练
    for images, targets in train_loader:
         # 将数据移动到设备
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict_train = model(images, targets)
        # print("loss_dict_train: ", loss_dict_train)
        # 计算总loss
        losses:torch.Tensor = sum(loss for loss in loss_dict_train.values()) #type:ignore
        # 将优化器中的所有梯度设置为零，为计算新的梯度做准备
        optimizer.zero_grad()
        # 将总loss进行反向传播，计算出每个参数的梯度
        losses.backward()
        # 根据计算出的梯度更新模型的参数
        optimizer.step()
        train_loss += losses.item()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation，切换为评测模式
    model.eval() 
    with torch.no_grad():
        for images, targets in val_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            prediction = model(images)
            mAP=map_metric(prediction,targets)

    # 计算总mAP
    total_mAP=map_metric.compute()
    # pprint(total_mAP) #可以看到更详细的mAP指标，如map_50,map_75,map_class等
    print(f"Epoch {epoch+begin_epoch}/{num_epochs+begin_epoch}, Train Loss: {train_loss:.6f}, Validation mAP: {total_mAP['map'].item():.6f}")
    # 重置评测metric
    map_metric.reset()
# Save the model weights
torch.save(model.state_dict(), "my_weight"+str(begin_epoch+num_epochs)+".pth")

