import time
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
from cfg import CARLA_CLASSES, transforms, transforms_val, YOLODataset, TRAIN_BATCHSIZE



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

print(f"{transforms=} \t | {transforms_val=}")
print(f"{len(train_dataset)=} \t | {len(val_dataset)=}")

# Define the data loaders
# 可手动调整batch_size大小，如果爆显存了，请适当调小batch_size
train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCHSIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

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
num_epochs = 50
train_losses = []

# 定义评测指标
# 关于mAP：
# https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
# 关于torchmetrics中的mAP：
# https://lightning.ai/docs/torchmetrics/stable/detection/mean_average_precision.html
map_metric = MeanAveragePrecision(box_format='xyxy',iou_type='bbox',class_metrics=True)

for epoch in range(num_epochs):
    t1 = time.time()
    print(f"Epoch {epoch+begin_epoch}/{num_epochs+begin_epoch} Training...")
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
    
    t2 = time.time()
    print(f"Epoch {epoch+begin_epoch}/{num_epochs+begin_epoch} Evaluating...")
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
    t3 = time.time()
    print(f"Epoch {epoch+begin_epoch}/{num_epochs+begin_epoch}, Train Loss: {train_loss:.6f}, Validation mAP: {total_mAP['map'].item():.6f}, Train Time: {t2-t1:.2f}s, Eval Time: {t3-t2:.2f}s")
    # 重置评测metric
    map_metric.reset()
# Save the model weights
    if epoch % 5 == 0 or epoch == num_epochs-1:
        torch.save(model.state_dict(), "my_weight"+str(begin_epoch+epoch)+".pth")

