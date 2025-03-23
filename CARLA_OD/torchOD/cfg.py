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

transforms = v2.Compose([
    v2.Resize((640,640)),
    # TODO:


    # 
    v2.SanitizeBoundingBoxes(), #移除退化的boundingbox
])


# 验证集的transform仅进行resize即可
transforms_val = v2.Compose([
    v2.Resize((640,640)),
    v2.SanitizeBoundingBoxes(), #移除退化的boundingbox
])