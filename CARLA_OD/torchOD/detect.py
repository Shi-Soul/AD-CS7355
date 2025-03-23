import torch
import torchvision
from torchvision import tv_tensors
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torchvision.transforms import v2, ToTensor
import os
import numpy as np
from PIL import Image
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys 
sys.path.append(os.path.dirname(__file__))
from cfg import *

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型并设置为评估模式
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 7  # 6 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
# 加载你的训练权重
model.load_state_dict(torch.load("my_weight5.pth"))
model.to(device)
model.eval()

# 定义可视化函数
def visualize_prediction(image, prediction, threshold=0.5, save_path="output.png"):
    image = image.cpu()
    image = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    for element in prediction:
        boxes = element['boxes'].cpu().detach().numpy()
        labels = element['labels'].cpu().detach().numpy()
        scores = element['scores'].cpu().detach().numpy()
        for box, label, score in zip(boxes, labels, scores):
            if score > threshold:
                x_min, y_min, x_max, y_max = box
                width, height = x_max - x_min, y_max - y_min
                rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x_min, y_min, f"{label},{CARLA_CLASSES[label]}: {score:.2f}", color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 检测文件夹中的所有图片
image_folder = "testimg"
output_folder = "test_output3"
os.makedirs(output_folder, exist_ok=True)

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    output_path = os.path.join(output_folder, f"detected_{image_name}")
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = ToTensor()(img).to(device)
        with torch.no_grad():
            prediction = model([img_tensor])
        pprint(prediction)
        visualize_prediction(img_tensor, prediction, save_path=output_path)
        print(f"Processed {image_name}")
    except Exception as e:
        print(f"Error processing {image_name}: {e}")

print("Done!")
