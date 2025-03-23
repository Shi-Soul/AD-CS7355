
需要 CUDA 12.4

在gpu29上run



Baseline Result

```bash
(carla) wjxie@gpu29-wjxie:~/wjxie/env/AD-CS7355/CARLA_OD/torchOD$ python train.py
Using device: cuda
/TinyNAS2024/wjxie/anaconda3/envs/carla/lib/python3.8/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/TinyNAS2024/wjxie/anaconda3/envs/carla/lib/python3.8/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1`. You can also use `weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" to /home/wjxie/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
100.0%
Epoch 0/5
Epoch 0/5, Train Loss: 0.6612064852796752, Validation mAP: 0.17526477575302124
Epoch 1/5
Epoch 1/5, Train Loss: 0.38884701527061005, Validation mAP: 0.2432290017604828
Epoch 2/5
Epoch 2/5, Train Loss: 0.3293461233634373, Validation mAP: 0.29640305042266846
Epoch 3/5
Epoch 3/5, Train Loss: 0.2812147793841773, Validation mAP: 0.3378617763519287
Epoch 4/5
Epoch 4/5, Train Loss: 0.25515636829980487, Validation mAP: 0.3451273441314697
```


