
需要 CUDA 12.4

在gpu29上run


### Try tuning transforms


8 batch size
```
Epoch 0/10
Epoch 0/10, Train Loss: 0.992581, Validation mAP: 0.070720
Epoch 1/10
Epoch 1/10, Train Loss: 0.573795, Validation mAP: 0.094233
Epoch 2/10
Epoch 2/10, Train Loss: 0.491974, Validation mAP: 0.119483
Epoch 3/10
Epoch 3/10, Train Loss: 0.438596, Validation mAP: 0.153211
Epoch 4/10
Epoch 4/10, Train Loss: 0.409807, Validation mAP: 0.171414
Epoch 5/10
Epoch 5/10, Train Loss: 0.395033, Validation mAP: 0.180470
Epoch 6/10
Epoch 6/10, Train Loss: 0.375817, Validation mAP: 0.193265
Epoch 7/10
Epoch 7/10, Train Loss: 0.353708, Validation mAP: 0.209517
Epoch 8/10
Epoch 8/10, Train Loss: 0.334651, Validation mAP: 0.222374
Epoch 9/10
Epoch 9/10, Train Loss: 0.323503, Validation mAP: 0.238336
```


add `    v2.RandomPhotometricDistort(p=1),`
```
Epoch 0/10
Epoch 0/10, Train Loss: 0.815773, Validation mAP: 0.097816
Epoch 1/10
Epoch 1/10, Train Loss: 0.498071, Validation mAP: 0.137055
Epoch 2/10
Epoch 2/10, Train Loss: 0.426972, Validation mAP: 0.179946
Epoch 3/10
Epoch 3/10, Train Loss: 0.382123, Validation mAP: 0.194174
Epoch 4/10
Epoch 4/10, Train Loss: 0.356438, Validation mAP: 0.204573
Epoch 5/10
Epoch 5/10, Train Loss: 0.335969, Validation mAP: 0.231087
Epoch 6/10
Epoch 6/10, Train Loss: 0.318716, Validation mAP: 0.236904
Epoch 7/10
Epoch 7/10, Train Loss: 0.304216, Validation mAP: 0.251997
Epoch 8/10
Epoch 8/10, Train Loss: 0.294906, Validation mAP: 0.276828
Epoch 9/10
Epoch 9/10, Train Loss: 0.285017, Validation mAP: 0.294115

```





add `    v2.RandomHorizontalFlip(p=0.5),`

```
Epoch 0/10
        Epoch 0/10, Train Loss: 0.768464, Validation mAP: 0.094540
Epoch 1/10
Epoch 1/10, Train Loss: 0.460149, Validation mAP: 0.172374
Epoch 2/10
Epoch 2/10, Train Loss: 0.398709, Validation mAP: 0.208836
Epoch 3/10
Epoch 3/10, Train Loss: 0.364259, Validation mAP: 0.233606
Epoch 4/10
Epoch 4/10, Train Loss: 0.338794, Validation mAP: 0.253526
Epoch 5/10
Epoch 5/10, Train Loss: 0.314116, Validation mAP: 0.285747
Epoch 6/10
Epoch 6/10, Train Loss: 0.303332, Validation mAP: 0.300502
Epoch 7/10
Epoch 7/10, Train Loss: 0.283487, Validation mAP: 0.314306
Epoch 8/10
Epoch 8/10, Train Loss: 0.272489, Validation mAP: 0.321428
Epoch 9/10
Epoch 9/10, Train Loss: 0.259684, Validation mAP: 0.340764
```

10 epoch, 4 batch size

```python
transforms = v2.Compose([
    # v2.Resize((640,640)),
    v2.RandomResizedCrop(size=(640,640), antialias=True),
    # TODO:


    # 
    v2.SanitizeBoundingBoxes(), #移除退化的boundingbox
])
```
```
Epoch 0/10
Epoch 0/10, Train Loss: 0.606546, Validation mAP: 0.087889
Epoch 1/10
Epoch 1/10, Train Loss: 0.419109, Validation mAP: 0.127669
Epoch 2/10
Epoch 2/10, Train Loss: 0.321105, Validation mAP: 0.155788
Epoch 3/10
Epoch 3/10, Train Loss: 0.308924, Validation mAP: 0.176821
Epoch 4/10
Epoch 4/10, Train Loss: 0.319598, Validation mAP: 0.188380
Epoch 5/10
Epoch 5/10, Train Loss: 0.256970, Validation mAP: 0.206651
Epoch 6/10
Epoch 6/10, Train Loss: 0.252792, Validation mAP: 0.216128
Epoch 7/10
Epoch 7/10, Train Loss: 0.252458, Validation mAP: 0.243605
Epoch 8/10
Epoch 8/10, Train Loss: 0.232711, Validation mAP: 0.254126
Epoch 9/10
Epoch 9/10, Train Loss: 0.216474, Validation mAP: 0.278443
```

### Baseline Result
5 epoch, 2 batch size
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


10 epoch, 4 batch size
```bash
Epoch 0/10
Epoch 0/10, Train Loss: 0.792431, Validation mAP: 0.081878
Epoch 1/10
Epoch 1/10, Train Loss: 0.463116, Validation mAP: 0.168855
Epoch 2/10
Epoch 2/10, Train Loss: 0.392102, Validation mAP: 0.198006
Epoch 3/10
Epoch 3/10, Train Loss: 0.350475, Validation mAP: 0.216997
Epoch 4/10
Epoch 4/10, Train Loss: 0.319959, Validation mAP: 0.275025
Epoch 5/10
Epoch 5/10, Train Loss: 0.297264, Validation mAP: 0.292917
Epoch 6/10
Epoch 6/10, Train Loss: 0.275230, Validation mAP: 0.322422
Epoch 7/10
Epoch 7/10, Train Loss: 0.255446, Validation mAP: 0.333265
Epoch 8/10
Epoch 8/10, Train Loss: 0.239868, Validation mAP: 0.346306
Epoch 9/10
Epoch 9/10, Train Loss: 0.230721, Validation mAP: 0.36397
```

10 epoch, 12 batch size
```bash
Epoch 0/10                                                                                                   [13/859]Epoch 0/10, Train Loss: 1.107006, Validation mAP: 0.012267
Epoch 1/10
Epoch 1/10, Train Loss: 0.679604, Validation mAP: 0.080725                                                           Epoch 2/10                                                                                                           Epoch 2/10, Train Loss: 0.563384, Validation mAP: 0.100720
Epoch 3/10
Epoch 3/10, Train Loss: 0.506111, Validation mAP: 0.124903
Epoch 4/10
Epoch 4/10, Train Loss: 0.483495, Validation mAP: 0.148418
Epoch 5/10
Epoch 5/10, Train Loss: 0.428043, Validation mAP: 0.171296
Epoch 6/10
Epoch 6/10, Train Loss: 0.423096, Validation mAP: 0.186358
Epoch 7/10
Epoch 7/10, Train Loss: 0.378521, Validation mAP: 0.207136
Epoch 8/10
Epoch 8/10, Train Loss: 0.373245, Validation mAP: 0.219661
Epoch 9/10
Epoch 9/10, Train Loss: 0.353998, Validation mAP: 0.234165
```
