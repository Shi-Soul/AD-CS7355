

### Try tuning transforms
Exp
```

TRAIN_BATCHSIZE = 4

transforms = v2.Compose([
    v2.Resize((640,640)),
    # v2.RandomResizedCrop(size=(640,640), antialias=True),
    # TODO:
    v2.RandomHorizontalFlip(p=0.5),
    # v2.RandomPhotometricDistort(p=0.5),
    # v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
    v2.RandomIoUCrop(),


    # 
    v2.SanitizeBoundingBoxes(), #移除退化的boundingbox
])

```

```
Epoch 42/50 Training...
Epoch 42/50 Evaluating...
Epoch 42/50, Train Loss: 0.183614, Validation mAP: 0.384923, Train Time: 34.41s, Eval Time: 6.57s
Epoch 43/50 Training...
Epoch 43/50 Evaluating...
Epoch 43/50, Train Loss: 0.177568, Validation mAP: 0.395700, Train Time: 35.00s, Eval Time: 6.39s
Epoch 44/50 Training...
Epoch 44/50 Evaluating...
Epoch 44/50, Train Loss: 0.176043, Validation mAP: 0.374159, Train Time: 35.53s, Eval Time: 6.60s
Epoch 45/50 Training...
Epoch 45/50 Evaluating...
Epoch 45/50, Train Loss: 0.175446, Validation mAP: 0.368404, Train Time: 33.85s, Eval Time: 6.07s
Epoch 46/50 Training...
Epoch 46/50 Evaluating...
Epoch 46/50, Train Loss: 0.170652, Validation mAP: 0.382776, Train Time: 32.62s, Eval Time: 6.27s
Epoch 47/50 Training...
Epoch 47/50 Evaluating...
Epoch 47/50, Train Loss: 0.178573, Validation mAP: 0.387945, Train Time: 34.48s, Eval Time: 6.49s
Epoch 48/50 Training...
Epoch 48/50 Evaluating...
Epoch 48/50, Train Loss: 0.176166, Validation mAP: 0.402906, Train Time: 36.90s, Eval Time: 6.44s
Epoch 49/50 Training...
Epoch 49/50 Evaluating...
Epoch 49/50, Train Loss: 0.169224, Validation mAP: 0.391925, Train Time: 33.11s, Eval Time: 6.14s
```



Exp
```
TRAIN_BATCHSIZE = 4

transforms = v2.Compose([
    v2.Resize((640,640)),
    # v2.RandomResizedCrop(size=(640,640), antialias=True),
    # TODO:
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomPhotometricDistort(p=0.5),
    # v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
    v2.RandomIoUCrop(),


    # 
    v2.SanitizeBoundingBoxes(), #移除退化的boundingbox
])


```

```
Epoch 32/50 Training...
Epoch 32/50 Evaluating...
Epoch 32/50, Train Loss: 0.201843, Validation mAP: 0.376722, Train Time: 37.15s, Eval Time: 6.75s
Epoch 33/50 Training...
Epoch 33/50 Evaluating...
Epoch 33/50, Train Loss: 0.196292, Validation mAP: 0.355726, Train Time: 34.91s, Eval Time: 6.68s
Epoch 34/50 Training...
Epoch 34/50 Evaluating...
Epoch 34/50, Train Loss: 0.196320, Validation mAP: 0.381086, Train Time: 34.81s, Eval Time: 6.87s
Epoch 35/50 Training...
Epoch 35/50 Evaluating...
Epoch 35/50, Train Loss: 0.194502, Validation mAP: 0.379778, Train Time: 35.69s, Eval Time: 6.60s
Epoch 36/50 Training...
Epoch 36/50 Evaluating...
Epoch 36/50, Train Loss: 0.195570, Validation mAP: 0.387250, Train Time: 35.45s, Eval Time: 6.69s
Epoch 37/50 Training...
Epoch 37/50 Evaluating...
Epoch 37/50, Train Loss: 0.195503, Validation mAP: 0.360588, Train Time: 34.60s, Eval Time: 6.69s
Epoch 38/50 Training...
Epoch 38/50 Evaluating...
Epoch 38/50, Train Loss: 0.192365, Validation mAP: 0.372847, Train Time: 34.61s, Eval Time: 7.44s
Epoch 39/50 Training...
Epoch 39/50 Evaluating...
Epoch 39/50, Train Loss: 0.189221, Validation mAP: 0.370675, Train Time: 34.96s, Eval Time: 6.69s
Epoch 40/50 Training...
Epoch 40/50 Evaluating...
Epoch 40/50, Train Loss: 0.194235, Validation mAP: 0.367554, Train Time: 33.97s, Eval Time: 6.70s
Epoch 41/50 Training...
Epoch 41/50 Evaluating...
Epoch 41/50, Train Loss: 0.190700, Validation mAP: 0.371277, Train Time: 33.37s, Eval Time: 6.66s
Epoch 42/50 Training...
Epoch 42/50 Evaluating...
Epoch 42/50, Train Loss: 0.199245, Validation mAP: 0.370901, Train Time: 36.34s, Eval Time: 6.70s
Epoch 43/50 Training...

```




Exp
```
TRAIN_BATCHSIZE = 4

transforms = v2.Compose([
    v2.Resize((640,640)),
    # v2.RandomResizedCrop(size=(640,640), antialias=True),
    # TODO:
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomPhotometricDistort(p=1),
    # v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
    v2.RandomIoUCrop(),


    # 
    v2.SanitizeBoundingBoxes(), #移除退化的boundingbox
])
```

```

Epoch 44/50 Training...
Epoch 44/50 Evaluating...
Epoch 44/50, Train Loss: 0.191890, Validation mAP: 0.389351, Train Time: 36.18s, Eval Time: 6.47s
Epoch 45/50 Training...
Epoch 45/50 Evaluating...
Epoch 45/50, Train Loss: 0.187659, Validation mAP: 0.372175, Train Time: 36.64s, Eval Time: 6.42s
Epoch 46/50 Training...
Epoch 46/50 Evaluating...
Epoch 46/50, Train Loss: 0.191698, Validation mAP: 0.380203, Train Time: 36.77s, Eval Time: 6.54s
Epoch 47/50 Training...
Epoch 47/50 Evaluating...
Epoch 47/50, Train Loss: 0.189582, Validation mAP: 0.396137, Train Time: 36.32s, Eval Time: 6.28s
Epoch 48/50 Training...
Epoch 48/50 Evaluating...
Epoch 48/50, Train Loss: 0.187434, Validation mAP: 0.393206, Train Time: 37.59s, Eval Time: 6.62s
Epoch 49/50 Training...
Epoch 49/50 Evaluating...
Epoch 49/50, Train Loss: 0.185033, Validation mAP: 0.392753, Train Time: 38.26s, Eval Time: 6.25s
```


4 bs, main dr
```
TRAIN_BATCHSIZE = 4

transforms = v2.Compose([
    # v2.Resize((640,640)),
    v2.RandomResizedCrop(size=(640,640), antialias=True),
    # TODO:
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomPhotometricDistort(p=1),
    v2.RandomIoUCrop(),


    # 
    v2.SanitizeBoundingBoxes(), #移除退化的boundingbox
])
```
Loss 下去了, mAP没上去, 不行啊

```
Epoch 2/50, Train Loss: 0.308318, Validation mAP: 0.139277, Train Time: 36.25s, Eval Time: 7.09s
Epoch 3/50 Training...
Epoch 3/50 Evaluating...
Epoch 3/50, Train Loss: 0.299827, Validation mAP: 0.163476, Train Time: 34.92s, Eval Time: 7.10s
Epoch 4/50 Training...
Epoch 4/50 Evaluating...
Epoch 4/50, Train Loss: 0.260312, Validation mAP: 0.168928, Train Time: 34.36s, Eval Time: 6.61s
Epoch 5/50 Training...
Epoch 5/50 Evaluating...
Epoch 5/50, Train Loss: 0.243947, Validation mAP: 0.181369, Train Time: 34.25s, Eval Time: 7.01s
Epoch 6/50 Training...
Epoch 6/50 Evaluating...
Epoch 6/50, Train Loss: 0.249202, Validation mAP: 0.181515, Train Time: 33.84s, Eval Time: 6.75s
Epoch 7/50 Training...
Epoch 7/50 Evaluating...
Epoch 7/50, Train Loss: 0.224746, Validation mAP: 0.194958, Train Time: 33.51s, Eval Time: 6.88s
Epoch 8/50 Training...
Epoch 8/50 Evaluating...
Epoch 8/50, Train Loss: 0.218703, Validation mAP: 0.205410, Train Time: 34.54s, Eval Time: 6.78s
Epoch 9/50 Training...
Epoch 9/50 Evaluating...
Epoch 9/50, Train Loss: 0.223761, Validation mAP: 0.215644, Train Time: 35.25s, Eval Time: 6.98s
Epoch 10/50 Training...
Epoch 10/50 Evaluating...
Epoch 10/50, Train Loss: 0.217706, Validation mAP: 0.239499, Train Time: 34.36s, Eval Time: 6.55s
Epoch 11/50 Training...
Epoch 11/50 Evaluating...
Epoch 11/50, Train Loss: 0.200133, Validation mAP: 0.243156, Train Time: 34.28s, Eval Time: 6.24s
```


还是train epoch少了, 加DR必须是增大了training才见效果

4 bs, use only `v2.RandomIoUCrop(),`
```
Epoch 0/10, Train Loss: 0.700625, Validation mAP: 0.091218
Epoch 1/10 Training...
Epoch 1/10 Evaluating...
Epoch 1/10, Train Loss: 0.435292, Validation mAP: 0.161462
Epoch 2/10 Training...
Epoch 2/10 Evaluating...
Epoch 2/10, Train Loss: 0.377475, Validation mAP: 0.203311
Epoch 3/10 Training...
Epoch 3/10 Evaluating...
Epoch 3/10, Train Loss: 0.332771, Validation mAP: 0.215232
Epoch 4/10 Training...
Epoch 4/10 Evaluating...
Epoch 4/10, Train Loss: 0.316180, Validation mAP: 0.238443
Epoch 5/10 Training...
Epoch 5/10 Evaluating...
Epoch 5/10, Train Loss: 0.290503, Validation mAP: 0.264657
Epoch 6/10 Training...
Epoch 6/10 Evaluating...
Epoch 6/10, Train Loss: 0.263790, Validation mAP: 0.298093
Epoch 7/10 Training...
Epoch 7/10 Evaluating...
Epoch 7/10, Train Loss: 0.260776, Validation mAP: 0.300723
Epoch 8/10 Training...
Epoch 8/10 Evaluating...
Epoch 8/10, Train Loss: 0.246127, Validation mAP: 0.310481
Epoch 9/10 Training...
Epoch 9/10 Evaluating...
Epoch 9/10, Train Loss: 0.246502, Validation mAP: 0.327509



Epoch 24/50 Training...
Epoch 24/50 Evaluating...
Epoch 24/50, Train Loss: 0.173450, Validation mAP: 0.406175, Train Time: 34.34s, Eval Time: 6.34s
Epoch 25/50 Training...
Epoch 25/50 Evaluating...
Epoch 25/50, Train Loss: 0.165497, Validation mAP: 0.397725, Train Time: 34.32s, Eval Time: 6.76s
Epoch 26/50 Training...
Epoch 26/50 Evaluating...
Epoch 26/50, Train Loss: 0.169197, Validation mAP: 0.402863, Train Time: 37.05s, Eval Time: 5.92s

Epoch 29/50, Train Loss: 0.159216, Validation mAP: 0.412164, Train Time: 34.46s, Eval Time: 6.16s
Epoch 30/50 Training...
Epoch 30/50 Evaluating...
Epoch 30/50, Train Loss: 0.164157, Validation mAP: 0.408460, Train Time: 33.40s, Eval Time: 6.30s
Epoch 31/50 Training...
Epoch 31/50 Evaluating...
Epoch 31/50, Train Loss: 0.160813, Validation mAP: 0.411151, Train Time: 33.11s, Eval Time: 6.10s
Epoch 32/50 Training...
Epoch 32/50 Evaluating...
Epoch 32/50, Train Loss: 0.160675, Validation mAP: 0.412031, Train Time: 35.78s, Eval Time: 6.37s
Epoch 33/50 Training...
Epoch 33/50 Evaluating...
Epoch 33/50, Train Loss: 0.154091, Validation mAP: 0.408751, Train Time: 36.35s, Eval Time: 5.90s
Epoch 34/50 Training...
Epoch 34/50 Evaluating...
Epoch 34/50, Train Loss: 0.155444, Validation mAP: 0.417362, Train Time: 36.28s, Eval Time: 6.27s
Epoch 35/50 Training...



Epoch 37/50, Train Loss: 0.147129, Validation mAP: 0.419932, Train Time: 36.59s, Eval Time: 6.29s
Epoch 38/50 Training...
Epoch 38/50 Evaluating...
Epoch 38/50, Train Loss: 0.148577, Validation mAP: 0.431669, Train Time: 34.14s, Eval Time: 6.28s
Epoch 39/50 Training...
Epoch 39/50 Evaluating...
Epoch 39/50, Train Loss: 0.148855, Validation mAP: 0.419941, Train Time: 35.33s, Eval Time: 6.26s
Epoch 40/50 Training...
Epoch 40/50 Evaluating...
Epoch 40/50, Train Loss: 0.140701, Validation mAP: 0.433824, Train Time: 33.67s, Eval Time: 6.39s


Epoch 47/50 Training...
Epoch 47/50 Evaluating...
Epoch 47/50, Train Loss: 0.137269, Validation mAP: 0.433363, Train Time: 35.77s, Eval Time: 5.88s
Epoch 48/50 Training...
Epoch 48/50 Evaluating...
Epoch 48/50, Train Loss: 0.132558, Validation mAP: 0.431459, Train Time: 35.81s, Eval Time: 6.30s
Epoch 49/50 Training...
Epoch 49/50 Evaluating...
Epoch 49/50, Train Loss: 0.128677, Validation mAP: 0.431462, Train Time: 35.28s, Eval Time: 6.29s
```

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
