## 作业一 自动驾驶中的2D目标检测
在自动驾驶系统中，感知模块是数据流的起点。准确，高效的环境感知对于自动驾驶安全和效率有着重要意义。本次作业中，你将完成感知模块最简单的任务：利用从CARLA前视摄像头中收集的数据，利用pytorch框架训练一个模型，进行2D目标检测。

### 1. 环境配置
本次作业可以在交我算或者自己的电脑上完成。
#### 1.1 交我算
由于交我算平台环境配置非常麻烦，我们已经为大家准备好了开箱即用的container
  1. 将CARLA_OD上传到交我算 π2.0 Home 目录，并解压

        ```
        $ cd ~ && unzip CARLA_OD
        ```

  2. 在公共文件夹/lustre/share/class/CS7355 中复制用于目标检测的 singularity 容器 carla_project1_img.sif 到合适的位置

  3. 定位到 carla_project1_img.sif 所在的目录，使用 Singularity 启动带有 NVIDIA GPU 支持的 CARLA 容器：

        ```
        Apptainer> singularity shell --nv carla_project1_img.sif
        ```

  4. 在Singularity中cd到解压后的CARLA_OD目录，例如：

        ```
        Apptainer> cd ~/CARLA_OD/torchOD
        ```

  5. 即可在Singularity环境下运行代码（注意要用python3），例如

        ```
        Apptainer> python3 detect.py
        ```

#### 1.2 本地配置（建议使用Anaconda，将依赖装在之前的carla环境中，以免后面的作业重复安装）
1. 系统要求：Windows/Linux
2. 显卡要求：显存最低为4G，推荐使用显存为6G以上的显卡
3. 需要安装CUDA，cuDNN
   Windows: https://blog.csdn.net/jhsignal/article/details/111401628
   Linux: https://blog.csdn.net/weixin_37926734/article/details/123033286
4. Python版本：建议环境为Python3.8
5. 安装torch，torchvision： https://pytorch.org/get-started/locally/
6. 安装其他依赖项 `pip install -r requirments.txt`

### 2. 数据集
- 本次作业采用的是YOLO格式的数据集，在carla_object_detection_dataset3下面。分为train和valid两部分。
- 每张图片对应一个txt格式的label。txt每一行代表了一个物体，第一个数字为类别，其余四个数字分别为物体框的x_center, y_center, w, h。
- 数据集共有六个类别，分别是'motorbike', 'pedestrian', 'traffic_light', 'traffic_sign', 'vehicle', 'bike'

### 3. Pytorch介绍
PyTorch是一个开源的深度学习框架，由Facebook的人工智能研究团队（FAIR）开发。PyTorch以其灵活性、易用性和强大的功能，广泛应用于研究和产业界，特别受计算机视觉、自然语言处理和强化学习领域的欢迎。在深度学习领域赢得了广泛的认可。本次作业需要用到Pytorch中的torch和torchvision两个库。

#### torch

torch是PyTorch的核心库，它提供了一系列用于深度学习和张量计算的工具。下面是torch的一些主要特点和组件：

- 张量（Tensor）：torch.Tensor是PyTorch的基本数据结构，类似于NumPy的多维数组，但它支持在GPU上进行加速计算。张量可以用于存储和操作数据，例如模型的输入和输出、模型的权重等。

- 自动微分（Autograd）：torch.autograd提供了自动微分功能，能够自动计算张量的梯度。这对于训练神经网络是至关重要的，因为它可以自动地进行反向传播。

- 神经网络（nn）：torch.nn模块提供了构建深度神经网络所需的层和函数。这包括常用的层类型（如全连接层、卷积层、循环层等）和激活函数（如ReLU、Sigmoid等）。

- 优化（optim）：torch.optim模块包含了各种优化算法，如SGD、Adam等，用于训练神经网络。

- 并行计算：PyTorch支持多GPU计算，可以通过torch.nn.DataParallel或torch.nn.parallel.DistributedDataParallel实现模型的并行训练。

- 序列化：PyTorch提供了保存和加载模型的功能，可以通过torch.save和torch.load函数实现模型的序列化和反序列化。

#### torchvision
torchvision是一个与PyTorch紧密结合的库，专注于计算机视觉领域。它提供了以下功能：

- 预训练模型：torchvision.models提供了一些经典的预训练模型，如ResNet、VGG等，可以用于迁移学习或微调。

- 数据集（datasets）：torchvision.datasets提供了常用的计算机视觉数据集，如CIFAR、ImageNet等，方便加载和使用。

- 图像转换（transforms）：torchvision.transforms提供了一系列图像预处理函数，如裁剪、旋转、归一化等，用于数据增强和预处理。

- 数据加载器（DataLoader）：结合PyTorch的torch.utils.data.DataLoader，可以方便地批量加载和处理图像数据。

- 实用工具（utils）：包括一些实用工具，如模型保存与加载、图像和视频的读写等。

### 4. 代码介绍
#### 4.1 系统结构
为了减轻同学们的工作量，也便于进行评测，我们提供了基础的训练框架，固定了训练所采用的模型。本次作业所需代码存放于torchOD文件夹下，包括以下文件（夹）：
##### 4.1.1 train.py , 训练模型所需代码
- 运行: `python train.py`
- 加载现有权重（可选）：`model.load_state_dict(torch.load("my_weights.pth"))` （133行）
- 自定义数据集位置：`train_dataset = ..., val_dataset = ... `（109行）
- 设置输出权重名称：`torch.save(model.state_dict(), "my_weight"+str(begin_epoch+num_epochs)+".pth")` （186行）

- 为了提升模型效果，你可以尝试：
  - 修改dataloader的batchsize `train_loader = DataLoader(...,batch_size=2,...)` （114行）
  - 训练不同的epoch轮数 `num_epochs` （137行）
  - 尝试其他的optimizer,学习率等 `optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)` （127行）
  - 加入不同的transform来进行数据增强 （90行）

##### 4.1.2 test.py , 测试模型再某个数据集下准确率所需代码（也是最后用于模型评测的代码）
- 运行： `python test.py`
- 运行前，请记得修改数据集路径（93行），加载训练好的权重（108行）

##### 4.1.3 detect.py 用于实际检测的代码
- 运行：`python detect.py`
- 运行前，请记得修改输入，输出文件夹路径（53，54行），加载训练好的权重（26行）

- `testimg` 中提供有三张检测样例，是detect.py 的默认输入
- `test_output3`：`detect.py`的默认输出文件夹


每个代码文件内都有详细的注释，请同学们认真阅读代码，完成此次作业。
需要修改的代码都在train.py中 TODO
运行的命令

#### 4.2 模型架构
本次作业使用采用ResNet50作为backbone的FasterRCNN模型来完成目标检测任务。

FasterRCNN于2016年被任少卿，何恺明，Ross Girshick和孙剑提出，经由R-CNN和Fast RCNN发展而来，是一种非常经典的二阶段目标检测模型，具有非常快的检测速度。
原文链接：https://arxiv.org/abs/1506.01497
参考资料：https://zhuanlan.zhihu.com/p/31426458

#### 4.3 训练loss
在 PyTorch 中，fasterrcnn_resnet50_fpn 模型的损失是由以下几部分组成的（可在代码中用`pprint(loss_dict_train)`查看）：
1. **分类损失（Classification Loss）**:
    - 公式: 
    $
    L_{cls} = - \sum_{i} y_{i} \log(p_{i})
    $
    - 其中，$y_{i}$ 是真实类别的 one-hot 编码，$p_{i}$ 是模型预测的对应类别的概率。这个损失函数针对所有预测的边界框计算交叉熵损失。

2. **边界框回归损失（Box Regression Loss）**:
    - 公式: 
    $
    L_{box} = \text{SmoothL1Loss}(t, t^*)
    $
    - 其中，$t$ 是模型预测的边界框参数，$t^*$ 是与真实边界框对应的参数。Smooth L1 Loss 定义为:
    $
    \text{SmoothL1Loss}(x) = \begin{cases} 
    0.5x^2 & \text{if } |x| < 1 \\
    |x| - 0.5 & \text{otherwise}
    \end{cases}
    $

3. **对象性损失（Objectness Loss）**:
    - 公式: 
    $
    L_{obj} = -\log(p_{obj})
    $
    - 其中，$p_{obj}$ 是模型预测的边界框为物体的概率。对于每个候选框，这个损失评估模型区分前景和背景的能力。


4. **RPN 边界框回归损失（RPN Box Regression Loss）**:
    - 与边界框回归损失类似，但针对的是RPN生成的候选框。
    - 公式: 
    $
    L_{rpn, box} = \text{SmoothL1Loss}(t^{rpn}, t^{rpn*})
    $

总损失是上述损失的加权和：
$
L = \lambda_{cls} L_{cls} + \lambda_{box} L_{box} + \lambda_{obj} L_{obj} + \lambda_{rpn, box} L_{rpn, box}
$
其中，$\lambda_{cls}$, $\lambda_{box}$, $\lambda_{obj}$, 和 $\lambda_{rpn, box}$ 是各部分损失的权重。

#### 4.4 评价指标
本次作业的评价指标为测试集上的平均准确率mAP，关于mAP请参考：
https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

#### 4.5 Notes：
- **请将训练完成后效果最好的权重命名为best.pt**
- **请在训练完成后使用 detect.py 完成对testimg中样例的检测，存放到test_output3下**
  
### 5. 作业提交
完成作业后，请把以下几个文件压缩为 **.zip** 格式，命名为"学号_姓名.zip" ，提交到Canvas
- train.py
- test_output3
- best.pt

我们将在未公布的测试集上对模型进行评测，根据mAP完成作业评分。
**请注意保存你的权重，我们将会在后续作业中继续使用**

**请确保你的作业为独立完成，我们将采取各种方式对提交代码进行查重，一经确认为抄袭，该次作业计0分！**
