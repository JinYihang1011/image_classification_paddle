# 课程设计报告

## 基于飞桨的 CIFAR-10 图像分类识别系统

| 项目 | 信息 |
|---|---|
| 课程名称 | 人工智能实践 |
| 项目方向 | 基于飞桨（PaddlePaddle）的图像分类识别系统 |
| 深度学习框架 | PaddlePaddle 3.3.1（GPU 版，兼容 CPU） |
| 运行环境 | Windows 11 · NVIDIA GeForce RTX 5060 Laptop GPU (8GB) · Python 3.13 |
| 参考教材 | 邱锡鹏《神经网络与深度学习》5.4/5.5 节 ResNet 实践案例（课程配套代码 Practice-in-Paddle-main/chap5卷积神经网络） |

---

## 摘要

本项目基于百度飞桨（PaddlePaddle）深度学习框架，设计并实现了一套完整的 CIFAR-10 图像分类识别系统。系统使用 `paddle.nn` 原生算子搭建了适配 32×32 小分辨率输入的 ResNet18 残差网络，在 CIFAR-10 数据集（60000 张 32×32 彩色图，10 类物体）上完成了从数据自动下载与预处理、模型训练（数据增强、学习率衰减、早停、最佳模型保存）、多维度评估（准确率/精确率/召回率/F1/混淆矩阵）、鲁棒性测试（高斯噪声、旋转、裁剪、低分辨率等 5 种扰动场景）到 Streamlit 图形界面的全流程工程实现。项目采用多模块化结构（data / models / utils / tests / docs / report），模块间接口清晰、异常处理完善，并配套单元测试与三份文档（技术文档、用户说明书、本报告）。实测在 RTX 5060 GPU 上训练约 48 分钟（第 40 轮触发早停，最佳模型为第 30 轮、验证准确率 92.56%），测试集准确率 **91.75%**（宏平均 F1 0.9174，见 5.3 节实际运行结果），单张图片 GPU 推理延迟 3.23 ms（batch=128 时吞吐 4942 img/s），满足课程全部设计与验收要求。

**关键词**：图像分类；卷积神经网络；残差网络；PaddlePaddle；CIFAR-10；Streamlit

---

## 1. 绪论

### 1.1 项目背景

图像分类是计算机视觉的基础任务，也是工业界应用最广泛的 AI 能力之一（质检、安防、医疗影像、内容审核等）。深度学习出现后，卷积神经网络（CNN）大幅超越了传统手工特征方法；其中何恺明等提出的 ResNet 通过残差连接解决了深层网络退化问题，成为分类任务的标准骨干网络。

CIFAR-10 是学术界最常用的入门级图像分类基准数据集：10 类物体、32×32 彩色小图。它的分辨率低、类间差异小（如猫/狗、汽车/卡车），既能完整体验"数据—模型—训练—评估—部署"的全流程，又能在普通笔记本 GPU 上快速实验，非常适合作为课程设计载体。

### 1.2 项目意义

1. **工程完整性**：不同于只写一个训练脚本的课程作业，本项目覆盖课程要求的需求分析、界面设计、数据管理、模型搭建、测试分析、文档撰写全部环节，形成可直接交付的软件系统；
2. **框架实践**：深入使用飞桨的高层 API（`paddle.io.Dataset` / `paddle.io.DataLoader` / `paddle.vision.transforms` / `paddle.optimizer` / `paddle.save`），与课程教材《神经网络与深度学习》5.5 节案例的 API 风格保持一致，并在此基础上工程化扩展；
3. **可靠性意识**：通过鲁棒性测试、异常处理、单元测试等环节，体会"能跑通"与"可交付"之间的差距。

### 1.3 项目目标

- 在 CIFAR-10 测试集上达到 **90% 以上**的 Top-1 准确率；
- 提供可视化 Web 界面，上传图片即可得到 Top-3 类别与置信度；
- 系统在数据异常、模型缺失、图片损坏等非正常输入下不崩溃，给出明确提示；
- 输出完整的量化评估（指标 + 混淆矩阵 + 鲁棒性对比 + 耗时）与三份文档。

## 2. 需求分析

### 2.1 用户角色与需求

| 用户角色 | 核心需求 |
|---|---|
| 普通使用者 | 上传图片、查看识别结果与置信度；界面简单、出错有提示 |
| 实验者/助教 | 重新训练模型、复现实验结果、查看训练曲线与评估报告 |
| 开发维护者 | 模块化代码、接口文档、单元测试、日志可追溯 |

### 2.2 功能需求

1. **数据管理**：CIFAR-10 自动下载（MD5 校验、断点重试）、训练/验证集按固定种子划分、DataLoader 构建；
2. **模型训练**：支持命令行超参数调整；数据增强；Adam/Momentum 可选；学习率阶梯/余弦衰减；早停；验证集最优模型自动保存；
3. **模型评估**：准确率、精确率、召回率、F1 分类报告、混淆矩阵可视化与导出；
4. **鲁棒性测试**：构建高斯噪声/旋转/裁剪平移/低分辨率/中心裁剪 5 类扰动测试集并对比准确率；
5. **推理服务**：命令行单图推理（Top-3 + JSON 输出）与 Streamlit Web 界面；
6. **运行支撑**：日志记录、模型保存/加载、训练曲线绘制、单元测试。

### 2.3 非功能需求

- **可靠性**：文件不存在、格式错误、图片损坏、模型结构不匹配等异常均有中文提示，不崩溃；
- **可复现性**：固定随机种子（SEED=42），数据划分与实验结果可复现；
- **性能**：GPU 训练吞吐 ≥ 800 img/s；单张推理延迟 ≤ 10 ms；
- **可移植性**：路径集中于 `config.py`，CPU/GPU 自动检测，跨平台运行。

### 2.4 系统架构与模块划分

系统分为四层（详细架构图与接口表见 `docs/technical_doc.md`）：

```
用户交互层：app.py(Streamlit) / predict.py(CLI)
业务逻辑层：train.py(训练) / eval.py(评估+鲁棒性+耗时)
核心组件层：models/resnet.py · data/dataset.py · data/preprocess.py
           utils/logger.py · utils/metrics.py · utils/io_utils.py
配置与产物：config.py · outputs/（模型、日志、报告）· data/cifar10/
```

### 2.5 数据流

```
CIFAR-10 tar 包(自动下载+MD5校验)
   → paddle.vision.datasets.Cifar10 解析（train 50000 / test 10000）
   → 90/10 固定种子划分 train(45000)/val(5000)
   → transforms：RandomCrop+HFlip → ToTensor(CHW,[0,1]) → Normalize(CIFAR-10 均值方差)
   → DataLoader 批读取(GPU 张量)
   → ResNet18 前向 → cross_entropy → 反向传播
   → 验证最优权重 → outputs/best_model.pdparams
   → eval.py / predict.py / app.py 加载权重完成推理
```

## 3. 系统设计

### 3.1 模型选型

| 候选模型 | CIFAR-10 参考准确率 | 参数量 | 结论 |
|---|---|---|---|
| 简单 CNN（3 层卷积） | ~75% | ~0.3M | 精度不足 |
| VGG13 | ~92% | ~9.4M | 参数多、推理慢 |
| **ResNet18（选定）** | **~93%** | **11.17M** | 精度/复杂度平衡最佳，且为教材案例 |
| MobileNetV2 | ~91% | ~2.3M | 备选（轻量场景） |

选 **ResNet18**：教材 5.5 节即以 ResNet18 完成 CIFAR-10 分类，本项目沿用其结构思想并做 32×32 适配（首层 3×3 s=1、去 MaxPool、AdaptiveAvgPool 收尾）。

### 3.2 数据模型设计

- **数据集组织**：`data/cifar10/cifar-10-python.tar.gz`（唯一数据源，170MB，MD5=c58f…349a）；
- **内存表示**：`CIFAR10Dataset`（继承 `paddle.io.Dataset`）持有底层官方数据集对象与划分索引列表，`__getitem__` 返回 `(Tensor[3,32,32] float32, int)`；
- **划分策略**：`np.random.RandomState(42).permutation(50000)`，前 5000 为验证集，其余 45000 为训练集——保证任何机器上划分完全一致。

### 3.3 接口设计

各模块对外接口（函数签名、参数、返回值、异常约定）汇总表见 `docs/technical_doc.md` 第 2 节。设计原则：

1. **单一职责**：数据/模型/指标/IO 分离；
2. **共用核心**：CLI 与 UI 共用 `predict.predict_image`，评估与训练共用 `metrics` 与 `load_model`；
3. **异常契约**：底层抛带中文排查提示的明确异常类型（FileNotFoundError/ValueError/IOError/RuntimeError），上层捕获转译。

## 4. 系统实现

### 4.1 开发环境

Python 3.13.11（miniconda base）· PaddlePaddle 3.3.1 GPU（cu129，适配 RTX 50 系列 Blackwell 架构）· CUDA 12.9 运行时 · 其余依赖见 `requirements.txt`。

> 实施过程中发现 base 环境同时装有 CPU/GPU 两个飞桨包导致 GPU 失效（两包写同一 `paddle` 目录互相覆盖），全部卸载后重装 GPU 版解决——该问题已记入技术文档"错误排查记录"。

### 4.2 数据预处理与增强

训练集：`RandomCrop(32, padding=4)`（小幅平移扰动）+ `RandomHorizontalFlip`（CIFAR-10 水平翻转不改变语义）+ `ToTensor` + `Normalize(mean=[0.4914,0.4822,0.4465], std=[0.2023,0.1994,0.2010])`（与教材 5.5 节一致）。验证/测试/推理仅做 ToTensor+Normalize（推理额外 Resize 到 32×32），保证推理与训练分布一致。

### 4.3 模型搭建（关键代码）

```python
class BasicBlock(nn.Layer):
    """残差块：两层 3x3 Conv-BN + shortcut（下采样或通道不一致时用 1x1 卷积对齐）"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2D(out_channels)
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, 1, stride, bias_attr=False),
                nn.BatchNorm2D(out_channels))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)   # 残差相加
```

### 4.4 训练实现

训练循环沿用教材 RunnerV3 风格：每 epoch 训练一遍 + 验证一遍，验证准确率创新高即保存模型（`paddle.save` 至 `best_model.pdparams`，附 `.meta.json` 元信息），连续 10 个 epoch 未提升触发早停。默认 Adam(lr=1e-3, weight_decay=5e-4) + StepDecay(每 20 epoch ×0.1)，批大小 128。

### 4.5 用户界面实现

Streamlit 实现：`@st.cache_resource` 缓存模型（会话内只加载一次）、`@st.cache_data` 缓存相同图片的推理结果；上传组件限定 jpg/jpeg/png/bmp/webp；结果区展示 Top-3 表格 + 柱状图 + 全类别概率分布折叠区；模型缺失、图片损坏等异常均转换为 `st.error` 中文提示。

### 4.6 关键容错代码（示例）

```python
def load_model(model, path, strict=True, device=None):
    if not os.path.exists(path):
        raise FileNotFoundError("模型文件不存在: %s\n请先执行 `python train.py` ..." % path)
    if os.path.getsize(path) == 0:
        raise IOError("模型文件为空，可能上次训练中断导致损坏: %s" % path)
    state = paddle.load(path)          # 损坏时抛 RuntimeError + 中文提示
    missing = set(model.state_dict()) - set(state)
    if missing and strict:
        raise RuntimeError("权重与模型结构不匹配：缺失键 %d 个 ..." % len(missing))
    model.set_state_dict(state)
```

## 5. 系统测试

### 5.1 单元测试

`tests/` 目录 3 个测试文件共 **31 个用例**，全部通过（`python -m pytest tests/ -v`，实测 22.95s）：

| 文件 | 覆盖内容 | 用例数 |
|---|---|---|
| test_data.py | 变换流水线形状/数值范围/确定性、5 种扰动函数、数据划分确定性与无重叠 | 19 |
| test_model.py | 前向形状、各阶段特征图尺寸、参数量、反向传播、保存/加载往返一致、异常加载 | 8 |
| test_inference.py | 推理预处理、Top-K 结构与概率和、损坏/缺失/不支持格式异常 | 4 组 |

### 5.2 多类型测试数据集设计

在官方测试集（10000 张，基准）之外构建 5 种扰动测试集，模拟真实使用场景（每项默认 2000 张，固定 seed 保证公平对比）：

| 测试集 | 构造方法 | 模拟场景 |
|---|---|---|
| 高斯噪声 | 像素叠加 N(0, 0.08²) 噪声 | 低光照、传感器噪声 |
| 旋转 | 随机旋转 ±20°，空隙填黑 | 拍摄角度倾斜 |
| 裁剪平移 | 四周补黑后随机平移 ±4px | 目标偏离中心 |
| 低分辨率 | 缩至 12×12 再放大回 32×32 | 低清截图、远距小目标 |
| 中心裁剪 | 仅保留中心 80% 区域 | 主体被部分裁切 |

### 5.3 测试结果

（以下为系统实际运行 `python eval.py` 的自动输出，结果文件位于 `outputs/results/`。）

**（1）准确率对比表**——`comparison_table.md`（实测，RTX 5060 GPU，2026-09-14）：

| 测试集 | 准确率 | 相对基线下降 | 说明 |
|---|---|---|---|
| 标准测试集（无扰动，10000 张） | **91.75%** | — | baseline，平均损失 0.2838 |
| 随机裁剪平移（±4px，2000 张） | 91.55% | -0.20pp | 几乎无损：训练中 RandomCrop 增强生效 |
| 中心裁剪（80%，2000 张） | 88.05% | -3.70pp | 轻度下降：边缘信息丢失 |
| 旋转（±20°，2000 张） | 85.10% | -6.65pp | 中度下降：训练分布无旋转样本 |
| 高斯噪声（σ=0.08，2000 张） | 36.55% | -55.20pp | 大幅下降：对噪声敏感 |
| 低分辨率（12×12，2000 张） | 22.40% | -69.35pp | 大幅下降：细节纹理丢失严重 |

**（2）混淆矩阵**——`confusion_matrix.png`（按行归一化热力图）与 `confusion_matrix.csv`（原始计数）；对角线元素即各类正确率，非对角高亮区为易混淆类别对（如 cat↔dog、automobile↔truck），符合人类直觉。

**（3）分类报告**——`classification_report.txt`：逐类别 precision/recall/F1 与宏平均。

**（4）推理耗时**——batch=1/64/128 三档平均延迟与吞吐量（预热 20 次后统计，实测）：

| batch 大小 | 平均延迟 | 吞吐量 |
|---|---|---|
| 1 | 3.23 ms | 309.5 img/s |
| 64 | 19.04 ms | 3361.7 img/s |
| 128 | 25.90 ms | 4942.0 img/s |

另：`predict.py` 命令行单图预测端到端约 0.6 s（含模型加载与 GPU 初始化），纯推理仅 3.2 ms；实测 3 张样例图（cat/ship/horse）全部预测正确，Top-1 置信度分别为 99.99%、100.00%、88.92%。

### 5.4 结果分析与讨论

1. **基准性能**：ResNet18 在 CIFAR-10 测试集上达到 91.75% 的准确率（宏平均精确率/召回率/F1 均为 0.917x，各类均衡），接近该结构的公开最好水平（93~94%），验证了数据增强 + 学习率衰减 + 早停策略的有效性。训练在第 40 轮触发早停（最佳第 30 轮），避免了后期过拟合（第 30 轮后训练损失仍降但验证损失已回升），早停与最佳模型保存机制发挥了预期作用；
2. **鲁棒性**：裁剪平移（91.55%）与中心裁剪（88.05%）下准确率下降有限，说明随机裁剪增强确实提升了平移/裁剪不变性；旋转（85.10%）下降居中；高斯噪声（36.55%）与低分辨率（22.40%）下降剧烈，因为训练分布中不含这两类退化——CIFAR-10 图 32×32 的信息量本就有限，噪声或降到 12×12 后有效信噪比大幅流失。可通过在训练增强中加入高斯噪声（`ColorJitter`/自定义噪声层）与 `RandomRotation` 针对性改善；
3. **易混淆类别**：实测 Top-5 混淆对为 dog→cat（92 次）、cat→dog（63 次）、bird→airplane（33 次）、automobile→truck（32 次）、cat→deer（27 次），语义相近类别互混最多（猫狗双向互混占两者错误的大头），与人类直觉一致，可通过更大模型（ResNet34）或更强增强（AutoAugment、Mixup）改善；
4. **耗时**：GPU 上单张推理仅 3.23 ms，满足实时性要求；batch 从 1 增到 128 时吞吐量从 309.5 提升到 4942.0 img/s（16 倍），说明小批量场景 GPU 利用率不足，批量服务化部署时应凑批推理；

## 6. 可靠性与容错性设计

| 环节 | 容错措施 |
|---|---|
| 数据下载 | MD5 校验 + 3 次重试 + `.part` 临时文件 + 原子替换，损坏自动重下 |
| 模型加载 | 文件不存在/为空/键不匹配/损坏 → 分别给出针对性中文提示 |
| 图片输入 | 路径/目录/格式/损坏四类校验，`PIL img.load()` 提前解码暴露坏图 |
| 推理过程 | 未知异常兜底捕获，退出码区分错误类别（1 模型 / 2 图片 / 3 推理） |
| Web 界面 | 模型缺失引导训练、图片损坏提示重传、推理异常兜底，页面不白屏 |
| 训练中断 | 每 epoch 落盘最佳模型与历史 JSON；`--resume` 支持断点续训 |
| 日志追溯 | 全模块统一 logger，控制台 + `outputs/logs/run.log` 双写 |
| 回归保障 | 31 个单元测试覆盖数据/模型/推理核心链路 |

## 7. 总结与展望

### 7.1 工作总结

本项目完整实现了基于飞桨的 CIFAR-10 图像分类识别系统：以 `paddle.nn` 原生搭建 32×32 适配版 ResNet18，实现并验证了数据自动管理、训练策略（增强/衰减/早停/最佳保存）、多维评估、5 类扰动鲁棒性测试、推理耗时分析与 Streamlit 交互界面；项目结构模块化、接口清晰、异常处理完备，配有 31 个单元测试与技术文档/用户说明书/本报告三份文档，达到课程设计全部要求。开发过程中还实际排查了 CPU/GPU 包冲突、paddle 3.x API 变更（Cifar10 类名、LR 调度器、grad 属性化）等真实工程问题，记录于技术文档错误排查表。

### 7.2 展望

1. **精度提升**：引入 Mixup/CutMix、AutoAugment、标签平滑，或升级 ResNet34/SE-ResNet；
2. **鲁棒性增强**：在训练增强中加入高斯噪声与随机旋转，针对性提升扰动场景表现；
3. **功能扩展**：支持自定义数据集训练（替换 config 类别表即可）、批量图片推理、模型导出 ONNX 跨框架部署；
4. **工程深化**：Docker 容器化交付、GPU TensorRT 加速推理、Gradio 多模型对比界面。

## 附录 A：课程要求验收清单

| # | 课程要求 | 实现情况 | 验收方式 |
|---|---|---|---|
| 1 | 需求分析 | 本报告第 2 节 | 阅读 |
| 2 | 用户交互界面设计 | app.py（Streamlit：上传/Top-3/置信度/异常提示） | `python -m streamlit run app.py` |
| 3 | 数据模型构建与数据预处理 | data/dataset.py、data/preprocess.py | `python data/dataset.py` 自检 |
| 4 | AI 模型选型与搭建 | models/resnet.py（paddle.nn 原生 ResNet18） | `python models/resnet.py` 自检 |
| 5 | 核心算法设计与程序实现 | 残差结构+数据增强+lr 衰减+早停（train.py） | `python train.py --smoke` |
| 6 | 数据集组织与管理、文件读写 | 自动下载+MD5 校验+固定种子划分（dataset.py、io_utils.py） | 删除 data/ 后重跑 |
| 7 | 模型文件保存与加载 | outputs/best_model.pdparams + 容错 load_model | eval/predict 均实测 |
| 8 | 多模块项目结构 | data/models/utils/tests/docs/report 分层 | 查看目录树 |
| 9 | 模块接口设计、调用与联调 | technical_doc.md 第 2、8 节接口表 + 联调说明 | 阅读 |
| 10 | 训练与推理错误排查 | technical_doc.md 第 6 节 10 条排查记录 | 阅读 |
| 11 | 算法性能优化 | technical_doc.md 第 7 节优化记录（增强/衰减/早停等） | 对比消融 |
| 12 | 多类型测试数据集 | 标准集+高斯噪声+旋转+裁剪平移+低分辨率+中心裁剪 | `python eval.py` |
| 13 | 运行效果与性能分析 | 准确率对比表/混淆矩阵/分类报告/推理耗时（5.3 节） | 查看 outputs/results/ |
| 14 | 可靠性与容错性 | 本报告第 6 节措施 + 31 个单元测试 | `python -m pytest tests/ -v` |
| 15 | 技术文档 | docs/technical_doc.md | 阅读 |
| 16 | 用户使用说明书 | docs/user_manual.md（面向非技术用户） | 阅读 |
| 17 | 课程设计报告 | report/course_design_report.md | 本文档 |
| 18 | 实践 Notebook（教材风格，含真实运行输出） | cifar10_practice.ipynb（11 节 / 44 单元） | `python -m jupyter notebook cifar10_practice.ipynb` |

### 验收运行步骤（5 分钟快速版）

```bash
cd image_classification_paddle
python -m pytest tests/ -v          # ① 单元测试（31 个用例）
python eval.py                      # ② 评估+鲁棒性+耗时（需已训练模型）
python predict.py --image test_imgs/1_ship.png   # ③ 单图推理
python -m streamlit run app.py      # ④ Web 界面上传 test_imgs/ 中图片
python -m jupyter notebook cifar10_practice.ipynb  # ⑤ 查看完整实践 Notebook
```

### 可能遇到的问题与解决方案

| 问题 | 解决 |
|---|---|
| 显存不足 | `python train.py --batch-size 64`（或 32） |
| 数据下载失败 | 手动下载 tar.gz 放入 `data/cifar10/` |
| GPU 未启用 | 确认只装了一个 paddle 包；`python -c "import paddle;print(paddle.get_device())"` |
| 端口被占用 | `--server.port 8502` |
| 中文乱码 | 本项目输出已统一 UTF-8；若终端乱码执行 `chcp 65001` |

## 参考文献

1. 邱锡鹏. 神经网络与深度学习[M]. 北京：机械工业出版社, 2020.（第 5 章 卷积神经网络；5.4 节 ResNet；5.5 节实践案例）—— 课程教材
2. He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]// CVPR. 2016: 770-778.
3. Krizhevsky A. Learning Multiple Layers of Features from Tiny Images[R]. University of Toronto, 2009.
4. PaddlePaddle 官方文档. https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/Overview_cn.html
5. Streamlit 官方文档. https://docs.streamlit.io/
6. 百度飞桨. PaddlePaddle 图像分类实战教程. [https://aistudio.baidu.com/](https://cloud.baidu.com/article/3275942?_=1789378876110)
