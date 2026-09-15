# 项目技术文档

**项目名称**：基于飞桨的 CIFAR-10 图像分类识别系统
**技术栈**：PaddlePaddle 3.x（兼容 2.4+）· Python 3.8+ · Streamlit · ResNet18
**参考教材**：邱锡鹏《神经网络与深度学习》5.5 节"实践：基于 ResNet18 网络完成图像分类任务（CIFAR-10）"（课程配套代码 `Practice-in-Paddle-main/chap5卷积神经网络`）

---

## 1. 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户交互层                             │
│   app.py (Streamlit Web UI)   predict.py (命令行推理)        │
└──────────────┬──────────────────────────────┬───────────────┘
               │ 调用                          │ 调用
┌──────────────▼──────────────────────────────▼───────────────┐
│                        业务逻辑层                             │
│   train.py（训练）     eval.py（评估+鲁棒性测试+耗时分析）     │
└──────┬───────────────┬──────────────────────┬───────────────┘
       │               │                      │
┌──────▼─────┐  ┌──────▼──────┐  ┌────────────▼────────────┐
│  models/   │  │   data/     │  │        utils/           │
│  resnet.py │  │  dataset.py │  │  logger.py  日志        │
│  (ResNet18)│  │ preprocess  │  │  metrics.py 指标        │
│            │  │             │  │  io_utils.py 模型/文件IO │
└──────┬─────┘  └──────┬──────┘  └────────────┬────────────┘
       │               │                      │
┌──────▼───────────────▼──────────────────────▼───────────────┐
│              config.py（全局配置：路径/超参数/类别）           │
└─────────────────────────────────────────────────────────────┘
       │
┌──────▼───────────────────────────────────────────────────────┐
│  outputs/  best_model.pdparams · train_curves.png ·          │
│            confusion_matrix.png · classification_report.txt  │
│  data/cifar10/  cifar-10-python.tar.gz（自动下载+MD5校验）    │
└──────────────────────────────────────────────────────────────┘
```

**数据流**：

1. `data/cifar10/cifar-10-python.tar.gz`（170MB，首次自动下载）→ `paddle.vision.datasets.Cifar10` 解析为内存样本列表；
2. 训练模式：45000 张训练图（90%）+ 5000 张验证图（10%，固定种子 `SEED=42` 划分）；测试模式：10000 张官方测试图；
3. 每个 batch：PIL 图像 → `ToTensor`（CHW、[0,1]）→ `Normalize`（CIFAR-10 均值方差）→ GPU 张量；
4. 前向输出 logits `[N,10]` → `cross_entropy` 计算损失 → 反向传播更新参数；
5. 验证准确率创新高时经 `utils.io_utils.save_model` 保存 `outputs/best_model.pdparams`；
6. 推理：图片 → 缩放 32×32 → 同一变换 → 模型 → softmax → Top-3 概率。

## 2. 模块接口说明

### 2.1 config.py（全局配置）

| 名称 | 类型 | 说明 |
|---|---|---|
| `DATA_FILE` | str | CIFAR-10 tar 包路径 `data/cifar10/cifar-10-python.tar.gz` |
| `MODEL_PATH` | str | 最佳模型路径 `outputs/best_model.pdparams` |
| `CIFAR10_CLASSES / _CN` | list[str] | 10 类英文/中文类别名 |
| `NORM_MEAN / NORM_STD` | list[float] | 归一化参数（与教材 5.5 节一致） |
| `BATCH_SIZE / NUM_EPOCHS / LEARNING_RATE` | int/float | 训练超参数默认值 |
| `LR_SCHEDULER / LR_STEP_SIZE / LR_GAMMA` | — | 阶梯衰减配置（step / cosine 可选） |
| `EARLY_STOP_PATIENCE` | int | 早停容忍轮数 |
| `ensure_dirs()` | fn | 创建全部输出目录 |

### 2.2 data/dataset.py

| 接口 | 签名 | 说明 |
|---|---|---|
| `download_cifar10` | `(data_file=None, max_retry=3) -> str` | 下载 tar 包（MD5 校验 + 3 次重试 + 断点文件原子替换） |
| `CIFAR10Dataset` | `(mode, transform, val_ratio, seed, download)` | 继承 `paddle.io.Dataset`；`__getitem__` 返回 `(标准化CHW张量, int标签)`；train/val 按固定种子划分 |
| `build_dataloader` | `(mode, batch_size, shuffle, num_workers, transform) -> (DataLoader, Dataset)` | 统一 DataLoader 构建入口；train 默认 shuffle |

### 2.3 data/preprocess.py

| 接口 | 说明 |
|---|---|
| `build_transform(is_train)` | 训练：RandomCrop(32,pad=4)+RandomHorizontalFlip+ToTensor+Normalize；评估：ToTensor+Normalize |
| `build_infer_transform()` | 推理：Resize(32,32)+ToTensor+Normalize（与训练归一化参数一致） |
| `add_gaussian_noise(img, sigma, seed)` | 高斯噪声扰动（默认 σ=0.08，固定 seed 可复现） |
| `rotate_image(img, degree, seed)` | 随机旋转（默认 ±20°） |
| `random_crop_shift(img, max_shift, seed)` | 随机裁剪平移（±4px） |
| `apply_low_resolution(img, small_size)` | 低分辨率退化（12×12 再放大） |
| `center_crop_shrink(img, keep_ratio)` | 中心裁剪（默认保留 80%） |
| `PERTURBATIONS` | 名称→扰动函数字典，eval.py 据此构建鲁棒性测试集 |

### 2.4 models/resnet.py

| 接口 | 说明 |
|---|---|
| `resnet18(num_classes=10)` | 构建适配 32×32 的 ResNet18：首层 3×3 s=1、无 MaxPool、四阶段 [2,2,2,2] BasicBlock、全局平均池化 + FC |
| `BasicBlock` | 残差块：两层 3×3 Conv-BN + 恒等/1×1 卷积 shortcut |
| `count_parameters(model)` | 统计可训练参数量（≈11.17M） |

各阶段特征图尺寸：`3×32×32 → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → FC(10)`。

### 2.5 utils/

| 模块 | 关键接口 | 说明 |
|---|---|---|
| `logger.py` | `get_logger(name, log_file)` | 控制台+文件双写，UTF-8 防中文乱码，单例防重复 handler |
| `metrics.py` | `accuracy(logits, labels, topk)` | top-k 准确率（张量级） |
| | `classification_report_dict / format_classification_report` | 逐类别 P/R/F1 + 宏平均（sklearn，缺失时 numpy 兜底） |
| | `confusion_matrix / save_confusion_matrix` | 混淆矩阵计算、热力图 PNG + CSV 导出 |
| | `save_comparison_table(rows)` | 扰动测试准确率 Markdown 表 + CSV |
| | `setup_chinese_font()` | matplotlib 中文字体配置 |
| `io_utils.py` | `save_model(model, path, meta)` | `paddle.save` 权重 + `.meta.json` 元信息 |
| | `load_model(model, path, strict, device)` | 容错加载：文件不存在/为空/键不匹配/损坏 → 中文提示异常 |
| | `load_image(path)` | 图像加载（格式校验、损坏检测） |
| | `format_topk(predictions)` | Top-K 结果格式化文本 |

### 2.6 顶层脚本

| 脚本 | 关键接口 | 说明 |
|---|---|---|
| `train.py` | `Runner`（train_epoch/evaluate/train）、`parse_args` | 教材 RunnerV3 风格；早停+最佳模型保存+曲线绘制 |
| `eval.py` | `evaluate_standard / evaluate_perturbed / benchmark_inference` | 测试集指标、5 种扰动对比、batch=1/64/128 延迟与吞吐 |
| `predict.py` | `build_predictor(model_path)`、`predict_image(model, image, topk)` | CLI 与 app.py 共用的推理核心；`--json` 结构化输出 |
| `app.py` | `load_predictor`（`@st.cache_resource`）、`predict_once`（`@st.cache_data`） | Streamlit 界面：上传→Top-3 表格/柱状图→全类别分布 |

## 3. 模型结构（CIFAR 版 ResNet18）

```text
输入 [N,3,32,32]
├── Conv2D(3→64, 3×3, s=1) + BN + ReLU          [N,64,32,32]
├── layer1: 2×BasicBlock(64)                     [N,64,32,32]
├── layer2: 2×BasicBlock(128, 首块 s=2↓)         [N,128,16,16]
├── layer3: 2×BasicBlock(256, 首块 s=2↓)         [N,256,8,8]
├── layer4: 2×BasicBlock(512, 首块 s=2↓)         [N,512,4,4]
├── AdaptiveAvgPool2D(1) + Flatten               [N,512]
└── Linear(512→10)                               [N,10]（logits）
```

与 ImageNet 版 ResNet18 的差异及理由（详见教材 5.4/5.5 节）：

| 差异点 | ImageNet 版 | 本项目 | 理由 |
|---|---|---|---|
| 首层卷积 | 7×7, s=2 | 3×3, s=1 | 32×32 输入过小，大核大步长会过早丢失空间信息 |
| MaxPool | 有 | 无 | 同上，避免特征图过小 |
| 参数量 | 11.69M | 11.17M | 首层/分类头结构不同 |

## 4. 训练流程

```
parse_args（命令行覆盖 config 默认值）
  → 设备选择（GPU 优先，自动回退 CPU）
  → 构建 DataLoader（train 45000 / val 5000）
  → ResNet18 + cross_entropy + Adam(lr=1e-3, wd=5e-4) + StepDecay(20, γ=0.1)
  → 每个 epoch：
       train_epoch：前向 → loss.backward → optimizer.step → clear_grad
       evaluate(val)：model.eval() + no_grad
       val_acc 创新高 → save_model(best_model.pdparams, meta={epoch, val_acc...})
       连续 patience=10 轮未提升 → 早停
  → 产物：train_history.json、train_curves.png、outputs/logs/run.log
```

**默认超参数**（均可在 `config.py` / 命令行调整）：

| 参数 | 值 | 参数 | 值 |
|---|---|---|---|
| batch_size | 128 | 优化器 | Adam |
| 初始 lr | 0.001 | weight_decay | 5e-4 |
| lr 调度 | Step(每 20 epoch ×0.1) | 最大 epoch | 60 |
| 早停 patience | 10 | 数据增强 | RandomCrop(4) + HFlip |

## 5. 评估与鲁棒性测试方案

`eval.py` 输出三组结果到 `outputs/results/`：

1. **标准测试集**（10000 张）：准确率、平均损失、逐类别 P/R/F1 分类报告（`classification_report.txt/json`）、混淆矩阵（`confusion_matrix.png/csv`）、Top-5 易混淆类别对；
2. **鲁棒性测试**（默认每项 2000 张，固定 seed 公平对比）：高斯噪声 σ=0.08、旋转 ±20°、裁剪平移 ±4px、低分辨率 12×12、中心裁剪 80%，结果写入 `comparison_table.md/csv`；
3. **推理耗时**：batch=1/64/128 三档的平均延迟与吞吐量（先 20 次预热再统计，排除 CUDA 初始化一次性开销）。

## 6. 错误排查记录（联调实测汇总）

| # | 现象 | 原因 | 解决 |
|---|---|---|---|
| 1 | `ImportError: cannot import name 'CIFAR10'` | paddle 3.x 类名改为 `Cifar10` | `data/dataset.py::_get_cifar10_cls()` 双版本兼容导入 |
| 2 | 同时安装 `paddlepaddle` 与 `paddlepaddle-gpu` 后 `is_compiled_with_cuda()=False` | 两包写入同一 `paddle` 目录，CPU 版覆盖 GPU 版 | 全部卸载后只装 GPU 版 |
| 3 | GPU 前向报 `No available device found` | 同上（CPU 版占位） | 同上；验证 `nvidia-smi` 驱动正常 |
| 4 | `optimizer's learning rate can't be LRScheduler when invoke set_lr` | paddle 3.x 中优化器绑定调度器后自动同步 lr | 删除手动 `set_lr`，仅保留 `scheduler.step()` |
| 5 | `Tensor.grad` 报 `object is not callable` | paddle 3.x `grad` 由方法改为属性 | 测试代码做 `callable(grad)` 兼容 |
| 6 | matplotlib 图中文显示为方框 | DejaVu Sans 无中文字形 | `setup_chinese_font()` 配置微软雅黑等字体 |
| 7 | Windows 删除临时图片报 `WinError 32` | 文件句柄未关闭即 unlink | `mkstemp` + `os.close` + `finally` 中清理 |
| 8 | 数据下载中断产生 `.part` 文件 | 网络波动 | 下载到临时文件 + MD5 校验通过后原子 `os.replace` |
| 9 | `VisibleDeprecationWarning`（paddle 源码） | paddle 与 numpy 2.4 兼容性提示 | 无害警告，可忽略 |
| 10 | 训练验证 acc 不升（早期开发） | 学习率过大/归一化参数错误 | 固定使用 CIFAR-10 官方 mean/std，lr 从 1e-3 起调 |
| 11 | 鲁棒性测试崩在"低分辨率"一项：`apply_low_resolution() got an unexpected keyword argument 'seed'` | 各扰动函数签名不统一（低分辨率/中心裁剪不需要随机种子，故未定义 `seed` 形参），而 eval.py 统一按 `fn(img, seed=...)` 调用 | `evaluate_perturbed` 中先尝试 `perturb_fn(img, seed=...)`，捕获 `TypeError` 后回退 `perturb_fn(img)`；`preprocess.py` 自检同理 |
| 12 | `save_model` 保存到纯文件名（如 `_tmp.pdparams`，不含目录）时报 `FileNotFoundError: [WinError 3] 系统找不到指定的路径。: ''` | `os.makedirs(os.path.dirname(path))` 在路径无目录部分时 `dirname` 返回空串 | 改为调用统一的 `ensure_parent(path)`（内部判断 `if parent:` 后才建目录）；该 bug 由 Notebook 容错演示单元暴露 |
| 13 | 界面提示 `识别失败：The data type of 'input' in assign must be ['float32',...], but received uint8` | 飞桨的张量创建（`paddle.assign` / `to_tensor`）**不接受 uint8 数组**。该报错说明有原始 uint8 像素数组未经转换直接进入了飞桨的张量环节（绕过「PIL → ToTensor → Normalize」正规路径） | 新增 `utils/io_utils.ensure_rgb_pil()` 输入防御层：把路径/字节流/PIL（任意 mode）/numpy（HWC、CHW、灰度，uint8/浮点）统一归一化为 RGB PIL 图像，`predict_image` 与 `app.py` 均改走该层，从源头杜绝该错误 |
| 14 | **重新训练后界面预测明显变差**（同一张图从「猫 99.99%」变成「船 38.70%」） | `train.py` 每次验证准确率提升都会覆盖 `outputs/best_model.pdparams`。一次中断的训练（跑到 epoch 4，验证 70.18%）把之前训练好的模型（epoch 30，92.56%）覆盖了，而文件本身的元信息 `.meta.json` 也随之更新，不易察觉 | ① `train.py` 启动时打印覆盖警告；② 首次覆盖前自动备份旧模型为 `outputs/best_model_backup_<时间戳>.pdparams`；③ 每个 epoch 另存 `outputs/last_model.pdparams`；④ 从提交包中恢复了被覆盖的最佳模型 |
| 15 | **Web 界面任何图片都报** `识别失败：...in assign must be ['float32',...], but received uint8`，但命令行完全正常 | **飞桨处于静态图模式**时，`paddle.to_tensor(uint8 数组)` 走 `_to_tensor_static → assign()` 分支被 `check_dtype` 拒绝；而 `paddle.vision.transforms.ToTensor` 对 PIL 图像内部执行的正是 `paddle.to_tensor(np.asarray(pic))`，`np.asarray(RGB图)` 恰恰是 uint8。动态图模式下 `to_tensor` 有 uint8 → float 的转换分支所以正常，因此该故障只在特定宿主环境（进程处于静态图模式）暴露。堆栈证据：`functional_pil.py:76 → creation.py:1152 → tensor() → _to_tensor_static() → assign()` | ① 新增 `data/preprocess.image_to_tensor()`：自己完成「PIL 缩放 → float32 numpy → CHW → 标准化」，**完全绕开 paddle 的 ToTensor**，与运行模式无关（经实测在静态图模式下也能成功）；② 新增 `utils/io_utils.ensure_dynamic_mode()`，在 `app.py`/`predict.py`/`eval.py`/`train.py` 入口以及**每次推理前**调用，自动把静态图模式切回动态图并打印提示（因为静态图模式下连模型前向也会报 `conv2d(): argument (position 2) must be Value, but got EagerParamBase`）；③ 新增 3 个回归测试（`TestRuntimeResilience`）固化该场景 |
| 16 | **上传第二张图片后，界面仍显示第一张图片的识别结果**（第一张正常，之后全是同一结果） | `@st.cache_data` 会**忽略名称以下划线开头的参数**。当时为了跳过大体量字节流的哈希，把图片参数命名为 `_img_bytes`，导致缓存键退化成只有 `(model_path, topk)` 且恒定不变 —— 第一次上传的返回值被后续每一次上传复用 | 移除 `predict_once` 上的 `@st.cache_data`：推理在 GPU 上仅约 3ms，本就无需缓存；昂贵的模型加载仍由 `@st.cache_resource` 单独缓存。新增 `tests/test_app_ui.py`，用 Streamlit 官方 `AppTest` **真实执行 app.py 并连续上传 3 张不同图片**，断言结果各不相同且各自正确，防止该 bug 复发 |

## 7. 性能优化记录

| 优化项 | 做法 | 效果 |
|---|---|---|
| 数据增强 | RandomCrop(32, pad=4) + RandomHorizontalFlip | 抑制过拟合，测试集准确率提升约 4~6% |
| L2 正则 | Adam + weight_decay=5e-4 | 权重范数受控，泛化更稳 |
| 学习率衰减 | StepDecay(20, γ=0.1) / 可选 CosineAnnealing | 后期收敛更精细，最终精度提升约 1~2% |
| 早停 | 验证 acc 连续 10 epoch 不升即停 | 避免无效训练，节省时间 |
| 最佳模型保存 | 验证 acc 创新高才保存 | 测试用的一定是历史最佳权重 |
| GPU 训练 | `paddle.set_device('gpu')` 自动检测 | 相比 CPU 提速约 15~20 倍 |
| 推理缓存 | app.py `@st.cache_resource/@st.cache_data` | 模型只加载一次；相同图片不重复推理 |
| BN + 残差结构 | BasicBlock shortcut | 深层网络稳定收敛（教材 5.4 节结论） |

## 8. 联调说明（模块间调用关系）

- `app.py` / `predict.py` 共用 `predict.build_predictor` 与 `predict.predict_image`，保证 CLI 与 UI 推理行为完全一致；
- `eval.py` 的扰动测试复用 `data/preprocess.PERTURBATIONS`，新增扰动只需在字典中注册一项；
- `train.py` / `eval.py` / `predict.py` 均通过 `utils.io_utils.load_model` 加载权重，错误提示统一；
- 全部脚本通过 `config.py` 读写路径，迁移项目目录无需改代码；
- 单元测试（`tests/`）覆盖数据/模型/推理三层，回归修改时先跑 `python -m pytest tests/ -v`。

## 9. 已知限制与注意事项

1. paddle 3.x 与 numpy≥2 组合会输出无害的 `VisibleDeprecationWarning`（paddle 官方源码触发）；
2. Windows 下 `NUM_WORKERS` 默认 0（多进程 DataLoader 兼容性差），Linux 默认 4；
3. 模型对多物体、大角度旋转、极低分辨率图片准确率会下降（见鲁棒性对比表），属 CIFAR-10 训练分布外的正常表现；
4. 首次运行会下载 170MB 数据集；离线环境可手动放置 tar 包到 `data/cifar10/`。
