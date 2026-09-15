# 用户使用说明书

**基于飞桨的 CIFAR-10 图像分类识别系统**

本说明书面向非计算机专业用户，手把手教你完成安装、训练和使用。不需要编程基础也能按步骤操作；涉及命令的部分只需复制粘贴运行。

---

## 1. 这个软件是做什么的？

它能识别图片里是下面 10 种物体中的哪一种，并给出可信程度（置信度）：

| 英文 | 中文 | 英文 | 中文 |
|---|---|---|---|
| airplane | ✈️ 飞机 | dog | 🐕 狗 |
| automobile | 🚗 汽车 | frog | 🐸 青蛙 |
| bird | 🐦 鸟 | horse | 🐴 马 |
| cat | 🐱 猫 | ship | 🚢 船 |
| deer | 🦌 鹿 | truck | 🚚 卡车 |

> ⚠️ 注意：本系统只认识这 10 类物体。上传人物、食物、风景等其他图片时，它也会"勉强"给出 10 类中的一个答案，但结果不可信。

## 2. 安装与配置

### 2.1 准备环境

- 一台装有 **Windows / Linux / macOS** 的电脑（有 NVIDIA 显卡更快，没有也能用）；
- 安装 **Python 3.8 或更新版本**（推荐 Anaconda/Miniconda）。

### 2.2 安装依赖（约 10 分钟）

打开命令行（Windows 按 `Win+R` 输入 `cmd`），进入本项目文件夹后执行：

```bash
pip install -r requirements.txt
```

### 2.3 安装飞桨框架（二选一）

```bash
# 有 NVIDIA 显卡的用户（推荐，训练快 15 倍以上）
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/

# 没有独显的用户
pip install paddlepaddle==3.3.1
```

> ⚠️ **两个只能装一个**。如果都装过，请先执行 `pip uninstall paddlepaddle paddlepaddle-gpu` 再重装，否则会互相覆盖导致 GPU 失效。

### 2.4 检查安装是否成功

```bash
python -c "import paddle; print(paddle.__version__); print(paddle.get_device())"
```

显示版本号且设备为 `gpu:0`（或 `cpu`）即为成功。

## 3. 数据准备（自动完成，无需操作）

首次训练或评估时，程序会**自动下载** CIFAR-10 数据集（约 170MB）到项目的 `data/cifar10/` 文件夹，并自动校验文件完整性。

- 下载需要几分钟，请耐心等待；
- 如果网络不通，可从官网手动下载 `cifar-10-python.tar.gz` 放入 `data/cifar10/` 文件夹。

## 4. 训练模型

> 已有训练好的 `outputs/best_model.pdparams` 文件的话，可跳过本节直接看第 6 节。

```bash
python train.py
```

- 有显卡约 30~60 分钟，无显卡约 5~10 小时；
- 训练中会实时显示进度（损失 loss 越来越小、准确率 acc 越来越高是正常现象）；
- 训练完成后自动保存最佳模型到 `outputs/best_model.pdparams`，并在 `outputs/results/train_curves.png` 生成训练曲线图。

**快速试用（不追求精度，约 1 分钟）**：

```bash
python train.py --smoke
```

## 5. 检验模型效果

```bash
python eval.py
```

运行后查看 `outputs/results/` 文件夹：

| 文件 | 内容 |
|---|---|
| `comparison_table.md` | 各类干扰条件下（噪声/旋转/裁剪/模糊）的准确率对比表 |
| `confusion_matrix.png` | 混淆矩阵热力图（颜色越深表示该类识别越准） |
| `classification_report.txt` | 每一类的精确率/召回率/F1 报告 |

## 6. 识别一张图片（两种方式任选）

### 方式一：图形界面（推荐，最简单）

```bash
python -m streamlit run app.py
```

浏览器会自动打开网页（如未打开，手动访问 `http://localhost:8501`）：

1. 点击 **"浏览文件"** 或把图片**拖拽**到上传框；
2. 几秒后右侧显示 **Top-3 预测结果**（最可能的类别 + 置信度柱状图）；
3. 展开底部折叠区可查看全部 10 类的概率分布。

### 方式二：命令行

```bash
python predict.py --image 我的图片.jpg
```

输出示例：

```
==================================================
图片: 我的图片.jpg
Top-3 预测结果:
  1. automobile(汽车)  置信度  91.24%
  2. truck(卡车)       置信度   6.85%
  3. ship(船)          置信度   0.97%
==================================================
```

**识别效果更好的技巧**：上传"单个物体、居中、背景简单"的图片；物体占画面比例大一些；避免严重倾斜或过度模糊的照片。

## 7. 常见问题解答（FAQ）

**Q1：提示"未找到模型文件"怎么办？**
先训练一次：`python train.py`（或快速版 `python train.py --smoke`）。训练完成后再打开界面。

**Q2：`pip install` 很慢或超时？**
换国内镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

**Q3：训练时提示显存不足（CUDA out of memory）？**
把批次调小：`python train.py --batch-size 64`（还不够就改 32）。

**Q4：下载 CIFAR-10 失败？**
检查网络；或从 `https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz` 手动下载，放入 `data/cifar10/` 文件夹。

**Q5：识别结果明显不对？**
① 确认图片属于 10 类之一；② 换一张清晰、居中的图片试试；③ 查看第 5 节的混淆矩阵，某些类别（如猫/狗）本身容易混淆，属正常现象。

**Q5-2：上传的图片不属于这 10 类会怎样？**
系统内置了「其他」判定（开集识别）：当最高置信度低于 50%，或模型内部的 logit 分值低于 4.0 时，
界面会显示黄色警告并把它归为**「其他」**，而不是硬给一个 10 类里的答案。
注意：与 CIFAR-10 图像风格接近的平滑色块图仍可能被"自信地"误判，这是此类方法的已知局限。

**Q6：界面打不开 / 端口被占用？**
换端口运行：`python -m streamlit run app.py --server.port 8502`

**Q7：怎么把系统搬到别的电脑？**
整个项目文件夹直接拷贝即可。只需在新电脑重装依赖（第 2 节）；`data/cifar10/` 和 `outputs/` 一起拷贝可免去重新下载和训练。

**Q8：想识别更多类别的物体怎么办？**
需要更换/扩充训练数据并重新训练（修改 `config.py` 中的类别表并准备对应数据集），可参考项目技术文档 `docs/technical_doc.md`。

**Q9：我不懂编程，怎么了解这个系统到底做了什么？**
直接打开 `cifar10_practice.ipynb` 文件阅读即可——它是一份图文并茂的实践报告，
已经保存好了所有运行结果（样本图片、训练曲线、混淆矩阵、预测输出），
不需要运行任何代码就能看懂整个流程。安装 Jupyter 后可执行：

```bash
python -m jupyter notebook cifar10_practice.ipynb
```

**Q10：为什么我重新训练之后，识别结果反而变差了？**
因为**训练会覆盖模型文件**。`python train.py` 在验证准确率提升时会覆盖
`outputs/best_model.pdparams`。如果训练中途被中断（比如只跑了 4 轮就 Ctrl+C），
留下的就是一个还没训练好的半成品模型，识别自然不准。

**好消息是现在不会丢模型了**：
- 训练启动时会打印覆盖警告；
- 覆盖前会自动把旧模型备份为 `outputs/best_model_backup_<时间戳>.pdparams`；
- 每个 epoch 还会另存一份 `outputs/last_model.pdparams`。

如果发现模型被覆盖了，从 `outputs/` 里找到备份文件，重命名回
`best_model.pdparams` 即可恢复。

> 💡 **重要提醒**：只是想"看看识别效果"或"做演示"时，**不要运行 `train.py`**。
> 训练脚本会重写模型文件。日常使用只需要 `predict.py` / `app.py` / `eval.py`。

**Q11：界面提示 `识别失败：...but received uint8` 该怎么办？**
这是飞桨框架运行模式导致的预处理报错（静态图模式下不接受 uint8 像素数组）。
**当前版本已经修复**（改用自实现的图像转换 + 自动切换运行模式）。
如果你看到这条报错，只需**重启界面**即可：

```bash
# 先按 Ctrl+C 停掉正在运行的界面，然后重新启动
python -m streamlit run app.py
```

> 注意：Streamlit 会缓存已加载的模型和已导入的代码，
> **修改代码或替换模型文件后必须重启界面**才会生效。
> 如果确认代码是最新的仍报错，请把 `outputs/logs/run.log` 的最后几十行发出来，
> 里面有完整的错误堆栈，便于快速定位。

## 8. 文件夹说明

| 位置 | 内容 |
|---|---|
| `cifar10_practice.ipynb` | **完整实践 Notebook**：从数据预览、模型搭建、训练曲线、评估、鲁棒性测试到预测的全流程演示（已含运行结果，可直接阅读，无需运行代码） |
| `data/cifar10/` | 训练数据（自动下载） |
| `outputs/best_model.pdparams` | 训练好的模型（**最重要的文件**） |
| `outputs/results/` | 评估报告、曲线图、混淆矩阵 |
| `outputs/logs/run.log` | 运行日志（报错时可查看） |
| `docs/user_manual.md` | 本说明书 |
| `docs/technical_doc.md` | 技术文档（面向开发者） |
