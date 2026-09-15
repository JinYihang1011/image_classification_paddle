# 基于飞桨的 CIFAR-10 图像分类识别系统

> 东南大学《人工智能实践》课程设计：基于飞桨（PaddlePaddle）的图像分类识别系统
>
> 使用 **ResNet18**（适配 32×32 输入）在 **CIFAR-10** 上实现图像十分类，
> 提供训练 / 评估 / 鲁棒性测试 / 单图推理 / Streamlit 图形界面 全流程。

## ✨ 功能特性

- **自动数据管理**：CIFAR-10 自动下载（MD5 校验 + 断点重试），训练/验证集按固定种子划分
- **标准模型**：`paddle.nn` 原生搭建 ResNet18（首层 3×3、去掉 MaxPool，适配 32×32 小图）
- **完整训练策略**：数据增强（随机裁剪/翻转）、Adam/Momentum 可选、学习率衰减（阶梯/余弦）、早停、最佳模型自动保存
- **多维评估**：准确率、精确率、召回率、F1 分类报告、混淆矩阵（热力图 + CSV）
- **鲁棒性测试**：高斯噪声、旋转、裁剪平移、低分辨率、中心裁剪 5 种扰动测试集 + 推理耗时分析
- **图形界面**：Streamlit 上传图片，展示 Top-3 类别与置信度柱状图、全类别概率分布
- **开集识别**：非 10 类图片（噪声/无关图）自动归为「其他」，而不是硬给一个类别
- **容错设计**：模型加载/图片读取全链路中文异常提示，模块化结构、接口清晰
- **单元测试**：`tests/` 覆盖数据、模型、推理、界面四条链路（pytest / unittest 兼容）

## 📁 项目结构

```
image_classification_paddle/
├── config.py               # 全局配置：路径、超参数、类别名称
├── requirements.txt        # 依赖清单
├── cifar10_practice.ipynb  # ★ 完整实践 Notebook（已含真实运行输出，教材风格）
├── 课程设计汇报.pptx        # ★ 汇报用 PPT（14 页，含图表，附讲者备注）
├── train.py                # 训练主程序（命令行参数、早停、保存最佳模型）
├── eval.py                 # 评估：测试集指标 + 鲁棒性测试 + 推理耗时
├── predict.py              # 单张图片推理（Top-3 类别与置信度）
├── app.py                  # Streamlit 用户交互界面
├── run.sh                  # 一键运行脚本
├── data/
│   ├── dataset.py          # CIFAR-10 下载、train/val/test 划分、DataLoader
│   └── preprocess.py       # 数据增强、归一化、鲁棒性扰动函数
├── models/
│   └── resnet.py           # paddle.nn 原生 ResNet18（32×32 适配版）
├── utils/
│   ├── logger.py           # 日志（控制台 + 文件）
│   ├── metrics.py          # 准确率/精确率/召回率/F1/混淆矩阵
│   └── io_utils.py         # 模型保存加载、文件读写、异常处理
├── tests/                  # 单元测试（test_data / test_model / test_inference / test_app_ui）
├── docs/                   # 技术文档 + 用户使用说明书
├── report/                 # 课程设计报告
└── outputs/                # 运行产物：模型、日志、评估结果（自动生成）
```

> 📎 **可点击直达**：
> [汇报 PPT](课程设计汇报.pptx) ·
> [实践 Notebook](cifar10_practice.ipynb) ·
> [技术文档](docs/technical_doc.md) ·
> [用户使用说明书](docs/user_manual.md) ·
> [课程设计报告](report/course_design_report.md)

## 🚀 快速开始

### 1. 安装环境（Python 3.8+）

```bash
pip install -r requirements.txt
# GPU 用户（推荐，CUDA 12.8+ 显卡如 RTX 50 系）：
pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
# CPU 用户：
pip install paddlepaddle==3.3.1
```

### 2. 训练

> ⚠️ **重要**：训练会**覆盖** `outputs/best_model.pdparams`。如果只是想使用/演示
> 已经训练好的模型，**请跳过本节**，直接看第 3、4、5 节。
> （已内置保护：训练启动时会警告，覆盖前自动备份旧模型，并每个 epoch 另存
> `outputs/last_model.pdparams`。）

```bash
python train.py                    # 默认超参数（约 30~60 epoch，GPU 上约 1 小时）
python train.py --optimizer momentum --scheduler cosine   # 换优化器/调度器
python train.py --smoke            # 冒烟测试：1 分钟验证全流程可用
```

训练完成后自动生成：
- `outputs/best_model.pdparams` —— 验证集准确率最高的模型
- `outputs/last_model.pdparams` —— 最后一个 epoch 的模型（断点续训/对比用）
- `outputs/best_model_backup_<时间戳>.pdparams` —— 覆盖前自动备份的旧模型
- `outputs/results/train_curves.png` —— 损失/准确率/学习率曲线
- `outputs/results/train_history.json` —— 训练历史数据

### 3. 评估与鲁棒性测试

```bash
python eval.py                     # 测试集指标 + 混淆矩阵 + 5 种扰动测试 + 耗时分析
python eval.py --robust-num 1000   # 加速：每种扰动取 1000 张
```

### 4. 单张图片预测

```bash
python predict.py --image 你的图片.jpg
python predict.py --image 你的图片.jpg --json     # 输出 JSON 格式
```

### 5. 图形界面

```bash
python -m streamlit run app.py     # 浏览器打开 http://localhost:8501
```

### 6. 一键运行

```bash
bash run.sh        # 测试 -> 训练 -> 评估
bash run.sh ui     # 只启动界面
```

## 🧪 运行单元测试

```bash
python -m pytest tests/ -v
# 或
python -m unittest discover tests -v
```

测试共 **46 项**，覆盖数据、模型、推理与界面四条链路：

| 文件 | 覆盖内容 |
|---|---|
| `tests/test_data.py` | 变换流水线、5 种扰动函数、数据划分 |
| `tests/test_model.py` | 前向形状、各阶段特征图、保存/加载、异常加载 |
| `tests/test_inference.py` | 输入归一化、Top-K、异常处理、**静态图模式兼容性** |
| `tests/test_app_ui.py` | **Streamlit 界面端到端**：用官方 `AppTest` 真实执行 app.py 并连续上传 3 张图片，断言结果各不相同且正确 |

## 📓 实践 Notebook（推荐入门阅读）

📄 **[打开 cifar10_practice.ipynb](cifar10_practice.ipynb)** —— 与教材风格一致的**完整实践 Notebook**，
已保存真实运行输出（图形、指标、预测结果均内嵌），涵盖：
环境检测 → 数据集预览 → 数据增强可视化 → 模型搭建与前向验证 → 训练曲线解读
→ 测试集评估（分类报告 + 混淆矩阵）→ 5 种扰动鲁棒性测试 → 推理耗时 → 单图预测
→ 容错性验证 → 总结，共 11 节。

```bash
python -m jupyter notebook cifar10_practice.ipynb
```

> 第 5 节训练默认关闭（`RUN_TRAINING = False`，直接复用已训练的权重）；
> 改为 `True` 可从头完整训练（GPU 约 40 分钟）。

## 📊 汇报 PPT

📄 **[打开 课程设计汇报.pptx](课程设计汇报.pptx)** —— 14 页，从课程设计报告提炼，
含系统架构图、残差块示意图、训练曲线、逐类 F1 柱状图、混淆矩阵、鲁棒性对比图等图表；
每页附**讲者备注**（PPT 里按「演讲者视图」或备注栏查看），可直接用于答辩。

| 页 | 内容 | 页 | 内容 |
|---|---|---|---|
| 1 | 封面（小组成员署名） | 8 | 评估结果（逐类 F1） |
| 2 | 小组分工 | 9 | 混淆矩阵分析 |
| 3 | 背景与目标 | 10 | 多类型测试数据集（鲁棒性） |
| 4 | 系统架构 | 11 | 推理性能分析 |
| 5 | 数据集组织与预处理 | 12 | 用户交互界面与容错性 |
| 6 | 模型选型与搭建 | 13 | 错误排查记录（真实故障复盘） |
| 7 | 模型训练 | 14 | 总结与展望 |

> PPT 生成脚本 `outputs/make_ppt.js` 与布局自检脚本 `outputs/qa_ppt.py` 属开发期脚手架，
> 保留在作者源码目录中，**未包含在提交包内**（不影响本系统的运行与验收）。

## 📊 实测结果（RTX 5060 GPU）

| 项目 | 数值 |
|---|---|
| 测试集准确率（10000 张） | **92.07%**（宏平均 F1 0.9205） |
| 鲁棒性测试（每项 2000 张） | 裁剪平移 91.85% · 中心裁剪 90.30% · 旋转 85.25% · 高斯噪声 32.55% · 低分辨率 22.60% |
| 推理延迟（batch=1, GPU） | 3.30 ms/张（吞吐 303 img/s；batch=128 时 7073 img/s） |
| 训练时长 | 57 epoch 早停（最佳第 47 轮，验证 acc 92.88%），约 40 分钟 |
| 模型参数量 | 11.17 M（权重文件约 43 MB） |
| 开集识别（「其他」判定） | 真实测试图误判 2.4%（1000 张）；噪声/涂鸦类拦截 40.7%（双条件阈值 softmax<0.5 或 max-logit<4.0） |

> 详细报告见 `outputs/results/`（分类报告、混淆矩阵、对比表、耗时分析、开集标定 `ood_calibration.json`）。

## 📖 更多文档

| 文档 | 内容 |
|---|---|
| 📊 [课程设计汇报.pptx](课程设计汇报.pptx) | 答辩用 PPT（14 页，含图表与讲者备注） |
| 📓 [cifar10_practice.ipynb](cifar10_practice.ipynb) | 完整实践 Notebook（含真实运行输出） |
| 📄 [技术文档](docs/technical_doc.md) | 模块接口、数据流、模型结构、16 条错误排查记录、联调说明 |
| 📘 [用户使用说明书](docs/user_manual.md) | 面向非技术用户：安装、训练、识别、12 条常见问题 |
| 📑 [课程设计报告](report/course_design_report.md) | 完整课程报告（含实测数据与 19 项验收清单） |
| 📈 [评估结果目录](outputs/results) | 分类报告、混淆矩阵、鲁棒性对比表、训练历史、开集标定 |

## 📚 参考

邱锡鹏，《神经网络与深度学习》，5.5 节"实践：基于 ResNet18 网络完成图像分类任务（CIFAR-10）"，
课程教材配套代码 `Practice-in-Paddle-main/chap5卷积神经网络`。
