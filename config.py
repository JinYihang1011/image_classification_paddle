# -*- coding: utf-8 -*-
"""
config.py —— 全局配置文件
================================
集中管理本项目的所有可调参数，包括：
  1. 路径配置（数据集、输出、日志、模型保存位置）；
  2. CIFAR-10 数据集元信息（类别名称、归一化均值/方差、下载地址与 MD5）；
  3. 训练超参数（批次大小、学习率、优化器、学习率衰减、早停等）；
  4. 运行环境配置（设备选择、随机种子、DataLoader 工作进程数）。

参考教材：邱锡鹏《神经网络与深度学习》5.5 节
"实践：基于 ResNet18 网络完成图像分类任务（CIFAR-10）"
（东南大学人工智能实践课程教材配套代码 Practice-in-Paddle-main/chap5卷积神经网络）。
"""

import os

# ---------------------------------------------------------------------------
# 1. 路径配置
# ---------------------------------------------------------------------------
# 项目根目录（config.py 所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CIFAR-10 数据目录：tar 包与解压缓存均放在项目 data/ 下，便于离线管理与迁移
DATA_DIR = os.path.join(BASE_DIR, "data", "cifar10")
# CIFAR-10 原始 tar 包完整路径（paddle.vision.datasets.Cifar10 通过它读取数据）
DATA_FILE = os.path.join(DATA_DIR, "cifar-10-python.tar.gz")

# 输出目录：模型、日志、评估结果均保存到 outputs/ 下
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")

# 最佳模型保存路径（训练过程中验证集准确率最高的模型）
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pdparams")
# 最后一个 epoch 的模型保存路径（用于断点续训 / 对比实验）
LAST_MODEL_PATH = os.path.join(OUTPUT_DIR, "last_model.pdparams")
# 训练历史（JSON 格式，便于绘图与分析）
HISTORY_PATH = os.path.join(RESULT_DIR, "train_history.json")

# ---------------------------------------------------------------------------
# 2. CIFAR-10 数据集元信息
# ---------------------------------------------------------------------------
# 官方下载地址（BCC 镜像，与飞桨官方一致）与文件 MD5 校验码
DATA_URL = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"
DATA_MD5 = "c58f30108f718f92721af3b95e74349a"

# CIFAR-10 的 10 个类别（英文原名 + 中文对照，UI 中均会展示）
CIFAR10_CLASSES = [
    "airplane",    # 飞机
    "automobile",  # 汽车
    "bird",        # 鸟
    "cat",         # 猫
    "deer",        # 鹿
    "dog",         # 狗
    "frog",        # 青蛙
    "horse",       # 马
    "ship",        # 船
    "truck",       # 卡车
]
CIFAR10_CLASSES_CN = [
    "飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车",
]
NUM_CLASSES = len(CIFAR10_CLASSES)

# 图像尺寸（CIFAR-10 原始分辨率 32x32，3 通道彩色图）
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3

# 归一化均值与标准差（CIFAR-10 训练集统计值，与教材 5.5 节保持一致）
NORM_MEAN = [0.4914, 0.4822, 0.4465]
NORM_STD = [0.2023, 0.1994, 0.2010]

# ---------------------------------------------------------------------------
# 3. 数据集划分与预处理超参数
# ---------------------------------------------------------------------------
# 从 50000 张训练图中划出的验证集比例（教材做法是用 batch5 作验证集；
# 本项目改为按固定随机种子分层抽样，保证 10 类样本分布均衡）
VAL_RATIO = 0.1
# 数据集划分与权重初始化使用的随机种子（保证实验可复现）
SEED = 42

# 数据增强参数
CROP_PADDING = 4          # 随机裁剪前四周填充的像素数（32+4*2=40 再裁回 32）

# ---------------------------------------------------------------------------
# 4. 训练超参数（可被 train.py 命令行参数覆盖）
# ---------------------------------------------------------------------------
BATCH_SIZE = 128          # 批次大小
NUM_EPOCHS = 60           # 最大训练轮数
LEARNING_RATE = 0.001     # 初始学习率
WEIGHT_DECAY = 5e-4       # L2 正则化系数（权重衰减）
MOMENTUM = 0.9            # Momentum 优化器动量项（选用 momentum 优化器时生效）
OPTIMIZER = "adam"        # 优化器类型："adam" 或 "momentum"（均可通过命令行切换）
LR_SCHEDULER = "step"     # 学习率衰减策略："step"（阶梯衰减）或 "cosine"（余弦退火）
LR_STEP_SIZE = 20         # step 策略：每多少个 epoch 衰减一次
LR_GAMMA = 0.1            # step 策略：衰减系数（lr *= gamma）

# 早停（Early Stopping）参数：验证集准确率连续 PATIENCE 个 epoch 未提升则提前停止
EARLY_STOP_PATIENCE = 10

# 日志打印间隔（单位：训练 step 数）
LOG_INTERVAL = 100

# ---------------------------------------------------------------------------
# 5. 运行环境配置
# ---------------------------------------------------------------------------
# DataLoader 工作进程数：Windows 下多进程 DataLoader 兼容性差，默认置 0；
# Linux 下可设为 4 加速数据读取
NUM_WORKERS = 0 if os.name == "nt" else 4

# 评估 / 鲁棒性测试使用的测试样本数（为控制耗时，鲁棒性测试默认只取子集）
ROBUST_TEST_NUM = 2000
# 鲁棒性测试各扰动的强度参数
GAUSS_NOISE_SIGMA = 0.08      # 高斯噪声标准差（相对 [0,1] 像素范围）
ROTATE_DEGREE = 20            # 旋转角度（度）
LOWRES_SIZE = 12              # 低分辨率测试：先缩到 12x12 再放大回 32x32

# ---------------------------------------------------------------------------
# 6. 开集识别（「其他」类别判定）
# ---------------------------------------------------------------------------
# 模型本质是闭集分类器：无论输入什么图片，都会在 10 个类别里"硬猜"一个。
# 对不属于这 10 类的图片（人物、风景、噪声等），猜出来的概率没有意义。
#
# 判定规则（两条满足其一即判为「其他」，阈值依据实测标定，见
# outputs/results/ood_calibration.json）：
#   1) 最高 softmax 概率 < CONFIDENCE_THRESHOLD（默认 0.5）
#      —— 标定：1000 张测试图仅 1.2% 低于此值；
#   2) 最大 logit < LOGIT_THRESHOLD（默认 4.0）
#      —— softmax 对噪声等输入会"过度自信"（实测噪声图中位数高达 0.75），
#         而 logit 分值区分度更好（分布内中位数 10.1 vs 噪声类 3.9），
#         与 Hendrycks 等提出的 Energy/MaxLogit 开集识别基线一致。
# 联合标定结果：真实测试图约 3.6% 会被误判为「其他」（1000 张测试集实测：
# softmax 条件单独命中 1.2%、logit 条件单独命中 3.2%，并集 3.6%）；
# 噪声/涂鸦类图片约有一半以上被拦截（具体数值随分布外样本的构造方式波动）。
CONFIDENCE_THRESHOLD = 0.5
LOGIT_THRESHOLD = 4.0
OTHER_CLASS_EN = "other"   # 「其他」类别的英文显示名（类别索引约定为 -1）
OTHER_CLASS_CN = "其他"    # 「其他」类别的中文显示名


def ensure_dirs():
    """确保所有输出目录存在。程序入口处调用一次即可。"""
    for d in (DATA_DIR, OUTPUT_DIR, LOG_DIR, RESULT_DIR):
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    # 直接运行本文件可快速检查配置与目录
    ensure_dirs()
    print("=" * 50)
    print("项目根目录      :", BASE_DIR)
    print("数据文件路径    :", DATA_FILE)
    print("最佳模型路径    :", MODEL_PATH)
    print("类别数          :", NUM_CLASSES)
    print("批次大小        :", BATCH_SIZE)
    print("DataLoader 进程 :", NUM_WORKERS)
    print("目录已就绪。")
