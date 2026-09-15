# -*- coding: utf-8 -*-
"""
data/preprocess.py —— 数据预处理与数据增强
================================
职责：
  1. 构建训练 / 评估 / 推理三种图像变换流水线（paddle.vision.transforms）；
  2. 提供鲁棒性测试所需的多种扰动函数：高斯噪声、旋转、裁剪、低分辨率，
     用于模拟真实应用场景中的"脏数据"，检验模型泛化能力。

变换流水线设计（与教材 5.5 节 Normalize 写法一致）：
  训练：随机裁剪(32x32, padding=4) -> 随机水平翻转 -> ToTensor -> Normalize
  评估/推理：ToTensor -> Normalize
其中 ToTensor 将 PIL 图像转为 [0,1] 的 CHW 张量；
Normalize 使用 CIFAR-10 训练集的均值/方差（见 config.NORM_MEAN/NORM_STD）。
"""

import io
import random

import numpy as np
import paddle
import paddle.vision.transforms as T
from PIL import Image

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# 1. 变换流水线
# ---------------------------------------------------------------------------
def build_transform(is_train=True):
    """构建图像变换流水线。

    Args:
        is_train (bool): True 返回训练增强流水线；False 返回评估流水线。

    Returns:
        paddle.vision.transforms.Compose: 可作用于 PIL 图像的变换对象。
    """
    if is_train:
        return T.Compose([
            # 随机裁剪：先四周补 0 再随机裁回 32x32，等效于小幅平移扰动
            T.RandomCrop(config.IMAGE_SIZE, padding=config.CROP_PADDING),
            # 随机水平翻转：以 50% 概率左右镜像（CIFAR-10 水平翻转不改变语义）
            T.RandomHorizontalFlip(),
            # PIL(HWC, uint8) -> CHW Tensor，像素缩放到 [0,1]
            T.ToTensor(),
            # 逐通道标准化：(x - mean) / std，与教材 5.5 节一致
            T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD,
                        data_format="CHW"),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD,
                        data_format="CHW"),
        ])


def build_infer_transform():
    """推理用变换：任意尺寸 PIL 图 -> 缩放到 32x32 -> 标准化张量。

    与训练/评估流水线保持相同的归一化参数，保证推理与训练分布一致。
    """
    return T.Compose([
        T.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=config.NORM_MEAN, std=config.NORM_STD,
                    data_format="CHW"),
    ])


def image_to_tensor(image, normalize=True, size=None):
    """PIL 图像 -> 标准化后的 CHW float32 张量（**不依赖** paddle 的 ToTensor）。

    为什么不直接用 ``T.ToTensor()``：
    飞桨的 ``ToTensor`` 对 PIL 图像内部执行的是
    ``paddle.to_tensor(np.asarray(pic))``，而 ``np.asarray(RGB 图片)`` 的类型是
    **uint8**。uint8 在动态图模式下能被接受，但在**静态图模式**下会走
    ``assign()`` 分支并被拒绝，报
    ``The data type of 'input' in assign must be [...], but received uint8``
    —— 这正是"命令行能识别、Web 界面任何图片都报错"的根因。

    本函数先把图像转成 float32 的 numpy 数组再建张量，**与飞桨运行模式无关**，
    两种模式下都安全，同时省去了 Compose 的多次类型分派，推理更快。

    Args:
        image (PIL.Image): RGB（或可直接转 RGB）的 PIL 图像。
        normalize (bool): 是否按 CIFAR-10 均值方差标准化。
        size (int|None): 目标边长；给定则先用 PIL 缩放到 size×size
            （默认 None，即调用方已保证尺寸）。

    Returns:
        paddle.Tensor: 形状 [3, H, W] 的 float32 张量。
    """
    if size:
        # 用 PIL 自身缩放，不经过飞桨，因此与运行模式无关
        image = image.resize((size, size), Image.BILINEAR)

    arr = np.asarray(image, dtype=np.float32)      # HWC, [0,255]
    if arr.ndim == 2:                              # 灰度图 -> 3 通道
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:                         # 丢弃 alpha 通道
        arr = arr[:, :, :3]
    arr = np.ascontiguousarray(arr / 255.0)        # 归一到 [0,1]，保证内存连续

    x = paddle.to_tensor(arr, dtype="float32")     # float32 在两种模式下都安全
    x = x.transpose([2, 0, 1])                     # HWC -> CHW

    if normalize:
        # 等价于 ToTensor + Normalize，结果与训练/评估流水线完全一致
        mean = paddle.to_tensor(config.NORM_MEAN, dtype="float32").reshape([3, 1, 1])
        std = paddle.to_tensor(config.NORM_STD, dtype="float32").reshape([3, 1, 1])
        x = (x - mean) / std
    return x


# ---------------------------------------------------------------------------
# 2. 鲁棒性测试扰动函数
#    输入/输出均为 PIL.Image（RGB, uint8），可无缝嵌入变换流水线。
# ---------------------------------------------------------------------------
def _to_pil(img):
    """统一转换为 RGB PIL 图像（兼容输入为 ndarray 的情形）。"""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    arr = np.asarray(img)
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def add_gaussian_noise(img, sigma=config.GAUSS_NOISE_SIGMA, seed=None):
    """叠加高斯噪声：模拟传感器噪声 / 低光照拍摄场景。

    Args:
        img (PIL.Image): 输入图像。
        sigma (float): 噪声标准差，基于 [0,1] 归一化像素范围。
        seed (int|None): 随机种子，便于复现测试结果。

    Returns:
        PIL.Image: 加噪后的图像（像素截断到 [0,255]）。
    """
    rng = np.random.RandomState(seed if seed is not None else random.randint(0, 2**31))
    arr = np.asarray(_to_pil(img)).astype(np.float32) / 255.0
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def rotate_image(img, degree=config.ROTATE_DEGREE, seed=None):
    """随机旋转图像：模拟拍摄角度倾斜的场景。

    旋转后空白区域填充黑色（与模型推理时缩放填充行为一致）。
    """
    rng = random.Random(seed) if seed is not None else random
    angle = rng.uniform(-degree, degree)
    # expand=False 保持 32x32 尺寸，超出部分为填充色
    return _to_pil(img).rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))


def random_crop_shift(img, max_shift=4, seed=None):
    """随机裁剪平移：模拟目标偏离画面中心的场景。

    先将图像四周各补 max_shift 像素的黑色边，再随机裁回原尺寸。
    """
    rng = random.Random(seed) if seed is not None else random
    w, h = _to_pil(img).size
    padded = T.Pad(max_shift, fill=0)(_to_pil(img))
    left = rng.randint(0, 2 * max_shift)
    top = rng.randint(0, 2 * max_shift)
    return padded.crop((left, top, left + w, top + h))


def apply_low_resolution(img, small_size=config.LOWRES_SIZE):
    """低分辨率退化：先下采样到 small_size 再上采样回原尺寸。

    模拟用户上传低清截图 / 远距离小目标的场景。
    """
    im = _to_pil(img)
    w, h = im.size
    small = im.resize((small_size, small_size), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def center_crop_shrink(img, keep_ratio=0.8):
    """中心裁剪：只保留中心 keep_ratio 比例的区域，模拟主体被部分裁切。"""
    im = _to_pil(img)
    w, h = im.size
    nw, nh = int(w * keep_ratio), int(h * keep_ratio)
    left, top = (w - nw) // 2, (h - nh) // 2
    return im.crop((left, top, left + nw, top + nh))


# 扰动名称 -> 函数映射，eval.py 据此批量构建鲁棒性测试集
PERTURBATIONS = {
    "高斯噪声(sigma=0.08)": add_gaussian_noise,
    "旋转(±20°)": rotate_image,
    "随机裁剪平移(±4px)": random_crop_shift,
    "低分辨率(12x12)": apply_low_resolution,
    "中心裁剪(80%)": center_crop_shrink,
}


# ---------------------------------------------------------------------------
# 3. 自检：直接运行本文件可可视化各扰动效果
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.metrics import setup_chinese_font
    setup_chinese_font()

    # 用一张随机图自检（无需下载数据集）
    demo = Image.fromarray(
        (np.random.rand(config.IMAGE_SIZE, config.IMAGE_SIZE, 3) * 255).astype(np.uint8)
    )
    def _apply(fn, name):
        # 各扰动函数签名不一（部分带 seed），这里统一只传必需参数
        try:
            return fn(demo, seed=0)
        except TypeError:
            return fn(demo)

    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    axes[0, 0].imshow(demo)
    axes[0, 0].set_title("原图")
    for ax, (name, fn) in zip(axes.flat[1:], PERTURBATIONS.items()):
        ax.imshow(_apply(fn, name))
        ax.set_title(name)
    for ax in axes.flat:
        ax.axis("off")
    out = os.path.join(config.RESULT_DIR, "perturb_demo.png")
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print("扰动效果图已保存至:", out)

    # 检查变换流水线输出形状与数值范围
    tr = build_transform(True)
    ev = build_transform(False)
    x_tr, x_ev = tr(demo), ev(demo)
    print("训练增强输出:", x_tr.shape, x_tr.dtype,
          "min=%.3f max=%.3f" % (float(x_tr.min()), float(x_tr.max())))
    print("评估变换输出:", x_ev.shape, x_ev.dtype)
