# -*- coding: utf-8 -*-
"""
tests/test_data.py —— 数据模块单元测试（pytest / unittest 兼容写法）
================================
测试内容：
  1. 预处理变换流水线的输出形状、维度与数值范围；
  2. 各扰动函数（高斯噪声/旋转/裁剪/低分辨率）不改变尺寸、结果可复现；
  3. 数据集划分的确定性与标签范围。
说明：为避免测试依赖网络下载，数据集相关测试在本地无数据文件时自动跳过。
"""

import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from data.preprocess import (build_transform, build_infer_transform,  # noqa: E402
                             add_gaussian_noise, rotate_image,
                             random_crop_shift, apply_low_resolution,
                             center_crop_shrink)


def _fake_image(size=32):
    """生成一张随机测试图（不依赖真实数据集）。"""
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


class TestTransforms(unittest.TestCase):
    """预处理变换流水线测试。"""

    def test_train_transform_shape(self):
        """训练增强输出应为 [3,32,32] 标准化张量。"""
        x = build_transform(is_train=True)(_fake_image())
        self.assertEqual(tuple(x.shape), (3, config.IMAGE_SIZE, config.IMAGE_SIZE))

    def test_eval_transform_shape(self):
        """评估变换输出应为 [3,32,32] 标准化张量。"""
        x = build_transform(is_train=False)(_fake_image())
        self.assertEqual(tuple(x.shape), (3, config.IMAGE_SIZE, config.IMAGE_SIZE))

    def test_normalize_range(self):
        """标准化后数值应落在 [-5, 5] 区间（±4σ 以内即为合理）。"""
        x = build_transform(is_train=False)(_fake_image())
        self.assertGreaterEqual(float(x.min()), -5.0)
        self.assertLessEqual(float(x.max()), 5.0)

    def test_infer_transform_any_size(self):
        """推理变换应能处理任意尺寸输入（如 100x77）并输出 32x32。"""
        x = build_infer_transform()(_fake_image(100).resize((100, 77)))
        self.assertEqual(tuple(x.shape), (3, config.IMAGE_SIZE, config.IMAGE_SIZE))

    def test_deterministic_eval(self):
        """评估变换应是确定性的（同一输入两次输出一致）。"""
        img = _fake_image()
        f = build_transform(is_train=False)
        x1, x2 = f(img), f(img)
        self.assertTrue(np.allclose(x1.numpy(), x2.numpy()))


class TestPerturbations(unittest.TestCase):
    """鲁棒性扰动函数测试。"""

    def test_gaussian_noise_keeps_size(self):
        img = _fake_image()
        out = add_gaussian_noise(img, sigma=0.05, seed=1)
        self.assertEqual(out.size, img.size)

    def test_gaussian_noise_changes_pixels(self):
        img = _fake_image()
        out = np.asarray(add_gaussian_noise(img, sigma=0.2, seed=1), dtype=np.int16)
        diff = np.abs(out - np.asarray(img, dtype=np.int16)).mean()
        self.assertGreater(diff, 1.0)  # 强噪声应造成可观测的变化

    def test_gaussian_noise_reproducible(self):
        img = _fake_image()
        a = np.asarray(add_gaussian_noise(img, seed=42))
        b = np.asarray(add_gaussian_noise(img, seed=42))
        self.assertTrue(np.array_equal(a, b))  # 固定种子应可复现

    def test_rotate_keeps_size(self):
        img = _fake_image()
        self.assertEqual(rotate_image(img, degree=15, seed=0).size, img.size)

    def test_crop_shift_keeps_size(self):
        img = _fake_image()
        self.assertEqual(random_crop_shift(img, max_shift=4, seed=0).size, img.size)

    def test_low_resolution_same_size(self):
        img = _fake_image()
        out = apply_low_resolution(img, small_size=8)
        self.assertEqual(out.size, img.size)

    def test_center_crop_ratio(self):
        img = _fake_image()
        out = center_crop_shrink(img, keep_ratio=0.5)
        self.assertEqual(out.size, (16, 16))


class TestDataset(unittest.TestCase):
    """数据集测试：本地无数据文件时自动跳过（避免 CI 环境联网）。"""

    def setUp(self):
        self.has_data = os.path.exists(config.DATA_FILE)

    def test_split_deterministic(self):
        """train/val 划分应与随机种子绑定（两次构建索引一致）。"""
        if not self.has_data:
            self.skipTest("本地无 CIFAR-10 数据文件，跳过")
        from data.dataset import CIFAR10Dataset
        ds1 = CIFAR10Dataset(mode="train", download=False)
        ds2 = CIFAR10Dataset(mode="train", download=False)
        self.assertEqual(ds1.indices[:100], ds2.indices[:100])

    def test_split_no_overlap(self):
        """训练集与验证集样本不应重叠。"""
        if not self.has_data:
            self.skipTest("本地无 CIFAR-10 数据文件，跳过")
        from data.dataset import CIFAR10Dataset
        tr = set(map(int, CIFAR10Dataset(mode="train", download=False).indices))
        va = set(map(int, CIFAR10Dataset(mode="val", download=False).indices))
        self.assertFalse(tr & va)

    def test_sample_range(self):
        """样本张量形状与标签取值范围检查。"""
        if not self.has_data:
            self.skipTest("本地无 CIFAR-10 数据文件，跳过")
        from data.dataset import CIFAR10Dataset
        ds = CIFAR10Dataset(mode="val", download=False)
        img, label = ds[0]
        self.assertEqual(tuple(img.shape), (3, 32, 32))
        self.assertTrue(0 <= label <= 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
