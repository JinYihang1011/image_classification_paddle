# -*- coding: utf-8 -*-
"""
tests/test_model.py —— 模型模块单元测试
================================
测试内容：
  1. ResNet18 前向传播输出形状 [N, 10]；
  2. 各残差阶段特征图尺寸符合 32x32 适配设计；
  3. 参数量在合理范围内（CIFAR 版 ResNet18 约 11M）；
  4. 反向传播可计算（loss.backward 正常）；
  5. 模型保存 / 加载往返一致（io_utils 集成测试）。
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from models.resnet import resnet18, count_parameters  # noqa: E402


class TestResNet18(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        paddle.seed(config.SEED)
        cls.model = resnet18()
        cls.model.eval()

    def _input(self, n=2):
        return paddle.randn([n, config.IMAGE_CHANNELS,
                             config.IMAGE_SIZE, config.IMAGE_SIZE])

    def test_forward_shape(self):
        """输出应为 [N, 10]。"""
        y = self.model(self._input(4))
        self.assertEqual(tuple(y.shape), (4, config.NUM_CLASSES))

    def test_stage_shapes(self):
        """各残差阶段空间尺寸: 32 -> 32 -> 16 -> 8 -> 4。"""
        x = self._input(1)
        expected = [(1, 64, 32, 32), (1, 64, 32, 32), (1, 128, 16, 16),
                    (1, 256, 8, 8), (1, 512, 4, 4)]
        stages = [self.model.conv1, self.model.layer1, self.model.layer2,
                  self.model.layer3, self.model.layer4]
        for layer, exp in zip(stages, expected):
            x = layer(x)
            self.assertEqual(tuple(x.shape), exp, "阶段输出尺寸不符: %s" % str(layer))

    def test_param_count_reasonable(self):
        """CIFAR 版 ResNet18 参数量应在 10M~13M 之间。"""
        n = count_parameters(self.model)
        self.assertTrue(10e6 < n < 13e6, "参数量异常: %d" % n)

    def test_backward(self):
        """损失可反向传播，梯度存在且非全零。"""
        model = resnet18()
        model.train()
        x, y = self._input(2), paddle.randint(0, 10, [2])
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        grad = model.conv1.weight.grad
        if callable(grad):  # 兼容 paddle 2.x（方法）与 3.x（属性）
            grad = grad()
        self.assertIsNotNone(grad)
        self.assertGreater(float(paddle.abs(grad).sum()), 0)

    def test_softmax_sums_to_one(self):
        """softmax 概率之和应为 1。"""
        probs = F.softmax(self.model(self._input(3)), axis=1)
        sums = probs.numpy().sum(axis=1)
        self.assertTrue(np.allclose(sums, 1.0, atol=1e-5))


class TestModelSaveLoad(unittest.TestCase):
    """模型保存/加载集成测试（utils.io_utils）。"""

    def test_save_load_roundtrip(self):
        """保存后的权重加载回同一结构模型，输出应完全一致。"""
        from utils.io_utils import save_model, load_model
        paddle.seed(0)
        model = resnet18()
        x = paddle.randn([2, 3, 32, 32])
        model.eval()
        with paddle.no_grad():
            y1 = model(x)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test_model.pdparams")
            save_model(model, path, meta={"epoch": 1, "val_acc": 0.5})
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(path + ".meta.json"))  # 元信息同步保存

            model2 = resnet18()
            model2, meta = load_model(model2, path)
            model2.eval()
            self.assertEqual(meta.get("epoch"), 1)
            with paddle.no_grad():
                y2 = model2(x)
            self.assertTrue(np.allclose(y1.numpy(), y2.numpy(), atol=1e-6))

    def test_load_missing_model_raises(self):
        """加载不存在的模型应抛出 FileNotFoundError（带中文提示）。"""
        from utils.io_utils import load_model
        model = resnet18()
        with self.assertRaises(FileNotFoundError):
            load_model(model, os.path.join(tempfile.gettempdir(), "no_such.pdparams"))

    def test_load_shape_mismatch_raises(self):
        """加载结构与当前模型不符的权重应报结构不匹配错误。"""
        from utils.io_utils import save_model, load_model
        # 用一个全连接网络伪造"错误结构"的权重
        wrong = paddle.nn.Sequential(
            paddle.nn.Flatten(), paddle.nn.Linear(3072, 10))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wrong.pdparams")
            save_model(wrong, path)
            with self.assertRaises(RuntimeError):
                load_model(resnet18(), path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
