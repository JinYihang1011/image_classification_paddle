# -*- coding: utf-8 -*-
"""
tests/test_inference.py —— 推理链路单元测试
================================
测试内容：
  1. 推理预处理（任意尺寸 -> 32x32 标准化）；
  2. predict_image 返回 Top-K 结构与置信度排序正确性；
  3. 异常处理：文件不存在 / 损坏图片 / 不支持格式 均抛出预期异常；
  4. 保存随机初始化模型后走完整 CLI 推理函数链路（不依赖已训练权重）。
"""

import io
import os
import sys
import tempfile
import unittest

import numpy as np
import paddle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from data.preprocess import build_infer_transform  # noqa: E402


def _save_random_model(tmpdir):
    """构建随机初始化模型并保存，供推理测试使用。"""
    from models.resnet import resnet18
    from utils.io_utils import save_model
    paddle.seed(0)
    model = resnet18()
    path = os.path.join(tmpdir, "rand_model.pdparams")
    save_model(model, path)
    return path


class TestPreprocess(unittest.TestCase):

    def test_infer_transform_output(self):
        t = build_infer_transform()
        img = Image.fromarray((np.random.rand(50, 80, 3) * 255).astype(np.uint8))
        x = t(img)
        self.assertEqual(tuple(x.shape), (3, 32, 32))
        self.assertEqual(str(x.dtype), "paddle.float32")


class TestPredict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile
        from predict import build_predictor
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_path = _save_random_model(cls._tmp.name)
        cls.model, cls.meta = build_predictor(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_predict_topk_structure(self):
        """predict_image 应返回 k 个 (int索引, float概率)，且按置信度降序。"""
        from predict import predict_image
        img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
        preds = predict_image(self.model, img, topk=3)
        self.assertEqual(len(preds), 3)
        probs = [p for _, p in preds]
        self.assertEqual(probs, sorted(probs, reverse=True))
        for idx, prob in preds:
            self.assertIsInstance(idx, int)
            self.assertTrue(0 <= idx <= 9)
            self.assertTrue(0.0 <= prob <= 1.0)
        self.assertAlmostEqual(sum(probs[:3]) + 0, sum(probs[:3]), places=6)
        self.assertLessEqual(sum(probs), 1.0 + 1e-5)

    def test_predict_top1_sums_consistency(self):
        """topk=10 时概率之和应为 1。"""
        from predict import predict_image
        img = Image.fromarray((np.random.rand(16, 16, 3) * 255).astype(np.uint8))
        preds = predict_image(self.model, img, topk=10)
        self.assertAlmostEqual(sum(p for _, p in preds), 1.0, places=4)

    def test_predict_from_path(self):
        """predict_image 支持直接传图片路径。"""
        from predict import predict_image
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
            preds = predict_image(self.model, path, topk=3)
        finally:
            if os.path.exists(path):
                os.unlink(path)
        self.assertEqual(len(preds), 3)


class TestPredictErrors(unittest.TestCase):
    """推理异常处理测试（容错性验收）。"""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_path = _save_random_model(cls._tmp.name)
        from predict import build_predictor
        cls.model, _ = build_predictor(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_missing_file(self):
        from predict import predict_image
        with self.assertRaises(FileNotFoundError):
            predict_image(self.model, os.path.join(self._tmp.name, "不存在.jpg"))

    def test_corrupt_image(self):
        """损坏的图片文件应抛出 IOError 而非让程序崩溃。"""
        from predict import predict_image
        bad = os.path.join(self._tmp.name, "bad.jpg")
        with open(bad, "wb") as f:
            f.write(b"\x00\x01not-an-image")
        with self.assertRaises(IOError):
            predict_image(self.model, bad)

    def test_unsupported_format(self):
        """不支持的格式（如 .txt）应抛出 ValueError。"""
        from predict import predict_image
        txt = os.path.join(self._tmp.name, "a.txt")
        with open(txt, "w", encoding="utf-8") as f:
            f.write("hello")
        with self.assertRaises(ValueError):
            predict_image(self.model, txt)


class TestEnsureRgbPil(unittest.TestCase):
    """输入归一化防御层测试（ensure_rgb_pil）。

    背景：飞桨的 paddle.assign/to_tensor 不接受 uint8 数组，若原始像素数组
    未经转换直接送入模型，会报 "but received uint8"。本类确保所有输入类型
    都被安全地转成 RGB PIL 图像。
    """

    def setUp(self):
        self.raw = (np.random.rand(40, 50, 3) * 255).astype(np.uint8)
        self.path = os.path.join(tempfile.gettempdir(), "_ensure_test.png")
        Image.fromarray(self.raw).save(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_numpy_uint8_hwc(self):
        from utils.io_utils import ensure_rgb_pil
        out = ensure_rgb_pil(self.raw)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, (50, 40))

    def test_numpy_float_and_chw_and_gray(self):
        from utils.io_utils import ensure_rgb_pil
        self.assertEqual(ensure_rgb_pil(self.raw.astype(np.float32)).mode, "RGB")
        self.assertEqual(ensure_rgb_pil(self.raw.transpose(2, 0, 1)).mode, "RGB")
        self.assertEqual(ensure_rgb_pil(self.raw[:, :, 0]).mode, "RGB")

    def test_pil_various_modes(self):
        from utils.io_utils import ensure_rgb_pil
        for mode in ("RGBA", "P", "L", "1", "CMYK", "I;16"):
            out = ensure_rgb_pil(Image.fromarray(self.raw).convert(mode))
            self.assertEqual(out.mode, "RGB", "mode %s 转换失败" % mode)

    def test_bytes_and_path(self):
        from utils.io_utils import ensure_rgb_pil
        self.assertEqual(ensure_rgb_pil(self.path).mode, "RGB")
        with open(self.path, "rb") as f:
            self.assertEqual(ensure_rgb_pil(f.read()).mode, "RGB")

    def test_invalid_inputs_raise(self):
        from utils.io_utils import ensure_rgb_pil
        with self.assertRaises(TypeError):
            ensure_rgb_pil(12345)
        with self.assertRaises(IOError):
            ensure_rgb_pil(b"not-an-image")
        with self.assertRaises(ValueError):
            ensure_rgb_pil(np.zeros((4, 4, 5), dtype=np.uint8))  # 5 通道

    def test_full_inference_from_numpy(self):
        """端到端：直接把 uint8 数组交给 predict_image 也应正常工作。"""
        from predict import predict_image
        preds = predict_image(TestPredict.model, self.raw, topk=3)
        self.assertEqual(len(preds), 3)


class TestRuntimeResilience(unittest.TestCase):
    """静态图模式兼容性回归测试（针对"Web 界面任何图片都报 uint8"的故障）。

    故障链条：飞桨处于静态图模式 → ``paddle.to_tensor(np.asarray(PIL图))``
    拿到 uint8 → 走 ``assign()`` 分支被拒 → 报
    ``The data type of 'input' in assign must be [...], but received uint8``。
    命令行动态图模式正常，因此该故障只在特定宿主环境下暴露。
    """

    def tearDown(self):
        # 无论如何都恢复到动态图模式，避免影响其他测试
        import paddle
        paddle.disable_static()

    def test_ensure_dynamic_mode_recovers_from_static(self):
        """ensure_dynamic_mode 应能把静态图模式切回动态图模式。"""
        import paddle
        from utils.io_utils import ensure_dynamic_mode

        paddle.enable_static()
        self.assertFalse(paddle.in_dynamic_mode())
        self.assertTrue(ensure_dynamic_mode(verbose=False))
        self.assertTrue(paddle.in_dynamic_mode())

    def test_image_to_tensor_mode_independent(self):
        """自实现的 image_to_tensor 与运行模式无关，静态图模式下也应成功。"""
        import paddle
        from data.preprocess import image_to_tensor

        img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype(np.uint8))
        paddle.enable_static()
        x = image_to_tensor(img, size=32)
        self.assertEqual(tuple(x.shape), (3, 32, 32))
        self.assertEqual(str(x.dtype), "paddle.float32")

    def test_document_root_cause(self):
        """固化根因认知：paddle 的 ToTensor 在静态图模式下会拒绝 uint8。

        （若未来飞桨版本修掉了该行为，本测试自动跳过而非失败。）
        """
        import paddle
        from data.preprocess import build_infer_transform

        img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))
        paddle.enable_static()
        try:
            build_infer_transform()(img)
            self.skipTest("当前飞桨版本已兼容静态图模式下的 uint8 输入")
        except Exception as e:
            self.assertIn("uint8", str(e))


class TestOtherClass(unittest.TestCase):
    """开集识别（「其他」类别）测试。

    判定规则：最高 softmax < CONFIDENCE_THRESHOLD 或最大 logit < LOGIT_THRESHOLD
    时，predict_image 返回首元素 (-1, 最高置信度)。
    """

    @classmethod
    def setUpClass(cls):
        import tempfile
        from predict import build_predictor
        cls._tmp = tempfile.TemporaryDirectory()
        cls.model_path = _save_random_model(cls._tmp.name)
        cls.model, _ = build_predictor(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _img(self):
        return Image.fromarray((np.random.rand(32, 32, 3) * 255).astype(np.uint8))

    def test_rejected_when_threshold_high(self):
        """阈值取 1.0（softmax 恒小于 1.0）→ 必然判定为「其他」。"""
        from predict import predict_image
        preds = predict_image(self.model, self._img(), topk=3,
                              reject_other=True, threshold=1.0)
        self.assertEqual(preds[0][0], -1)
        self.assertTrue(0.0 <= preds[0][1] < 1.0)
        for idx, _ in preds[1:]:
            self.assertTrue(0 <= idx <= 9)   # 其余仍为参考 Top-K

    def test_not_rejected_when_disabled(self):
        """关闭开关（reject_other=False）→ 永远返回 0~9 的类别。"""
        from predict import predict_image
        preds = predict_image(self.model, self._img(), topk=3,
                              reject_other=False)
        for idx, _ in preds:
            self.assertTrue(0 <= idx <= 9)

    def test_format_topk_shows_other(self):
        """format_topk 对 -1 索引显示「其他」并给出阈值提示。"""
        from utils.io_utils import format_topk
        text = format_topk([(-1, 0.3125), (3, 0.2), (5, 0.1)])
        self.assertIn("其他", text)
        self.assertIn("⚠", text)
        self.assertIn("31.25%", text)

    def test_config_thresholds_sane(self):
        """配置阈值处于合理区间。"""
        import config
        self.assertTrue(0.3 < config.CONFIDENCE_THRESHOLD < 0.9)
        self.assertTrue(0.0 < config.LOGIT_THRESHOLD < 15.0)


class TestTopkFormat(unittest.TestCase):
    """format_topk 输出格式测试。"""

    def test_format_contains_confidence(self):
        from utils.io_utils import format_topk
        text = format_topk([(0, 0.9), (1, 0.08), (2, 0.02)])
        self.assertIn("Top-3", text)
        self.assertIn("airplane", text)
        self.assertIn("90.00%", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
