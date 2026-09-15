# -*- coding: utf-8 -*-
"""
tests/test_app_ui.py —— Streamlit 界面端到端测试
================================
使用 Streamlit 官方测试框架 ``AppTest`` **真实执行 app.py** 并模拟多次文件上传，
用于守住两个曾经真实发生过的界面故障：

  1. **重复上传显示第一次结果**：``@st.cache_data`` 会忽略名称以下划线开头的参数
     （当时把图片字节流命名为 ``_img_bytes`` 以跳过哈希），导致缓存键退化成
     ``(model_path, topk)`` 恒定不变，第一次上传的结果被后续所有上传复用。
  2. **静态图模式报 uint8**：飞桨处于静态图模式时，``ToTensor`` 会把 uint8 的
     numpy 数组交给 ``paddle.to_tensor``，报
     ``...in assign must be [...], but received uint8``。

运行：``python -m pytest tests/test_app_ui.py -v``
（未找到训练好的模型时自动跳过，不会导致失败）
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# 三张样例图及其真实类别（文件名前缀即真实类别）
SAMPLES = [
    ("0_cat.png", "cat"),
    ("1_ship.png", "ship"),
    ("20_horse.png", "horse"),
]

_HAS_MODEL = os.path.exists(config.MODEL_PATH)
_HAS_SAMPLES = all(os.path.exists(os.path.join(config.BASE_DIR, "test_imgs", f))
                   for f, _ in SAMPLES)


@unittest.skipUnless(_HAS_MODEL, "未找到训练好的模型，跳过界面端到端测试")
@unittest.skipUnless(_HAS_SAMPLES, "未找到 test_imgs 样例图片，跳过界面端到端测试")
class TestAppUI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:                       # pragma: no cover
            raise unittest.SkipTest("当前环境未安装 Streamlit 测试框架")
        cls.at = AppTest.from_file(
            os.path.join(config.BASE_DIR, "app.py"), default_timeout=300)
        cls.at.run()

    def test_app_starts_without_exception(self):
        """app 脚本应能正常渲染，并给出模型加载成功提示。"""
        self.assertFalse(self.at.exception,
                         "启动异常: %s" % [str(e.value)[:300] for e in self.at.exception])
        self.assertTrue(self.at.success, "未出现模型加载成功提示")

    def test_multiple_uploads_are_independent(self):
        """连续上传多张不同图片，每次结果都必须与该图片对应（核心回归点）。

        若缓存键忽略了图片内容，这里第 2、3 次会得到与第 1 次相同的类别而失败。
        """
        got = []
        for fname, truth in SAMPLES:
            with open(os.path.join(config.BASE_DIR, "test_imgs", fname), "rb") as f:
                data = f.read()
            self.at.file_uploader[0].set_value((fname, data, "image/png"))
            self.at.run()
            self.assertFalse(self.at.exception,
                             "第 %s 次上传异常: %s" %
                             (fname, [str(e.value)[:300] for e in self.at.exception]))
            self.assertTrue(self.at.metric, "未渲染预测结果")
            top1 = self.at.metric[0].value
            got.append((fname, truth, top1))

        # 断言 1：结果不能全部相同（这正是"只显示第一次结果"的故障特征）
        self.assertGreaterEqual(len(set(t for _, _, t in got)), 2,
                                "多次上传得到相同结果，疑似结果被缓存复用: %s" % got)
        # 断言 2：每张图都预测正确
        for fname, truth, top1 in got:
            self.assertIn(truth, top1,
                          "%s 预测错误，期望包含 %s，实际 %s" % (fname, truth, top1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
