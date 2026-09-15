# -*- coding: utf-8 -*-
"""
utils/io_utils.py —— 模型保存/加载 与 通用文件读写
================================
职责（对应课程要求"模型文件的保存与加载、数据文件读写、异常处理"）：
  1. 模型权重保存（paddle.save，.pdparams 格式，与教材一致）与容错加载；
  2. JSON / CSV / 纯文本 文件的统一读写封装（自动建目录、UTF-8、异常捕获）；
  3. 图像文件加载（支持 jpg/png/bmp/webp，带异常处理，供推理与 UI 调用）；
  4. 推理结果 Top-K 的格式化输出。

所有函数均做了防御式编程：输入不合法时抛出带中文提示的明确异常，
而不是让程序在深层调用栈中报出难以理解的错误。
"""

import io
import json
import os
import sys
import time

import numpy as np
import paddle
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

# 支持的图像格式
SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# 1. 模型保存与加载
# ---------------------------------------------------------------------------
def save_model(model, path=None, meta=None):
    """保存模型权重为 .pdparams 文件（与教材 5.5 节保存格式一致）。

    Args:
        model (paddle.nn.Layer): 待保存模型。
        path (str|None): 保存路径，默认 config.MODEL_PATH（best_model.pdparams）。
        meta (dict|None): 附加元信息（如 epoch、验证准确率），
            将以 <path>.meta.json 同名文件保存，便于追溯。

    Returns:
        str: 实际保存路径。
    """
    path = path or config.MODEL_PATH
    ensure_parent(path)  # 兼容不带目录的纯文件名（os.makedirs('') 会报错）
    try:
        state = model.state_dict()
        paddle.save(state, path)
    except Exception as e:
        raise RuntimeError("模型保存失败（路径: %s）: %s" % (path, e)) from e

    # 附加元信息一并落盘
    if meta is not None:
        try:
            meta_out = dict(meta)
            meta_out["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            meta_out["model_path"] = path
            write_json(meta_out, path + ".meta.json")
        except Exception as e:  # noqa: BLE001 元信息保存失败不影响权重本身
            print("[io_utils] 元信息保存失败: %s" % e)
    return path


def load_model(model, path=None, strict=True, device=None):
    """加载模型权重（容错版）。

    常见错误的中文排查提示：
      - 文件不存在  -> 请先运行 python train.py 训练并保存模型；
      - 形状不匹配  -> 通常是加载了其他结构的权重（如官方 224x224 版），
                       请确认模型结构与训练时一致；
      - 权重损坏    -> 文件大小为 0 或 paddle.load 报错时需重新训练。

    Args:
        model (paddle.nn.Layer): 待载入权重的模型（就地修改）。
        path (str|None): 权重路径，默认 config.MODEL_PATH。
        strict (bool): True 表示发现键不匹配时抛出异常；False 时打印警告并
                       加载可以匹配的部分（用于迁移实验）。
        device (str|None): 张量加载后迁移的目标设备（如 "gpu:0"）。

    Returns:
        (model, meta): 载入权重后的模型 与 元信息字典（可能为空）。
    """
    path = path or config.MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            "模型文件不存在: %s\n请先执行 `python train.py` 训练并保存模型，"
            "或通过 --model 参数指定正确的模型路径。" % path)
    if os.path.getsize(path) == 0:
        raise IOError("模型文件为空（0 字节），可能上次训练中断导致损坏: %s" % path)

    try:
        state = paddle.load(path)
    except Exception as e:
        raise RuntimeError(
            "模型权重解析失败（文件可能已损坏）: %s\n错误信息: %s\n"
            "建议删除该文件后重新训练。" % (path, e)) from e

    # 权重键与模型参数对齐检查
    model_keys = set(model.state_dict().keys())
    weight_keys = set(state.keys())
    missing = model_keys - weight_keys
    unexpected = weight_keys - model_keys
    if missing or unexpected:
        msg = ("权重与模型结构不匹配：缺失键 %d 个（如 %s），"
               "多余键 %d 个（如 %s）。") % (
            len(missing), sorted(missing)[:3], len(unexpected), sorted(unexpected)[:3])
        if strict:
            raise RuntimeError(msg + " 请确认加载的 .pdparams 与当前模型结构一致。")
        print("[io_utils] 警告: " + msg + " 将只加载可匹配的部分。")

    model.set_state_dict(state)

    if device is not None:
        try:
            model.to(device)
        except Exception as e:  # noqa: BLE001 设备迁移失败不阻断流程
            print("[io_utils] 警告: 模型迁移到 %s 失败: %s" % (device, e))

    # 读取随模型保存的元信息（可能不存在）
    meta = {}
    if os.path.exists(path + ".meta.json"):
        try:
            meta = read_json(path + ".meta.json")
        except Exception:  # noqa: BLE001
            meta = {}
    return model, meta


# ---------------------------------------------------------------------------
# 2. 通用文件读写（JSON / 文本 / CSV 行）
# ---------------------------------------------------------------------------
def ensure_parent(path):
    """确保文件的父目录存在。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_json(obj, path):
    """写入 JSON 文件（UTF-8、缩进、中文原样输出）。"""
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def read_json(path):
    """读取 JSON 文件；文件不存在或格式错误时抛出带提示的异常。"""
    if not os.path.exists(path):
        raise FileNotFoundError("JSON 文件不存在: %s" % path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise IOError("JSON 解析失败（文件可能损坏）: %s, %s" % (path, e)) from e


def write_text(text, path):
    """写入 UTF-8 文本文件。"""
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def read_text(path):
    """读取 UTF-8 文本文件。"""
    if not os.path.exists(path):
        raise FileNotFoundError("文件不存在: %s" % path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# 3. 图像文件加载（供 predict.py / app.py 使用）
# ---------------------------------------------------------------------------
def load_image(path):
    """从磁盘加载图像为 RGB PIL.Image，带完整的异常处理。

    Raises:
        FileNotFoundError: 路径不存在。
        ValueError: 不是受支持的图像格式。
        IOError: 文件损坏、解码失败。
    """
    if not os.path.exists(path):
        raise FileNotFoundError("图片不存在: %s" % path)
    if os.path.isdir(path):
        raise ValueError("输入是一个目录而非文件: %s" % path)
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTS:
        raise ValueError(
            "不支持的图片格式 '%s'，当前支持: %s" % (ext, ", ".join(SUPPORTED_IMAGE_EXTS)))
    try:
        img = Image.open(path)
        img.load()  # 立即解码，提前暴露损坏文件
        return img.convert("RGB")
    except Exception as e:
        raise IOError("图片解码失败（文件可能损坏或格式特殊）: %s, %s" % (path, e)) from e


def ensure_dynamic_mode(verbose=True):
    """确保飞桨运行在**动态图（eager）模式**下。

    背景：飞桨在静态图模式下，``paddle.to_tensor(自 uint8 的 numpy 数组)`` 会走
    ``assign()`` 分支并报
    ``The data type of 'input' in assign must be [...], but received uint8``。
    而本项目的 ``T.ToTensor`` / ``paddle.vision`` 相关调用在底层正是把
    ``np.asarray(PIL 图片)``（uint8）交给 ``to_tensor``。

    正常情况下飞桨默认就是动态图模式，但若运行宿主（某些框架、被切换过模式的
    进程）不处于动态图模式，就会出现"命令行能识别、别的入口全部报错"的怪现象。
    各入口脚本启动时调用本函数即可彻底消除该隐患。

    Returns:
        bool: 调用后是否处于动态图模式。
    """
    try:
        if not paddle.in_dynamic_mode():
            paddle.disable_static()   # 语义即"关闭静态图"= 启用动态图
            if verbose:
                print("[runtime] 检测到飞桨处于静态图模式，已切换为动态图模式")
        return bool(paddle.in_dynamic_mode())
    except Exception as e:  # noqa: BLE001 模式检查失败不应阻断主流程
        if verbose:
            print("[runtime] 动态图模式检查失败（不影响继续运行）: %s" % e)
        return False


def ensure_rgb_pil(image):
    """把任意常见输入统一转换为 RGB 模式的 PIL 图像（推理入口的防御层）。

    为什么需要它：飞桨的张量创建接口（``paddle.assign`` / ``to_tensor``）**不接受
    uint8 数组**，若把未经转换的原始像素数组直接送进模型，就会报
    ``The data type of 'input' in assign must be [...], but received uint8``。
    本函数保证进入变换流水线的永远是 RGB 的 PIL 图像，从源头消除该错误。

    支持输入：图片路径 / 图片字节流 / PIL 图像（任意 mode）/ numpy 数组
    （HWC、CHW、灰度 HW，uint8、浮点等各种 dtype）。

    Raises:
        IOError: 字节流无法解析为图片。
        ValueError: 数组形状或通道数不支持。
        TypeError: 输入类型完全不支持。
    """
    # --- 1) 图片路径 ---
    if isinstance(image, (str, os.PathLike)):
        return load_image(image)

    # --- 2) 图片字节流（Streamlit 上传、网络请求等场景）---
    if isinstance(image, (bytes, bytearray, memoryview)):
        try:
            im = Image.open(io.BytesIO(bytes(image)))
            im.load()                      # 立即解码，损坏文件在此暴露
            return _pil_to_rgb(im)
        except Exception as e:
            raise IOError("字节流不是可识别的图片: %s" % e) from e

    # --- 3) numpy 数组（cv2.imread、np.array(PIL) 等场景）---
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:                                   # 灰度 HW -> HWC
            arr = np.stack([arr] * 3, axis=-1)
        elif (arr.ndim == 3 and arr.shape[0] in (1, 3, 4)
              and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]):
            # 仅当第 0 维明显小于后两维时才判定为 CHW 布局，
            # 避免把 (H, W, C) 的小图（如 4x4x5）误判为 CHW
            arr = arr.transpose(1, 2, 0)                    # CHW -> HWC
        if arr.ndim != 3:
            raise ValueError("无法识别的数组形状 %s（期望 HWC / CHW / HW）" % (image.shape,))
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]                             # 丢弃 alpha 通道
        elif arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        elif arr.shape[2] != 3:
            raise ValueError("通道数必须为 1/3/4，实际为 %d" % arr.shape[2])
        if np.issubdtype(arr.dtype, np.floating):
            # 浮点数组：按 [0,1] 与 [0,255] 两种常见约定自动判断
            if arr.size and float(np.nanmax(arr)) <= 1.0 + 1e-6:
                arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    # --- 4) PIL 图像 ---
    if isinstance(image, Image.Image):
        return _pil_to_rgb(image)

    raise TypeError(
        "不支持的输入类型 %s。请传入图片路径、图片字节流、PIL 图像或 numpy 数组。"
        % type(image).__name__)


def _pil_to_rgb(im):
    """PIL 图像 -> RGB（处理高位深 I/I;16/F 与调色板/透明通道等各种 mode）。"""
    if im.mode == "RGB":
        return im
    if im.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        # 高位深图像直接 convert 会得到全黑/全白，这里先线性拉伸到 8bit
        arr = np.asarray(im, dtype=np.float32)
        rng = float(arr.max() - arr.min())
        if rng > 1e-6:
            arr = (arr - arr.min()) / rng * 255.0
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB")
    return im.convert("RGB")


def list_images(folder, recursive=True):
    """列出目录下所有支持的图像文件路径。"""
    if not os.path.isdir(folder):
        raise NotADirectoryError("目录不存在: %s" % folder)
    result = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for fn in files:
                if fn.lower().endswith(SUPPORTED_IMAGE_EXTS):
                    result.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(folder):
            p = os.path.join(folder, fn)
            if os.path.isfile(p) and fn.lower().endswith(SUPPORTED_IMAGE_EXTS):
                result.append(p)
    return sorted(result)


# ---------------------------------------------------------------------------
# 4. 推理结果格式化
# ---------------------------------------------------------------------------
def format_topk(predictions, class_names=None, class_names_cn=None, topk=3):
    """将 Top-K 预测结果格式化为可读文本（控制台输出与 UI 共用）。

    Args:
        predictions (list[(idx, prob)]): predict.predict_image 返回的 Top-K 结果。
            当首元素 idx 为 -1 时表示「其他」（开集识别：置信度低于阈值）。
        class_names (list): 英文类别名。
        class_names_cn (list): 中文类别名。

    Returns:
        str: 形如
            Top-3 预测结果:
              1. airplane(飞机)   置信度 92.35%
              ...
            若判定为「其他」，则首行追加阈值说明。
    """
    class_names = class_names or config.CIFAR10_CLASSES
    class_names_cn = class_names_cn or config.CIFAR10_CLASSES_CN
    lines = ["Top-%d 预测结果:" % topk]

    # 开集识别：首元素为 (-1, 最高置信度) 时给出阈值提示
    if predictions and predictions[0][0] == -1:
        lines.append("  ⚠ 置信度特征异常（最高 softmax %.2f%% / 阈值 %.0f%%，"
                     "或最大 logit 低于 %.1f），判定为「其他」——"
                     "该图片可能不属于已训练的 10 个类别"
                     % (predictions[0][1] * 100, config.CONFIDENCE_THRESHOLD * 100,
                        config.LOGIT_THRESHOLD))

    for rank, (idx, prob) in enumerate(predictions[:topk], 1):
        if idx == -1:
            en, cn = config.OTHER_CLASS_EN, config.OTHER_CLASS_CN
        else:
            en, cn = class_names[idx], class_names_cn[idx]
        lines.append("  %d. %-10s(%s)  置信度 %6.2f%%" % (rank, en, cn, prob * 100))
    return "\n".join(lines)


if __name__ == "__main__":
    # 自检：JSON 读写 + 图像异常处理 + 输入归一化
    from PIL import Image

    p = write_json({"hello": "你好", "acc": 0.9}, os.path.join(config.OUTPUT_DIR, "_test.json"))
    print("json 写入:", p, "读回:", read_json(p))
    os.remove(p)

    try:
        load_image("不存在.jpg")
    except FileNotFoundError as e:
        print("异常处理自检(1) ✔:", e)
    fake = os.path.join(config.OUTPUT_DIR, "_fake.jpg")
    open(fake, "wb").write(b"not an image")
    try:
        load_image(fake)
    except IOError as e:
        print("异常处理自检(2) ✔:", e)
    os.remove(fake)

    real = os.path.join(config.OUTPUT_DIR, "_real.png")
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(real)
    img = load_image(real)
    print("正常加载自检 ✔:", img.size, img.mode)

    # 输入归一化自检：uint8 数组、CHW、灰度、字节流、调色板
    print("--- ensure_rgb_pil 输入归一化自检 ---")
    raw = np.random.rand(40, 50, 3)
    checks = {
        "np uint8 HWC": (raw * 255).astype(np.uint8),
        "np float32 HWC": raw.astype(np.float32),
        "np uint8 CHW": (raw.transpose(2, 0, 1) * 255).astype(np.uint8),
        "np uint8 灰度 HW": (raw[:, :, 0] * 255).astype(np.uint8),
        "路径": real,
        "字节流": open(real, "rb").read(),
        "PIL RGBA": Image.open(real).convert("RGBA"),
        "PIL 调色板": Image.open(real).convert("P"),
        "PIL 16位": Image.open(real).convert("I;16"),
    }
    for name, obj in checks.items():
        out = ensure_rgb_pil(obj)
        assert out.mode == "RGB", name
        print("  %-16s -> %s %s ✔" % (name, out.mode, out.size))
    try:
        ensure_rgb_pil(12345)
    except TypeError as e:
        print("  异常处理自检(3) ✔:", e)
    os.remove(real)
