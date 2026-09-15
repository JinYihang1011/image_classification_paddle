# -*- coding: utf-8 -*-
"""
predict.py —— 单张图片推理脚本
================================
功能（对应课程要求"模型加载、单张图片推理、Top-3 类别与置信度"）：
  1. 加载训练好的最佳模型（outputs/best_model.pdparams）；
  2. 读取任意尺寸的 jpg/png/bmp/webp 图片，统一预处理后推理；
  3. 输出 Top-K 类别与置信度；支持 --json 输出结构化结果供程序调用。

用法示例：
  python predict.py --image test_imgs/car.jpg
  python predict.py --image test_imgs/car.jpg --model outputs/best_model.pdparams --topk 3
  python predict.py --image test_imgs/car.jpg --json     # 供其他程序解析
"""

import argparse
import json
import os
import sys
import time

import paddle
import paddle.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from models.resnet import resnet18  # noqa: E402
from data.preprocess import build_infer_transform, image_to_tensor  # noqa: E402
from utils.io_utils import (load_image, load_model, format_topk,  # noqa: E402
                            ensure_rgb_pil, ensure_dynamic_mode)
from utils.logger import get_logger  # noqa: E402


# ---------------------------------------------------------------------------
# 核心推理接口（app.py 也复用本函数，保证 CLI 与 UI 行为一致）
# ---------------------------------------------------------------------------
def predict_image(model, image, topk=3, reject_other=False, threshold=None):
    """对单张图片执行推理。

    Args:
        model (paddle.nn.Layer): 已加载权重的模型（eval 模式）。
        image (PIL.Image|str|bytes|np.ndarray): 图像对象、路径、字节流或像素数组。
        topk (int): 返回前 k 个预测。
        reject_other (bool): 开集识别开关。True 时若最高置信度低于阈值，
            返回列表的首元素为 (-1, 最高置信度)，表示判定为「其他」类别。
        threshold (float|None): 置信度阈值；None 时取 config.CONFIDENCE_THRESHOLD。

    Returns:
        list[(int, float)]: [(类别索引, 置信度0~1), ...] 按置信度降序；
        判定为「其他」时首元素索引为 -1，其余保留原始 Top-K 供参考。
    """
    # 统一归一化为 RGB PIL 图像：兼容 PIL 各种 mode、numpy 数组（含 uint8）、字节流。
    image = ensure_rgb_pil(image)

    # 注意：这里用自实现的 image_to_tensor，而不是 paddle 的 ToTensor。
    # 后者会把 uint8 的 numpy 数组交给 paddle.to_tensor，在静态图模式下会报
    # "...but received uint8"；自实现的版本与运行模式无关。
    x = image_to_tensor(image, size=config.IMAGE_SIZE).unsqueeze(0)   # [1,3,32,32]

    # 每次推理前确认处于动态图模式：静态图模式下模型前向会因张量类型不匹配失败。
    # 放在这里而不是只在入口调用，是为了防止宿主（Streamlit 等）换线程后模式失效。
    ensure_dynamic_mode(verbose=False)
    model.eval()
    with paddle.no_grad():
        logits = model(x)
        probs = F.softmax(logits, axis=1)[0]

    k = min(topk, probs.shape[0])
    top_probs, top_idx = paddle.topk(probs, k=k)
    preds = [(int(i), float(p)) for i, p in zip(top_idx.numpy(), top_probs.numpy())]

    # ---- 开集识别：置信度过低（softmax 或 logit 双条件任一）→ 判定「其他」（索引 -1）----
    if reject_other:
        thr_p = config.CONFIDENCE_THRESHOLD if threshold is None else threshold
        max_p = preds[0][1] if preds else 0.0
        max_l = float(logits[0].max().item()) if logits.ndim == 2 else float(logits.max().item())
        if max_p < thr_p or max_l < config.LOGIT_THRESHOLD:
            preds = [(-1, max_p)] + preds[1:]
    return preds


def build_predictor(model_path=None, device=None):
    """加载模型并返回 (model, meta)，供 predict.py / app.py 共用。

    会先确保飞桨处于动态图模式（静态图模式下预处理会因 uint8 报错）。

    Raises:
        FileNotFoundError / RuntimeError: 由 load_model 抛出，
            均带中文排查提示（模型不存在 / 结构不匹配 / 文件损坏）。
    """
    model_path = model_path or config.MODEL_PATH
    ensure_dynamic_mode()
    if device is None:
        device = "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
    try:
        paddle.set_device(device)
    except Exception:
        paddle.set_device("cpu")
    model = resnet18()
    model, meta = load_model(model, model_path, device=device)
    model.eval()
    return model, meta


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 单张图片推理")
    parser.add_argument("--image", type=str, required=True,
                        help="待预测图片路径（jpg/png/bmp/webp）")
    parser.add_argument("--model", type=str, default=config.MODEL_PATH,
                        help="模型权重路径（默认 outputs/best_model.pdparams）")
    parser.add_argument("--topk", type=int, default=3, help="返回前 k 个预测")
    parser.add_argument("--threshold", type=float, default=None,
                        help="「其他」判定阈值（默认取 config.CONFIDENCE_THRESHOLD=0.5）："
                             "最高置信度低于该值时判定为「其他」")
    parser.add_argument("--no-other", action="store_true",
                        help="关闭「其他」判定，始终输出 10 类中的 Top-K")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 格式输出结果（便于程序调用）")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = get_logger("predict")

    # 逐层捕获异常，给出明确中文提示（容错性设计）
    try:
        model, meta = build_predictor(args.model)
    except (FileNotFoundError, RuntimeError, IOError) as e:
        logger.error("模型加载失败: %s", e)
        sys.exit(1)

    try:
        t0 = time.time()
        predictions = predict_image(model, args.image, topk=args.topk,
                                    reject_other=not args.no_other,
                                    threshold=args.threshold)
        elapsed = (time.time() - t0) * 1000
    except (FileNotFoundError, ValueError, IOError) as e:
        logger.error("图片读取/预处理失败: %s", e)
        sys.exit(2)
    except Exception as e:  # noqa: BLE001 推理阶段未知异常兜底
        logger.error("推理过程出现未知错误: %s", e)
        sys.exit(3)

    is_other = bool(predictions) and predictions[0][0] == -1

    if args.json:
        print(json.dumps({
            "image": os.path.abspath(args.image),
            "is_other": is_other,
            "confidence_threshold": (args.threshold if args.threshold is not None
                                     else config.CONFIDENCE_THRESHOLD),
            "predictions": [
                {"index": idx,
                 "class": config.OTHER_CLASS_EN if idx == -1 else config.CIFAR10_CLASSES[idx],
                 "class_cn": config.OTHER_CLASS_CN if idx == -1 else config.CIFAR10_CLASSES_CN[idx],
                 "confidence": round(prob, 4)}
                for idx, prob in predictions],
            "inference_ms": round(elapsed, 2),
            "model": os.path.abspath(args.model),
        }, ensure_ascii=False, indent=2))
    else:
        print("=" * 50)
        print("图片:", args.image)
        if meta.get("val_acc"):
            print("模型: %s（训练时验证准确率 %.2f%%）" % (args.model, meta["val_acc"] * 100))
        print(format_topk(predictions, topk=args.topk))
        print("推理耗时: %.1f ms" % elapsed)
        print("=" * 50)


if __name__ == "__main__":
    main()
