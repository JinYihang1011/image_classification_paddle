# -*- coding: utf-8 -*-
"""
eval.py —— 模型评估与鲁棒性测试
================================
功能（对应课程要求"多种类型测试数据集、运行效果与性能分析"）：
  1. 在 CIFAR-10 官方测试集（10000 张）上评估：
     准确率 / 精确率 / 召回率 / F1 / 混淆矩阵 / 分类报告；
  2. 构建 5 种鲁棒性测试集（高斯噪声、旋转、裁剪平移、低分辨率、中心裁剪），
     对比模型在各扰动下的准确率，输出对比表格；
  3. 推理耗时分析（单张延迟 + 吞吐量）；
  4. 全部结果保存到 outputs/results/ 供课程报告引用。

用法示例：
  python eval.py                         # 完整评估（测试集 + 鲁棒性 + 耗时）
  python eval.py --robust-num 1000       # 每种鲁棒性测试取 1000 张（加速）
  python eval.py --skip-robust           # 只评估标准测试集
"""

import argparse
import os
import sys
import time

import numpy as np
import paddle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from data.dataset import CIFAR10Dataset  # noqa: E402
from data.preprocess import (build_transform, PERTURBATIONS)  # noqa: E402
from models.resnet import resnet18  # noqa: E402
from utils.io_utils import load_model, write_json, write_text  # noqa: E402
from utils.io_utils import ensure_dynamic_mode  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.metrics import (accuracy_from_preds, classification_report_dict,  # noqa: E402
                           format_classification_report, confusion_matrix,
                           save_confusion_matrix, save_comparison_table)


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 模型评估与鲁棒性测试")
    parser.add_argument("--model", type=str, default=config.MODEL_PATH,
                        help="模型权重路径")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--robust-num", type=int, default=config.ROBUST_TEST_NUM,
                        help="每种鲁棒性测试使用的样本数（默认 %d，0 表示全部）"
                             % config.ROBUST_TEST_NUM)
    parser.add_argument("--skip-robust", action="store_true",
                        help="跳过鲁棒性测试")
    parser.add_argument("--skip-timing", action="store_true",
                        help="跳过推理耗时分析")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 1. 标准测试集评估
# ---------------------------------------------------------------------------
@paddle.no_grad()
def evaluate_standard(model, batch_size=128, logger=None):
    """在官方测试集上评估，返回 (accuracy, y_true, y_pred, avg_loss)。"""
    dataset = CIFAR10Dataset(mode="test", transform=build_transform(False))
    model.eval()

    y_true, y_pred, losses = [], [], []
    n_total = len(dataset)
    t0 = time.time()
    batch = max(batch_size, 1)
    for start in range(0, n_total, batch):
        imgs, labels = [], []
        for i in range(start, min(start + batch, n_total)):
            x, y = dataset[i]
            imgs.append(x)
            labels.append(y)
        x = paddle.stack(imgs)
        y = paddle.to_tensor(labels, dtype="int64")

        logits = model(x)
        losses.append(float(paddle.nn.functional.cross_entropy(logits, y).item()))
        y_true.extend(labels)
        y_pred.extend(paddle.argmax(logits, axis=1).numpy().tolist())

        done = min(start + batch, n_total)
        if done % 2000 < batch and logger:
            logger.info("测试进度: %d/%d (%.1f%%)",
                        done, n_total, done * 100.0 / n_total)

    elapsed = time.time() - t0
    acc = accuracy_from_preds(y_true, y_pred)
    if logger:
        logger.info("标准测试集评估完成: 准确率 %.2f%% (%d/%d), 平均损失 %.4f, 耗时 %.1fs",
                    acc * 100, int(acc * n_total), n_total,
                    float(np.mean(losses)), elapsed)
    return acc, y_true, y_pred, float(np.mean(losses))


# ---------------------------------------------------------------------------
# 2. 鲁棒性测试集评估（高斯噪声/旋转/裁剪/低分辨率/中心裁剪）
# ---------------------------------------------------------------------------
@paddle.no_grad()
def evaluate_perturbed(model, perturb_fn, num_samples=2000, batch_size=128,
                       seed=0, logger=None):
    """在扰动测试集上评估。

    做法：取测试集前 num_samples 张图，逐一施加扰动后再走标准预处理。
    固定 seed 保证各扰动测试集使用同一批样本、相同扰动参数（公平对比）。
    """
    dataset = CIFAR10Dataset(mode="test", transform=build_transform(False))
    num_samples = min(num_samples, len(dataset))
    model.eval()

    y_true, y_pred = [], []
    batch_imgs, batch_labels = [], []
    for i in range(num_samples):
        img, label = dataset._base[dataset.indices[i]]  # 取原始 PIL 图
        try:
            img = perturb_fn(img, seed=seed + i)        # 带种子扰动（可复现）
        except TypeError:
            img = perturb_fn(img)                       # 无种子参数的扰动函数
        x = build_transform(False)(img)
        batch_imgs.append(x)
        batch_labels.append(label)
        if len(batch_imgs) == batch_size or i == num_samples - 1:
            logits = model(paddle.stack(batch_imgs))
            y_pred.extend(paddle.argmax(logits, axis=1).numpy().tolist())
            y_true.extend(batch_labels)
            batch_imgs, batch_labels = [], []

    acc = accuracy_from_preds(y_true, y_pred)
    if logger:
        logger.info("  准确率 %.2f%% (%d 样本)", acc * 100, num_samples)
    return acc, y_true, y_pred


# ---------------------------------------------------------------------------
# 3. 推理耗时分析
# ---------------------------------------------------------------------------
@paddle.no_grad()
def benchmark_inference(model, warmup=20, repeat=100, batch_size=1):
    """测量单张图片推理延迟与吞吐量。

    先预热 warmup 次（排除首次 CUDA 初始化等一次性开销），再取 repeat 次平均。
    """
    model.eval()
    x = paddle.randn([batch_size, config.IMAGE_CHANNELS,
                      config.IMAGE_SIZE, config.IMAGE_SIZE])
    for _ in range(warmup):
        model(x)
    if paddle.device.is_compiled_with_cuda():
        paddle.device.synchronize()

    t0 = time.time()
    for _ in range(repeat):
        model(x)
    if paddle.device.is_compiled_with_cuda():
        paddle.device.synchronize()
    elapsed = (time.time() - t0) / repeat

    return {
        "batch_size": batch_size,
        "avg_latency_ms": elapsed * 1000.0,
        "throughput_img_per_s": batch_size / elapsed,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    config.ensure_dirs()
    ensure_dynamic_mode()   # 同上：保证预处理在动态图模式下运行
    logger = get_logger("eval")
    logger.info("=" * 60)

    # ---- 加载模型 ----
    try:
        paddle.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")
        model = resnet18()
        model, meta = load_model(model, args.model)
        model.eval()
    except (FileNotFoundError, RuntimeError, IOError) as e:
        logger.error("模型加载失败: %s", e)
        sys.exit(1)
    logger.info("模型已加载: %s", args.model)

    results = {"model": os.path.abspath(args.model),
               "train_meta": meta or {}}

    # ---- 1. 标准测试集 ----
    logger.info("[1/3] 标准测试集评估 ...")
    acc, y_true, y_pred, avg_loss = evaluate_standard(
        model, batch_size=args.batch_size, logger=logger)
    results["test_accuracy"] = acc
    results["test_avg_loss"] = avg_loss

    # 分类报告（逐类别 P/R/F1）
    report = classification_report_dict(y_true, y_pred)
    report_text = format_classification_report(report)
    logger.info("\n%s", report_text)
    write_text(report_text, os.path.join(config.RESULT_DIR, "classification_report.txt"))
    write_json(report, os.path.join(config.RESULT_DIR, "classification_report.json"))

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_png = save_confusion_matrix(cm)
    logger.info("混淆矩阵已保存: %s（同步导出 CSV）", cm_png)

    # 最易混淆类别对（报告分析用）
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(cm_off, axis=None)[::-1],
                                       cm_off.shape))[0][:5]
    confusion_pairs = [
        {"true": config.CIFAR10_CLASSES[t], "pred": config.CIFAR10_CLASSES[p],
         "count": int(cm_off[t, p])} for t, p in pairs]
    results["most_confused_pairs"] = confusion_pairs
    logger.info("最易混淆类别对 Top5: %s",
                ", ".join("%s->%s(%d)" % (c["true"], c["pred"], c["count"])
                          for c in confusion_pairs))

    # ---- 2. 鲁棒性测试 ----
    if not args.skip_robust:
        logger.info("[2/3] 鲁棒性测试（每项 %d 张）...", args.robust_num)
        rows = [("标准测试集(无扰动)", acc, "baseline")]
        for name, fn in PERTURBATIONS.items():
            logger.info("  扰动: %s", name)
            acc_p, _, _ = evaluate_perturbed(
                model, fn, num_samples=args.robust_num,
                batch_size=args.batch_size, logger=logger)
            rows.append((name, acc_p, "n=%d" % args.robust_num))
        table = save_comparison_table(rows)
        logger.info("\n%s", table)
        results["robustness"] = [{"name": r[0], "acc": r[1]} for r in rows]

    # ---- 3. 推理耗时 ----
    if not args.skip_timing:
        logger.info("[3/3] 推理耗时分析 ...")
        timing = {}
        for bs in (1, 64, 128):
            timing["batch_%d" % bs] = benchmark_inference(model, batch_size=bs)
            logger.info("  batch=%3d | 平均延迟 %6.2f ms | 吞吐 %8.1f img/s",
                        bs, timing["batch_%d" % bs]["avg_latency_ms"],
                        timing["batch_%d" % bs]["throughput_img_per_s"])
        results["timing"] = timing

    write_json(results, os.path.join(config.RESULT_DIR, "eval_results.json"))
    logger.info("全部评估结果已保存至 %s", config.RESULT_DIR)
    logger.info("评估完成 ✔")


if __name__ == "__main__":
    main()
