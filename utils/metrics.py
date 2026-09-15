# -*- coding: utf-8 -*-
"""
utils/metrics.py —— 评估指标计算与可视化
================================
职责（对应课程要求"准确率、精确率、召回率、F1、混淆矩阵"）：
  1. top-k 准确率计算（基于飞桨张量）；
  2. 精确率 / 召回率 / F1 分类报告（逐类别 + 宏平均）；
  3. 混淆矩阵计算、CSV 导出与热力图绘制；
  4. 鲁棒性测试准确率对比表生成。

优先使用 scikit-learn 实现（结果与主流工具一致），
若环境中未安装 sklearn 则自动退化为纯 numpy 实现，保证程序可运行。
"""

import os
import sys

import numpy as np
import paddle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

try:
    from sklearn.metrics import (confusion_matrix as _sk_cm,
                                 precision_recall_fscore_support)
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# 0. 绘图中文字体配置（Windows: 微软雅黑/黑体; Linux/macOS 常见中文字体）
# ---------------------------------------------------------------------------
def setup_chinese_font():
    """让 matplotlib 正常显示中文（防止图题/坐标轴出现方框乱码）。

    在任何绘图前调用一次即可；找不到中文字体时静默降级为英文默认字体。
    """
    import matplotlib
    from matplotlib import font_manager
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC",
                  "Noto Sans CJK SC", "WenQuanYi Zen Hei", "Arial Unicode MS"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [c for c in candidates if c in installed]
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = chosen + \
            matplotlib.rcParams["font.sans-serif"]
    matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示


# ---------------------------------------------------------------------------
# 1. 准确率
# ---------------------------------------------------------------------------
def accuracy(logits, labels, topk=(1,)):
    """计算 top-k 准确率。

    Args:
        logits (Tensor): 形状 [N, num_classes] 的未归一化输出。
        labels (Tensor): 形状 [N] 的真实标签。
        topk (tuple): 要计算的 k 值集合，如 (1,) 或 (1, 3)。

    Returns:
        list[float]: 与 topk 顺序对应的准确率（0~100 的百分数）。
    """
    maxk = max(topk)
    batch_size = labels.shape[0]

    # 取概率最大的前 maxk 个类别的索引
    _, pred = paddle.topk(logits, k=maxk, axis=1)
    pred = paddle.transpose(pred, [1, 0])          # [maxk, N]
    correct = paddle.equal(pred, labels.reshape([1, -1]).expand_as(pred))  # [maxk, N]

    res = []
    for k in topk:
        correct_k = paddle.sum(correct[:k].astype("float32")).item()
        res.append(correct_k * 100.0 / batch_size)
    return res


def accuracy_from_preds(y_true, y_pred):
    """由预测类别序列计算准确率（0~1 之间的小数）。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


# ---------------------------------------------------------------------------
# 2. 精确率 / 召回率 / F1 分类报告
# ---------------------------------------------------------------------------
def _prf_fallback(y_true, y_pred, num_classes):
    """纯 numpy 的 precision/recall/f1 实现（sklearn 缺失时的兜底方案）。"""
    result = {}
    eps = 1e-12
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        result[c] = (precision, recall, f1, int(np.sum(y_true == c)))
    return result


def classification_report_dict(y_true, y_pred, class_names=None):
    """输出逐类别 P/R/F1 与宏平均，返回结构化字典。

    Returns:
        dict: {
            "per_class": {类名: {"precision","recall","f1-score","support"}},
            "macro_avg": {"precision","recall","f1-score","support"},
            "accuracy": float,
        }
    """
    class_names = class_names or config.CIFAR10_CLASSES
    num_classes = len(class_names)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    per_class = {}
    if _HAS_SKLEARN:
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=list(range(num_classes)),
            zero_division=0)
        for i, name in enumerate(class_names):
            per_class[name] = {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1-score": round(float(f[i]), 4),
                "support": int(s[i]),
            }
    else:
        raw = _prf_fallback(y_true, y_pred, num_classes)
        for i, name in enumerate(class_names):
            pi, ri, fi, si = raw[i]
            per_class[name] = {
                "precision": round(pi, 4), "recall": round(ri, 4),
                "f1-score": round(fi, 4), "support": int(si),
            }

    # 宏平均：对 10 个类别求算术平均
    macro = {
        "precision": round(float(np.mean([v["precision"] for v in per_class.values()])), 4),
        "recall": round(float(np.mean([v["recall"] for v in per_class.values()])), 4),
        "f1-score": round(float(np.mean([v["f1-score"] for v in per_class.values()])), 4),
        "support": int(sum(v["support"] for v in per_class.values())),
    }
    return {
        "per_class": per_class,
        "macro_avg": macro,
        "accuracy": round(accuracy_from_preds(y_true, y_pred), 4),
    }


def format_classification_report(report, class_names=None):
    """将 classification_report_dict 格式化为对齐的文本表格（控制台友好）。"""
    class_names = class_names or config.CIFAR10_CLASSES
    lines = ["%-12s %10s %10s %10s %10s" % ("class", "precision", "recall", "f1", "support")]
    for name in class_names:
        v = report["per_class"][name]
        lines.append("%-12s %10.4f %10.4f %10.4f %10d" % (
            name, v["precision"], v["recall"], v["f1-score"], v["support"]))
    m = report["macro_avg"]
    lines.append("-" * 55)
    lines.append("%-12s %10.4f %10.4f %10.4f %10d" % (
        "macro avg", m["precision"], m["recall"], m["f1-score"], m["support"]))
    lines.append("%-12s %10s %10s %10.4f %10d" % ("accuracy", "", "", report["accuracy"], m["support"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. 混淆矩阵
# ---------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, num_classes=None):
    """计算混淆矩阵，返回 [num_classes, num_classes] 的 numpy 数组。

    行为真实类别、列为预测类别；cm[i][j] 表示真实为 i 被预测为 j 的样本数。
    """
    num_classes = num_classes or config.NUM_CLASSES
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if _HAS_SKLEARN:
        return _sk_cm(y_true, y_pred, labels=list(range(num_classes)))
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_confusion_matrix(cm, out_path=None, class_names=None, normalize=True):
    """将混淆矩阵绘制为热力图并保存 PNG，同时导出 CSV。

    Args:
        cm (np.ndarray): 混淆矩阵。
        out_path (str): PNG 保存路径（同时生成同名 .csv）。
        class_names (list): 类别名称。
        normalize (bool): 是否按行归一化显示百分比（原始计数写入 CSV）。
    """
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境也能绘图
    import matplotlib.pyplot as plt
    setup_chinese_font()

    class_names = class_names or config.CIFAR10_CLASSES
    out_path = out_path or os.path.join(config.RESULT_DIR, "confusion_matrix.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cm = np.asarray(cm)
    show = cm.astype(np.float64)
    if normalize:
        row_sums = show.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 防止除零
        show = show / row_sums

    fig, ax = plt.subplots(figsize=(8, 6.5))
    try:
        import seaborn as sns
        sns.heatmap(show, annot=True, fmt=".2f" if normalize else "d",
                    cmap="Blues", xticklabels=class_names,
                    yticklabels=class_names, ax=ax, cbar=True,
                    annot_kws={"size": 8})
    except ImportError:
        # seaborn 缺失时退化为 matplotlib imshow
        im = ax.imshow(show, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels(class_names, fontsize=8)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, "%.2f" % show[i, j], ha="center", va="center",
                        fontsize=7, color="black")

    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("混淆矩阵%s" % ("（按行归一化）" if normalize else ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    # 同步导出原始计数 CSV（便于报告引用）
    csv_path = os.path.splitext(out_path)[0] + ".csv"
    try:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["真实\\预测"] + class_names)
            for i, name in enumerate(class_names):
                writer.writerow([name] + cm[i].tolist())
    except Exception as e:  # noqa: BLE001 CSV 导出失败不影响主流程
        print("[metrics] 混淆矩阵 CSV 导出失败: %s" % e)
    return out_path


# ---------------------------------------------------------------------------
# 4. 对比表（鲁棒性测试等场景）
# ---------------------------------------------------------------------------
def save_comparison_table(rows, out_path=None):
    """将 [(测试集名称, 准确率, 备注), ...] 保存为 Markdown 表格 + CSV。

    Args:
        rows (list[tuple]): (名称, 准确率0~1, 备注)。
        out_path (str): 输出 .md 路径（同时生成同名 .csv）。

    Returns:
        str: Markdown 表格文本。
    """
    out_path = out_path or os.path.join(config.RESULT_DIR, "comparison_table.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    lines = ["| 测试集 | 准确率 | 备注 |", "|---|---|---|"]
    for name, acc, note in rows:
        lines.append("| %s | %.2f%% | %s |" % (name, acc * 100, note or ""))
    table = "\n".join(lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    try:
        import csv
        with open(os.path.splitext(out_path)[0] + ".csv", "w",
                  newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["测试集", "准确率", "备注"])
            for name, acc, note in rows:
                writer.writerow([name, "%.4f" % acc, note or ""])
    except Exception as e:  # noqa: BLE001
        print("[metrics] 对比表 CSV 导出失败: %s" % e)
    return table


if __name__ == "__main__":
    # 自检：构造随机预测结果，验证各指标函数
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 10, size=100)
    y_pred = np.where(rng.rand(100) < 0.8, y_true, rng.randint(0, 10, size=100))
    rep = classification_report_dict(y_true, y_pred)
    print(format_classification_report(rep))
    cm = confusion_matrix(y_true, y_pred)
    print("混淆矩阵形状:", cm.shape, "对角线和(正确样本数):", cm.trace())
    p = save_confusion_matrix(cm)
    print("混淆矩阵图:", p)
    logits = paddle.randn([16, 10])
    labels = paddle.randint(0, 10, [16])
    print("top-1 acc:", accuracy(logits, labels, topk=(1,)))
