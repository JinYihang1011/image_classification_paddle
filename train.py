# -*- coding: utf-8 -*-
"""
train.py —— 模型训练主程序
================================
功能（对应课程要求"模型训练、学习率衰减、早停、保存最佳模型、命令行参数"）：
  1. 解析命令行参数（覆盖 config 中的默认超参数）；
  2. 自动选择 GPU / CPU 设备；
  3. 构建训练/验证 DataLoader、ResNet18 模型、优化器（Adam/Momentum 可选）、
     学习率衰减调度器（阶梯 / 余弦退火）；
  4. 训练循环采用教材《神经网络与深度学习》RunnerV3 风格：
     每 epoch 在验证集上评估一次，验证准确率创新高时保存最佳模型；
  5. 支持早停（Early Stopping），训练曲线保存为 PNG，历史记录保存为 JSON。

用法示例：
  python train.py                          # 使用 config.py 默认超参数
  python train.py --epochs 30 --lr 0.001 --optimizer momentum
  python train.py --smoke                  # 快速冒烟测试（每个 epoch 只跑几步）
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import paddle
import paddle.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from data.dataset import build_dataloader  # noqa: E402
from models.resnet import resnet18, count_parameters  # noqa: E402
from utils.logger import get_logger  # noqa: E402
from utils.io_utils import save_model, write_json  # noqa: E402
from utils.io_utils import ensure_dynamic_mode  # noqa: E402
from utils.metrics import accuracy  # noqa: E402


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="基于飞桨的 CIFAR-10 图像分类训练脚本（ResNet18）")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help="最大训练轮数（默认 %d）" % config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE,
                        help="批次大小（默认 %d）" % config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE,
                        help="初始学习率（默认 %g）" % config.LEARNING_RATE)
    parser.add_argument("--optimizer", type=str, default=config.OPTIMIZER,
                        choices=["adam", "momentum"], help="优化器类型")
    parser.add_argument("--scheduler", type=str, default=config.LR_SCHEDULER,
                        choices=["step", "cosine", "none"], help="学习率衰减策略")
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY,
                        help="L2 正则化系数")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS,
                        help="DataLoader 工作进程数（Windows 建议 0）")
    parser.add_argument("--device", type=str, default=None,
                        choices=["gpu", "cpu"], help="强制指定设备（默认自动检测）")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOP_PATIENCE,
                        help="早停容忍轮数")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟测试模式：每 epoch 只跑少量 step，验证流程可用")
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定权重恢复训练（.pdparams 路径）")
    parser.add_argument("--seed", type=int, default=config.SEED, help="随机种子")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 训练器（教材 RunnerV3 风格：train / evaluate / predict 三位一体）
# ---------------------------------------------------------------------------
class Runner:
    """训练器：封装训练循环、验证、最佳模型保存与早停逻辑。"""

    def __init__(self, model, optimizer, scheduler, loss_fn, logger,
                 log_interval=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.logger = logger
        self.log_interval = log_interval or config.LOG_INTERVAL
        # 训练历史（用于绘图与报告）
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [], "lr": [],
        }
        self.best_acc = -1.0        # 历史最佳验证准确率
        self.best_epoch = -1        # 最佳模型对应的 epoch
        self.backup_done = False    # 是否已备份过原有模型（每次训练只备份一次）

    def _backup_existing_model(self):
        """首次覆盖 best_model.pdparams 之前，把已有模型另存备份。

        避免"重新训练一次就把之前辛苦训练好的模型覆盖掉"这类不可逆损失
        （实训中真实发生过：一次中断的训练把 92.56% 的模型覆盖成了 70% 的半成品）。
        """
        if self.backup_done:
            return
        self.backup_done = True
        if not os.path.exists(config.MODEL_PATH):
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(config.OUTPUT_DIR,
                              "best_model_backup_%s.pdparams" % stamp)
        try:
            shutil.copy2(config.MODEL_PATH, backup)
            self.logger.info("已自动备份原有最佳模型 -> %s", backup)
        except Exception as e:  # noqa: BLE001 备份失败不应中断训练
            self.logger.warning("备份原有模型失败（不影响本次训练）: %s", e)

    # ---- 单个 epoch 的训练 ----
    def train_epoch(self, train_loader, epoch, max_steps=None):
        self.model.train()
        total_loss, total_correct, total_num = 0.0, 0, 0
        t_start = time.time()

        for step, (imgs, labels) in enumerate(train_loader()):
            logits = self.model(imgs)
            loss = self.loss_fn(logits, labels)

            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()

            # 统计指标
            n = imgs.shape[0]
            total_loss += float(loss.item()) * n
            total_correct += paddle.sum(
                paddle.argmax(logits, axis=1) == labels).item()
            total_num += n

            if (step + 1) % self.log_interval == 0:
                self.logger.info(
                    "epoch %d step %4d | loss %.4f | acc %.4f | %.1f img/s",
                    epoch, step + 1, total_loss / total_num,
                    total_correct / total_num,
                    total_num / (time.time() - t_start))

            if max_steps is not None and step + 1 >= max_steps:
                break  # 冒烟测试模式提前截断

        return total_loss / max(total_num, 1), total_correct / max(total_num, 1)

    # ---- 验证 / 测试 ----
    @paddle.no_grad()
    def evaluate(self, data_loader, max_steps=None):
        self.model.eval()
        total_loss, total_correct, total_num = 0.0, 0, 0
        for step, (imgs, labels) in enumerate(data_loader()):
            logits = self.model(imgs)
            loss = self.loss_fn(logits, labels)
            n = imgs.shape[0]
            total_loss += float(loss.item()) * n
            total_correct += paddle.sum(
                paddle.argmax(logits, axis=1) == labels).item()
            total_num += n
            if max_steps is not None and step + 1 >= max_steps:
                break
        return total_loss / max(total_num, 1), total_correct / max(total_num, 1)

    # ---- 完整训练流程（含最佳模型保存、学习率调度、早停） ----
    def train(self, train_loader, val_loader, num_epochs, smoke=False):
        patience_cnt = 0
        max_steps = 5 if smoke else None          # 冒烟：每 epoch 5 步
        val_max_steps = 2 if smoke else None      # 冒烟：验证 2 步

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(
                train_loader, epoch, max_steps=max_steps)
            val_loss, val_acc = self.evaluate(val_loader, max_steps=val_max_steps)

            # 学习率调度：调度器与优化器绑定后，step() 会自动同步优化器学习率
            if self.scheduler is not None:
                self.scheduler.step()
            current_lr = self.optimizer.get_lr()

            # 记录历史
            self.history["train_loss"].append(round(train_loss, 6))
            self.history["train_acc"].append(round(train_acc, 6))
            self.history["val_loss"].append(round(val_loss, 6))
            self.history["val_acc"].append(round(val_acc, 6))
            self.history["lr"].append(current_lr)

            self.logger.info(
                "epoch %3d/%d | train_loss %.4f acc %.4f | val_loss %.4f acc %.4f"
                " | lr %.2e | %.1fs",
                epoch, num_epochs, train_loss, train_acc, val_loss, val_acc,
                current_lr, time.time() - t0)

            # ---- 保存最佳模型（验证准确率创新高时） ----
            if val_acc > self.best_acc:
                self.best_acc, self.best_epoch = val_acc, epoch
                patience_cnt = 0
                self._backup_existing_model()   # 首次覆盖前自动备份旧模型
                save_model(self.model, config.MODEL_PATH,
                           meta={"epoch": epoch, "val_acc": val_acc,
                                 "val_loss": val_loss, "train_acc": train_acc})
                self.logger.info(">>> 验证准确率创新高 %.4f，最佳模型已保存至 %s",
                                 val_acc, config.MODEL_PATH)
            else:
                patience_cnt += 1
                # ---- 早停 ----
                if patience_cnt >= self.args_patience:
                    self.logger.info(
                        "验证准确率已连续 %d 个 epoch 未提升，触发早停。"
                        "最佳 epoch=%d, acc=%.4f",
                        self.args_patience, self.best_epoch, self.best_acc)
                    break

            # ---- 每个 epoch 另存一份最新权重（便于断点续训 / 对比实验）----
            try:
                save_model(self.model, config.LAST_MODEL_PATH,
                           meta={"epoch": epoch, "val_acc": val_acc,
                                 "stage": "last"})
            except Exception as e:  # noqa: BLE001 附属产物失败不中断训练
                self.logger.warning("保存 last_model 失败: %s", e)

        return self.history

    args_patience = config.EARLY_STOP_PATIENCE  # 默认容忍度（可被 main 覆盖）


# ---------------------------------------------------------------------------
# 学习率曲线与训练曲线绘图
# ---------------------------------------------------------------------------
def plot_history(history, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from utils.metrics import setup_chinese_font
    setup_chinese_font()

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].plot(epochs, history["train_loss"], label="train loss", color="#8E004D")
    axes[0].plot(epochs, history["val_loss"], "--", label="val loss", color="#E20079")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("损失曲线"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train acc", color="#1f77b4")
    axes[1].plot(epochs, history["val_acc"], "--", label="val acc", color="#ff7f0e")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].set_title("准确率曲线"); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history["lr"], label="learning rate", color="green")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("lr")
    axes[2].set_title("学习率衰减"); axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    png = out_prefix + ".png"
    plt.savefig(png, dpi=150)
    plt.close(fig)
    return png


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    config.ensure_dirs()

    # ---- 随机种子（保证可复现） ----
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    # ---- 设备选择：GPU 优先 ----
    if args.device is not None:
        device = args.device
    else:
        device = "gpu" if paddle.device.is_compiled_with_cuda() else "cpu"
    try:
        paddle.set_device(device)
    except Exception as e:
        print("设备 %s 不可用（%s），回退到 CPU。" % (device, e))
        paddle.set_device("cpu")
        device = "cpu"

    ensure_dynamic_mode()   # 静态图模式下预处理会因 uint8 报错，统一切换动态图
    logger = get_logger("train")
    logger.info("=" * 60)
    logger.info("训练启动 | 设备: %s | 飞桨版本: %s", device, paddle.__version__)
    if device.startswith("gpu"):
        logger.info("GPU: %s", paddle.device.cuda.device_count() and "可用" or "不可用")
    # 覆盖提醒：避免在演示/提交前误把已训练好的模型覆盖成半成品
    if os.path.exists(config.MODEL_PATH):
        logger.warning("注意：本次训练会在验证准确率提升时覆盖已有模型 %s",
                       config.MODEL_PATH)
        logger.warning("      原有模型会在首次保存前自动备份为 "
                       "outputs/best_model_backup_<时间戳>.pdparams")
        logger.warning("      仅想使用现有模型请勿运行本脚本，直接执行 "
                       "predict.py / eval.py / app.py 即可")

    # ---- 数据 ----
    logger.info("构建数据集（首次运行会自动下载 CIFAR-10）...")
    train_loader, train_ds = build_dataloader(
        "train", batch_size=args.batch_size, num_workers=args.num_workers)
    val_loader, val_ds = build_dataloader(
        "val", batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers)
    logger.info("训练集 %d 张 / 验证集 %d 张 / 批次大小 %d",
                len(train_ds), len(val_ds), args.batch_size)

    # ---- 模型 ----
    model = resnet18()
    if args.resume:
        from utils.io_utils import load_model
        model, meta = load_model(model, args.resume)
        logger.info("从 %s 恢复训练（该权重对应验证 acc=%s）",
                    args.resume, meta.get("val_acc", "未知"))
    logger.info("模型: ResNet18(CIFAR-10) | 参数量 %.2f M",
                count_parameters(model) / 1e6)

    # ---- 损失函数：交叉熵（多分类标准选择，与教材一致） ----
    loss_fn = F.cross_entropy

    # ---- 学习率调度器 ----
    if args.scheduler == "step":
        scheduler = paddle.optimizer.lr.StepDecay(
            learning_rate=args.lr, step_size=config.LR_STEP_SIZE,
            gamma=config.LR_GAMMA)
    elif args.scheduler == "cosine":
        scheduler = paddle.optimizer.lr.CosineAnnealingDecay(
            learning_rate=args.lr, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None

    # ---- 优化器 ----
    if args.optimizer == "adam":
        optimizer = paddle.optimizer.Adam(
            learning_rate=scheduler if scheduler is not None else args.lr,
            parameters=model.parameters(),
            weight_decay=args.weight_decay)
    else:  # momentum
        optimizer = paddle.optimizer.Momentum(
            learning_rate=scheduler if scheduler is not None else args.lr,
            momentum=config.MOMENTUM,
            parameters=model.parameters(),
            weight_decay=args.weight_decay)
    logger.info("优化器: %s | 学习率: %g | 调度: %s | weight_decay: %g",
                args.optimizer, args.lr, args.scheduler, args.weight_decay)

    # ---- 训练 ----
    runner = Runner(model, optimizer, scheduler, loss_fn, logger)
    runner.args_patience = args.patience  # 早停容忍度
    if args.smoke:
        logger.info("*** 冒烟测试模式：只验证流程正确性，不追求精度 ***")
    history = runner.train(train_loader, val_loader, num_epochs=args.epochs,
                           smoke=args.smoke)

    # ---- 训练产物落盘 ----
    write_json(history, config.HISTORY_PATH)
    curve_png = plot_history(history, os.path.join(config.RESULT_DIR, "train_curves"))
    logger.info("训练历史已保存: %s", config.HISTORY_PATH)
    logger.info("训练曲线已保存: %s", curve_png)
    logger.info("最佳模型: %s（epoch %d, 验证 acc %.4f）",
                config.MODEL_PATH, runner.best_epoch, runner.best_acc)
    logger.info("训练完成。下一步可运行: python eval.py")


if __name__ == "__main__":
    main()
