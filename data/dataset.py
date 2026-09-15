# -*- coding: utf-8 -*-
"""
data/dataset.py —— 数据集组织、下载与 DataLoader 构建
================================
职责（对应课程要求"数据集组织与管理、数据文件读写"）：
  1. 自动下载 CIFAR-10 tar 包到项目 data/cifar10/ 目录（带 MD5 校验与断点重试）；
  2. 封装 paddle.vision.datasets.Cifar10，按固定随机种子划分训练集/验证集；
  3. 构建 paddle.io.DataLoader，供训练、评估、推理统一调用。

设计说明（与教材 5.5 节保持一致的风格）：
  教材中 CIFAR10Dataset 继承 paddle.io.Dataset，__getitem__ 返回 (变换后图像, 标签)；
  本项目沿用该写法，但底层数据读取改用飞桨官方 paddle.vision.datasets.Cifar10
  （自动下载 + 官方维护），并在其上增加 train/val 划分能力。

paddle 3.x 中类名为 Cifar10，2.x 旧版为 CIFAR10，本模块做了双版本兼容。
"""

import hashlib
import os
import sys
import time
import urllib.request

import numpy as np
import paddle
from paddle.io import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from data.preprocess import build_transform  # noqa: E402


def _get_cifar10_cls():
    """兼容不同飞桨版本：3.x 叫 Cifar10，2.x 部分版本叫 CIFAR10。"""
    try:
        from paddle.vision.datasets import Cifar10 as _C
        return _C
    except ImportError:
        from paddle.vision.datasets import CIFAR10 as _C  # noqa: F401
        return _C


def _md5(path, chunk_size=1 << 20):
    """计算文件 MD5（流式读取，避免 170MB 文件占用过多内存）。"""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_cifar10(data_file=None, max_retry=3):
    """下载 CIFAR-10 数据集到项目 data/cifar10/ 目录（带 MD5 校验与重试）。

    Args:
        data_file (str): tar 包目标路径，默认取 config.DATA_FILE。
        max_retry (int): 最大重试次数。

    Returns:
        str: 就绪的 tar 包路径。

    Raises:
        RuntimeError: 多次重试后仍未获得校验通过的文件。
    """
    if data_file is None:
        data_file = config.DATA_FILE
    os.makedirs(os.path.dirname(data_file), exist_ok=True)

    # 已存在且校验通过 -> 直接复用（数据文件读写管理：避免重复下载）
    if os.path.exists(data_file) and _md5(data_file) == config.DATA_MD5:
        print("[dataset] CIFAR-10 已存在，校验通过:", data_file)
        return data_file

    # 已存在但校验失败 -> 损坏文件，删除后重新下载
    if os.path.exists(data_file):
        print("[dataset] 检测到损坏的数据文件（MD5 不符），将重新下载")
        os.remove(data_file)

    for attempt in range(1, max_retry + 1):
        try:
            print("[dataset] 第 %d 次尝试下载 CIFAR-10 ..." % attempt)
            tmp = data_file + ".part"
            urllib.request.urlretrieve(config.DATA_URL, tmp)
            if _md5(tmp) != config.DATA_MD5:
                raise IOError("MD5 校验失败")
            os.replace(tmp, data_file)  # 原子替换，避免中途文件被误用
            print("[dataset] 下载完成:", data_file)
            return data_file
        except Exception as e:  # noqa: BLE001 网络类异常种类繁多，统一捕获重试
            print("[dataset] 下载失败: %s" % e)
            if attempt == max_retry:
                raise RuntimeError(
                    "CIFAR-10 下载失败（已重试 %d 次）。请检查网络后重试，"
                    "或手动下载 %s 并放置到 %s" % (max_retry, config.DATA_URL, data_file)
                ) from e
            time.sleep(3)


class CIFAR10Dataset(Dataset):
    """CIFAR-10 数据集封装类（继承 paddle.io.Dataset，与教材风格一致）。

    - mode="train"：50000 张训练图按固定种子随机划分为训练/验证两个子集；
    - mode="test" ：10000 张官方测试图；
    - __getitem__ 返回 (变换后图像张量 CHW float32, int 标签)。

    Args:
        mode (str): "train" / "val" / "test"。
        transform: paddle.vision.transforms 变换流水线；None 时自动按 mode 选择。
        val_ratio (float): 验证集比例（仅 mode 为 train/val 时生效）。
        seed (int): 划分用随机种子，保证每次运行划分一致。
        download (bool): 数据不存在时是否自动下载。
    """

    def __init__(self, mode="train", transform=None, val_ratio=None,
                 seed=None, download=True):
        super().__init__()
        assert mode in ("train", "val", "test"), "mode 必须是 train/val/test"
        self.mode = mode
        self.transform = transform if transform is not None else \
            build_transform(is_train=(mode == "train"))

        # ---- 数据文件就绪（下载 + MD5 校验） ----
        tar_path = download_cifar10() if download else config.DATA_FILE
        if not os.path.exists(tar_path):
            raise FileNotFoundError(
                "未找到数据文件 %s，且 download=False。请先运行 download_cifar10()" % tar_path
            )

        _Cifar10 = _get_cifar10_cls()
        # 底层统一用 pil 后端，保证 transforms 兼容
        base_train = _Cifar10(data_file=tar_path, mode="train", backend="pil")
        base_test = _Cifar10(data_file=tar_path, mode="test", backend="pil") \
            if mode == "test" else None

        if mode == "test":
            self._base = base_test
            self.indices = list(range(len(base_test)))
        else:
            # ---- 固定种子的随机划分：train / val ----
            val_ratio = config.VAL_RATIO if val_ratio is None else val_ratio
            seed = config.SEED if seed is None else seed
            n = len(base_train)
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n)
            n_val = int(n * val_ratio)
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            self._base = base_train
            self.indices = (train_idx.tolist() if mode == "train" else val_idx.tolist())

    def __getitem__(self, idx):
        """返回第 idx 个样本：(标准化后的 CHW Tensor, int 标签)。"""
        real_idx = self.indices[idx]
        img, label = self._base[real_idx]
        if self.transform is not None:
            img = self.transform(img)  # PIL 图 -> 标准化张量
        return img, int(label)

    def __len__(self):
        return len(self.indices)

    @property
    def num_samples(self):
        return len(self)


def build_dataloader(mode="train", batch_size=None, shuffle=None,
                     num_workers=None, transform=None, download=True):
    """构建 DataLoader（统一入口，训练/评估/推理共用）。

    Args:
        mode (str): "train" / "val" / "test"。
        batch_size (int): 批次大小，默认取 config.BATCH_SIZE。
        shuffle (bool): 是否打乱，默认仅训练集打乱。
        num_workers (int): 工作进程数，默认取 config.NUM_WORKERS。
        transform: 覆盖默认变换流水线（鲁棒性测试时使用）。
        download (bool): 是否允许自动下载数据。

    Returns:
        (paddle.io.DataLoader, CIFAR10Dataset)
    """
    batch_size = config.BATCH_SIZE if batch_size is None else batch_size
    num_workers = config.NUM_WORKERS if num_workers is None else num_workers
    if shuffle is None:
        shuffle = (mode == "train")

    dataset = CIFAR10Dataset(mode=mode, transform=transform, download=download)
    loader = paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        return_list=True,  # 返回普通 list[Tensor]，便于统一处理
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# 自检：python data/dataset.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("== 数据集自检（首次运行会自动下载约 170MB 数据） ==")
    for m in ("train", "val", "test"):
        ds = CIFAR10Dataset(mode=m)
        img, label = ds[0]
        print("mode=%-5s 样本数=%6d  单样本: shape=%s dtype=%s label=%d" % (
            m, len(ds), tuple(img.shape), str(img.dtype), label))
    loader, _ = build_dataloader("train", batch_size=64)
    imgs, labels = next(iter(loader))
    print("一个批次: images", tuple(imgs.shape), "labels", tuple(labels.shape))
