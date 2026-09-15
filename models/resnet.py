# -*- coding: utf-8 -*-
"""
models/resnet.py —— 基于 paddle.nn 原生算子搭建的 ResNet18（适配 32x32 输入）
================================
参考教材：邱锡鹏《神经网络与深度学习》5.4/5.5 节 ResNet 案例
（Practice-in-Paddle-main/chap5卷积神经网络）。

与标准 ImageNet 版 ResNet18 的差异（32x32 小图适配，业内通用做法）：
  1. 首层卷积改为 3x3、stride=1（原版 7x7、stride=2）；
  2. 去掉首个 MaxPool（否则 32x32 特征图会在第 2 个残差阶段就变成 1x1）；
  3. 末端使用全局平均池化 AdaptiveAvgPool2D(1)，对特征图尺寸不敏感。

网络结构（CIFAR-10 版 ResNet18）：
  输入 3x32x32
  -> Conv3x3(64) + BN + ReLU            输出 64x32x32
  -> 残差阶段1 (2 x BasicBlock, 64)     输出 64x32x32
  -> 残差阶段2 (2 x BasicBlock, 128)    输出 128x16x16（首个 block 下采样）
  -> 残差阶段3 (2 x BasicBlock, 256)    输出 256x8x8
  -> 残差阶段4 (2 x BasicBlock, 512)    输出 512x4x4
  -> 全局平均池化 -> FC(512 -> 10)
"""

import os
import sys

import paddle
import paddle.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 卷积 + 不使用偏置（后接 BN 时偏置冗余，可减少参数并稳定训练）。"""
    return nn.Conv2D(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=1, bias_attr=False,
    )


class BasicBlock(nn.Layer):
    """ResNet 基础残差块：两层 3x3 卷积 + 跳跃连接（恒等/1x1 卷积 shortcuts）。

    输出通道数 = out_channels * expansion（expansion 恒为 1）。
    当需要下采样（stride>1）或输入输出通道不一致时，shortcut 分支用
    1x1 卷积调整维度，保证主分支与跳跃分支可以逐元素相加。
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主分支：Conv-BN-ReLU -> Conv-BN
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2D(out_channels)

        # 跳跃分支：维度一致且不下采样时为恒等映射，否则用 1x1 卷积对齐
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(out_channels * self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # 残差相加（核心：让网络学习残差映射）
        out = self.relu(out)
        return out


class ResNet(nn.Layer):
    """通用 ResNet 骨架，通过 block 数量配置可实例化 ResNet18/34 等。"""

    def __init__(self, block, num_blocks, num_classes=None):
        super().__init__()
        num_classes = config.NUM_CLASSES if num_classes is None else num_classes
        self.in_channels = 64

        # 首层：3x3、stride=1、无 MaxPool —— 保留 32x32 的空间分辨率
        self.conv1 = conv3x3(config.IMAGE_CHANNELS, 64)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()

        # 四个残差阶段：通道 64 -> 128 -> 256 -> 512，除第一阶段外均下采样
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 全局平均池化 + 全连接分类头
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_block, stride):
        """构建一个残差阶段：第一个 block 负责下采样，其余 stride=1。"""
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        cur_in = self.in_channels
        for s in strides:
            layers.append(block(cur_in, out_channels, s))
            cur_in = out_channels * block.expansion
        self.in_channels = cur_in
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)          # [N, 512, 1, 1]
        out = paddle.flatten(out, 1)      # [N, 512]
        out = self.fc(out)                # [N, num_classes] 输出 logits
        return out


def resnet18(num_classes=None, pretrained=False):
    """构建适配 CIFAR-10 (32x32) 的 ResNet18。

    Args:
        num_classes (int): 分类类别数，默认 10。
        pretrained (bool): 是否加载 paddle 官方 ImageNet 预训练权重。
            注意：官方预训练模型按 224x224 训练，且首层结构不同，
            对 32x32 小图收益有限，默认关闭。
    """
    if pretrained:
        # 官方预训练权重针对 224x224 输入，与本项目 32x32 结构不兼容，
        # 这里显式提示并忽略，避免静默加载导致维度错误。
        print("[models] 警告: 32x32 版 ResNet18 结构与官方预训练权重不一致，"
              "忽略 pretrained 参数。")
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def count_parameters(model):
    """统计可训练参数总量（用于报告与模型复杂度分析）。"""
    return sum(int(p.numel()) for p in model.parameters() if not p.stop_gradient)


# ---------------------------------------------------------------------------
# 自检：python models/resnet.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    paddle.set_device("gpu" if paddle.device.is_compiled_with_cuda() else "cpu")
    model = resnet18()
    model.eval()
    print("模型结构:\n", model)
    print("参数总量: %.2f M" % (count_parameters(model) / 1e6))

    x = paddle.randn([4, config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE])
    y = model(x)
    print("输入:", tuple(x.shape), "-> 输出:", tuple(y.shape))
    assert y.shape == (4, config.NUM_CLASSES), "输出维度错误！"

    # 统计各阶段输出尺寸，验证 32x32 适配正确性
    feats = model.conv1(x); print("首层输出:", tuple(feats.shape))
    feats = model.layer1(feats); print("layer1  :", tuple(feats.shape))
    feats = model.layer2(feats); print("layer2  :", tuple(feats.shape))
    feats = model.layer3(feats); print("layer3  :", tuple(feats.shape))
    feats = model.layer4(feats); print("layer4  :", tuple(feats.shape))
    print("自检通过 ✔")
