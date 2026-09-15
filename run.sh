#!/usr/bin/env bash
# ============================================================
# run.sh —— 一键运行脚本（Windows Git Bash / Linux / macOS 通用）
# 用法：
#   bash run.sh          # 依次执行：依赖检查 -> 训练 -> 评估 -> 测试
#   bash run.sh train    # 只训练
#   bash run.sh eval     # 只评估
#   bash run.sh test     # 只跑单元测试
#   bash run.sh ui       # 启动 Streamlit 界面
# ============================================================
set -e  # 任一步骤失败立即退出

cd "$(dirname "$0")"   # 切换到项目根目录

PY=python
ACTION="${1:-all}"

echo "=============================================="
echo " 基于飞桨的 CIFAR-10 图像分类识别系统"
echo "=============================================="
$PY -c "import paddle; print('飞桨版本:', paddle.__version__, '| 设备:', paddle.get_device())"

case "$ACTION" in
  train)
    $PY train.py
    ;;
  eval)
    $PY eval.py
    ;;
  test)
    $PY -m pytest tests/ -v
    ;;
  ui)
    $PY -m streamlit run app.py
    ;;
  all)
    echo "--- [1/4] 运行单元测试 ---"
    $PY -m pytest tests/ -v
    echo "--- [2/4] 训练模型 ---"
    $PY train.py
    echo "--- [3/4] 评估与鲁棒性测试 ---"
    $PY eval.py
    echo "--- [4/4] 完成！---"
    echo "启动图形界面: $PY -m streamlit run app.py"
    echo "单张图片预测: $PY predict.py --image 你的图片.jpg"
    ;;
  *)
    echo "未知操作: $ACTION（可用: train / eval / test / ui / all）"
    exit 1
    ;;
esac
