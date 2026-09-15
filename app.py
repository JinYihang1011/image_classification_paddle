# -*- coding: utf-8 -*-
"""
app.py —— Streamlit 用户交互界面
================================
功能（对应课程要求"用户交互界面、上传图片、显示 Top-3 类别与置信度、异常处理"）：
  1. 支持 jpg/jpeg/png/bmp/webp 图片上传与拖拽；
  2. 显示上传图片、Top-3 预测类别与置信度（表格 + 柱状图）；
  3. 可展开查看全部 10 类概率分布；
  4. 完整异常处理：模型缺失 / 图片损坏 / 格式不支持 均给出中文提示；
  5. 侧边栏展示模型信息与使用说明。

启动方式：
  streamlit run app.py
  浏览器自动打开 http://localhost:8501
"""

import os
import sys
import time

import paddle
import paddle.nn.functional as F
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from models.resnet import resnet18  # noqa: E402
from data.preprocess import image_to_tensor  # noqa: E402
from utils.io_utils import (load_model, ensure_rgb_pil,  # noqa: E402
                            ensure_dynamic_mode)
from utils.logger import get_logger  # noqa: E402

logger = get_logger("app")

# 飞桨若处于静态图模式，预处理阶段会因 uint8 报错，这里统一切换为动态图模式
ensure_dynamic_mode()


# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CIFAR-10 图像分类系统",
    page_icon="🖼️",
    layout="wide",
)


@st.cache_resource(show_spinner="正在加载模型...")
def load_predictor(model_path):
    """加载并缓存模型（Streamlit 会话内只加载一次）。

    模型文件缺失时抛出异常，由调用方捕获后给出引导提示。
    """
    device = "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
    try:
        paddle.set_device(device)
    except Exception:
        paddle.set_device("cpu")
        device = "cpu"
    model = resnet18()
    model, meta = load_model(model, model_path, device=device)
    model.eval()
    return model, meta, device


# 注意：这里**不能**用 @st.cache_data 缓存。
# Streamlit 会忽略名称以下划线开头的参数（原本写作 _img_bytes 是为了跳过
# 大体量字节流的哈希），结果缓存键退化成只有 (model_path, topk) 恒定不变，
# 于是第一次上传的结果会被后续所有上传复用 —— 表现为"上传新图片仍显示上一张
# 的识别结果"。推理本身在 GPU 上仅约 3ms，无需缓存；昂贵的模型加载已由
# 上面的 @st.cache_resource 单独缓存。
def predict_once(model_path, img_bytes, topk=3):
    """执行推理。"""
    model, meta, device = load_predictor(model_path)
    # 统一归一化为 RGB PIL：兼容任意图片 mode / 位深 / 格式
    img = ensure_rgb_pil(img_bytes)
    # 用自实现的 image_to_tensor 而非 paddle 的 ToTensor：
    # 后者会把 uint8 的 numpy 数组交给 paddle.to_tensor，静态图模式下会报
    # "...but received uint8"；自实现版本与运行模式无关。
    x = image_to_tensor(img, size=config.IMAGE_SIZE).unsqueeze(0)
    # 每次推理前确认动态图模式（Streamlit 可能在不同线程中重跑脚本，
    # 且 @st.cache_resource 会让入口处的守卫被跳过）
    ensure_dynamic_mode(verbose=False)
    t0 = time.time()
    with paddle.no_grad():
        logits = model(x)
        probs = F.softmax(logits, axis=1)[0].numpy()
    elapsed_ms = (time.time() - t0) * 1000

    order = probs.argsort()[::-1]
    topk_list = [(int(i), float(probs[i])) for i in order[:topk]]
    # 开集识别：softmax 过低或 logit 过低（双条件任一）→ 判定「其他」，
    # 与 predict.predict_image 的判定逻辑保持一致
    max_p = float(probs.max())
    max_l = float(logits[0].max().item()) if logits.ndim == 2 else float(logits.max().item())
    is_other = (max_p < config.CONFIDENCE_THRESHOLD) or (max_l < config.LOGIT_THRESHOLD)
    return {
        "img_size": img.size,
        "topk": topk_list,
        "is_other": is_other,
        "threshold": config.CONFIDENCE_THRESHOLD,
        "all_probs": probs.tolist(),
        "elapsed_ms": elapsed_ms,
        "meta": meta,
        "device": device,
    }


# ---------------------------------------------------------------------------
# 界面
# ---------------------------------------------------------------------------
st.title("🖼️ 基于飞桨的 CIFAR-10 图像分类识别系统")
st.caption("东南大学《人工智能实践》课程设计 | ResNet18 + PaddlePaddle | "
           "支持识别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车")

# ---- 侧边栏 ----
with st.sidebar:
    st.header("ℹ️ 模型信息")
    model_path = st.text_input("模型权重路径", value=config.MODEL_PATH)
    st.divider()
    st.markdown(
        "**使用说明**\n"
        "1. 点击「浏览文件」或拖拽图片到上传框\n"
        "2. 支持 jpg / jpeg / png / bmp / webp 格式\n"
        "3. 系统自动缩放到 32×32 后识别\n"
        "4. 结果展示 Top-3 类别与置信度\n\n"
        "**提示**：CIFAR-10 模型对「单个居中物体」这类图片效果最好，"
        "对复杂背景或多物体图片准确率会下降。")

# ---- 模型预加载状态 ----
model_ready = os.path.exists(model_path)
if not model_ready:
    st.error(
        f"未找到模型文件：`{model_path}`\n\n"
        "请先训练模型：在项目根目录执行 `python train.py`，"
        "训练完成后重新刷新本页面。")
    st.stop()

try:
    model, meta, device = load_predictor(model_path)
except Exception as e:  # noqa: BLE001 任何加载异常都以中文提示呈现
    st.error(f"模型加载失败：{e}")
    st.stop()

st.success(
    "模型加载成功 | 设备: **%s** | 飞桨: %s%s" % (
        device.upper(), paddle.__version__,
        " | 训练时验证准确率: %.2f%%" % (meta["val_acc"] * 100)
        if meta.get("val_acc") else ""))

# ---- 图片上传 ----
uploaded = st.file_uploader(
    "上传一张图片进行识别",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="上传后将自动进行预处理并输出 Top-3 预测结果",
)

if uploaded is not None:
    img_bytes = uploaded.getvalue()

    col_img, col_result = st.columns([2, 3])
    with col_img:
        try:
            st.image(img_bytes, caption="待识别图片", use_container_width=True)
        except Exception as e:  # noqa: BLE001 图片展示失败（常见于损坏文件）
            st.error(f"图片显示失败，文件可能已损坏：{e}")
            st.stop()

    with col_result:
        try:
            with st.spinner("识别中..."):
                result = predict_once(model_path, img_bytes, topk=3)
        except Exception as e:  # noqa: BLE001 推理异常兜底，页面不白屏
            logger.exception("UI 推理失败")   # 完整堆栈写入 outputs/logs/run.log
            st.error(f"识别失败：{e}")
            st.caption(
                "排查建议：① 确认图片是 jpg/png 等常见格式且能正常打开；"
                "② 确认 `outputs/best_model.pdparams` 是训练好的模型"
                "（重新训练会在首次保存时自动备份旧模型，备份文件在 outputs/ 下）；"
                "③ 详细错误堆栈已写入 `outputs/logs/run.log`。")
            st.stop()

        # ---- 开集识别：置信度异常时判定为「其他」----
        top_idx, top_prob = result["topk"][0]
        if result["is_other"]:
            st.warning(
                "⚠ **判定为「%s」**：这张图片的置信度特征异常"
                "（最高 softmax %.2f%%，阈值 %.0f%%；或最大 logit 低于 %.1f）。"
                "它很可能不属于本系统已训练的 10 个类别"
                "（飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）。"
                % (config.OTHER_CLASS_CN, top_prob * 100,
                   result["threshold"] * 100, config.LOGIT_THRESHOLD))
            st.metric(
                label="判定结果",
                value="%s（%s）" % (config.OTHER_CLASS_EN, config.OTHER_CLASS_CN),
                delta="最高置信度 %.2f%%（低于判定标准）" % (top_prob * 100),
                delta_color="off")
        else:
            st.metric(
                label="最可能的类别",
                value="%s（%s）" % (config.CIFAR10_CLASSES[top_idx],
                                   config.CIFAR10_CLASSES_CN[top_idx]),
                delta="置信度 %.2f%%" % (top_prob * 100),
                delta_color="off")

        # Top-3 表格（判定为「其他」时首行显示「其他」）
        import pandas as pd
        rows = []
        for i, (idx, prob) in enumerate(result["topk"]):
            rows.append({"排名": i + 1,
                         "类别": config.OTHER_CLASS_EN if (result["is_other"] and i == 0)
                                 else config.CIFAR10_CLASSES[idx],
                         "中文": config.OTHER_CLASS_CN if (result["is_other"] and i == 0)
                                 else config.CIFAR10_CLASSES_CN[idx],
                         "置信度": "%.2f%%" % (prob * 100)})
        st.table(pd.DataFrame(rows).set_index("排名"))

        # Top-3 置信度柱状图
        chart_df = pd.DataFrame(
            {"置信度": [prob * 100 for _, prob in result["topk"]]},
            index=[("%s(%s)" % (config.OTHER_CLASS_EN, config.OTHER_CLASS_CN)
                    if (result["is_other"] and i == 0)
                    else "%s(%s)" % (config.CIFAR10_CLASSES[idx],
                                     config.CIFAR10_CLASSES_CN[idx]))
                   for i, (idx, _) in enumerate(result["topk"])])
        st.bar_chart(chart_df, height=280)

        st.caption("推理耗时 %.1f ms | 原图尺寸 %dx%d | 运行设备 %s" % (
            result["elapsed_ms"], result["img_size"][0], result["img_size"][1],
            result["device"].upper()))

    # ---- 全类别概率分布（可展开） ----
    with st.expander("查看全部 10 类概率分布"):
        all_df = pd.DataFrame({
            "类别": config.CIFAR10_CLASSES,
            "中文": config.CIFAR10_CLASSES_CN,
            "概率": ["%.2f%%" % (p * 100) for p in result["all_probs"]],
        }).set_index("类别")
        st.dataframe(all_df, use_container_width=True)
        st.bar_chart(
            pd.DataFrame({"概率%": [p * 100 for p in result["all_probs"]]},
                         index=config.CIFAR10_CLASSES),
            height=300)

else:
    st.info("⬆️ 请先上传一张图片。没有现成图片？可在命令行运行 "
            "`python eval.py` 使用测试集批量验证模型效果。")

# ---- 页脚 ----
st.divider()
st.caption(
    "技术栈：PaddlePaddle %s · Streamlit · ResNet18（32×32 适配版） | "
    "详细文档见 docs/ 目录" % paddle.__version__)
