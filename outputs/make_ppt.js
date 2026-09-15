// =============================================================================
// ⚠️ 历史开发脚本，与当前 PPT 已不同步 —— 请勿直接运行覆盖 课程设计汇报.pptx
// -----------------------------------------------------------------------------
// 说明：本脚本最初用于生成 课程设计汇报.pptx，之后 PPT 由人工多次修改
//       （首页署名改为五位组员、备注与页码对齐、接口表条数、开集识别、
//         演示图与各项指标更新等），本脚本无法复现这些改动。
//       现仅作历史参考保留；如需再次自动生成，请先人工核对并对齐全部内容。
// 指标已同步至 2026-09-15 最新一次训练与评估（92.07% / 第 47 轮 / 57 轮早停）。
// =============================================================================

// 生成《基于飞桨的 CIFAR-10 图像分类识别系统》课程设计汇报 PPT
// 用 法: NODE_PATH=<npm root -g> node make_ppt.js
const pptxgen = require("pptxgenjs");
const path = require("path");

const PROJ = path.resolve(__dirname, "..");
const OUT = path.join(PROJ, "课程设计汇报.pptx");
const IMG = (p) => path.join(PROJ, p);

// ── 画布与版式常量 ────────────────────────────────────────────────
const W = 13.33, H = 7.5, M = 0.6;
const CW = W - 2 * M;              // 内容宽度

// ── 配色（取自"飞桨/水"的深青主色 + 琥珀强调色）────────────────
const DARK = "0D3A4C";             // 深色页背景
const DARK2 = "14526B";            // 深色页次级块
const LIGHT = "F4F7F8";            // 浅色页背景
const PRIMARY = "14617C";          // 主色
const PRI_LT = "DCE9EE";           // 主色浅调
const PRI_MD = "8FB4C4";
const ACCENT = "DE7A1F";           // 强调色（关键数字/关键柱）
const TEXT = "16262E";
const MUTED = "6E8189";
const WHITE = "FFFFFF";
const FONT = "微软雅黑";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "人工智能实践课程设计";
pres.title = "基于飞桨的 CIFAR-10 图像分类识别系统";

// ── 复用工厂（pptxgenjs 会就地修改选项对象，必须每次新建）────────
const shadow = () => ({ type: "outer", color: "0D3A4C", blur: 8, offset: 3, angle: 90, opacity: 0.16 });
const bullet = () => ({ code: "25AA", indent: 12 });
const chip = () => ({ fill: { color: PRI_LT }, line: { color: PRI_MD, width: 0.75 }, rectRadius: 0.06 });

// ── 通用版式helper ───────────────────────────────────────────────
function titleSlide(s, zh, en) {
  s.addText(zh, { x: M, y: 0.45, w: CW, h: 0.62, fontSize: 31, bold: true, color: PRIMARY, fontFace: FONT, margin: 0, valign: "middle" });
  if (en) s.addText(en, { x: M, y: 1.08, w: CW, h: 0.34, fontSize: 14, color: MUTED, fontFace: FONT, margin: 0, valign: "middle" });
}

function source(s, txt) {
  s.addText(txt, { x: M, y: H - 0.62, w: CW, h: 0.3, fontSize: 11, color: MUTED, fontFace: FONT, margin: 0, valign: "middle" });
}

function statCard(s, x, y, w, h, num, unit, label, opts = {}) {
  const accent = opts.accent;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: opts.fill || WHITE }, line: { color: PRI_LT, width: 1 }, rectRadius: 0.08, shadow: opts.flat ? undefined : shadow() });
  s.addText([{ text: num, options: { fontSize: opts.numSize || 32, bold: true, color: accent ? ACCENT : PRIMARY } },
             { text: " " + (unit || ""), options: { fontSize: 13, bold: true, color: MUTED } }],
    { x: x + 0.18, y: y + 0.08, w: w - 0.36, h: h * 0.46, fontFace: FONT, margin: 0, valign: "middle" });
  s.addText(label, { x: x + 0.18, y: y + h * 0.52, w: w - 0.36, h: h * 0.44, fontSize: 12, color: TEXT, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.2 });
}

// ══════════════════════════════════════════════════════════════════
// 1 · 封面
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: DARK };

  s.addText("深度学习实践 · 方向一", { x: M, y: 1.25, w: CW, h: 0.32, fontSize: 14, color: PRI_MD, fontFace: FONT, margin: 0, charSpacing: 3 });
  s.addText("基于飞桨的\nCIFAR-10 图像分类识别系统", {
    x: M, y: 1.75, w: 8.6, h: 2.0, fontSize: 40, bold: true, color: WHITE,
    fontFace: FONT, margin: 0, lineSpacingMultiple: 1.15, valign: "top"
  });
  s.addText("ResNet18（32×32 适配版）· PaddlePaddle 3.3 · Streamlit 交互界面", {
    x: M, y: 3.95, w: 8.6, h: 0.36, fontSize: 16, color: PRI_LT, fontFace: FONT, margin: 0
  });

  // 关键数字条（本页视觉焦点之一）
  const nums = [["92.07%", "测试集准确率"], ["3.30 ms", "单张推理延迟"], ["46 项", "自动化测试"]];
  nums.forEach(([n, l], i) => {
    const x = M + i * 2.75;
    s.addText(n, { x, y: 4.85, w: 2.6, h: 0.55, fontSize: 27, bold: true, color: i === 0 ? ACCENT : WHITE, fontFace: FONT, margin: 0 });
    s.addText(l, { x, y: 5.38, w: 2.6, h: 0.3, fontSize: 12.5, color: PRI_MD, fontFace: FONT, margin: 0 });
  });

  s.addText("汇报人：__________　　学号：__________　　指导教师：__________", {
    x: M, y: 6.35, w: CW, h: 0.35, fontSize: 13, color: "9DB9C4", fontFace: FONT, margin: 0
  });
  s.addNotes("开场：一句话点题——本项目不是只写一个训练脚本，而是把『数据—模型—训练—评估—部署→文档』整条链跑通。" +
             "先说三个数字：测试集准确率 92.07%、单张推理 3.30 毫秒、46 项自动化测试全部通过。");
}

// ══════════════════════════════════════════════════════════════════
// 2 · 背景与目标
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  titleSlide(s, "图像分类是 CV 的基础任务，CIFAR-10 是最合适的练兵场", "项目背景与目标");

  s.addText([
    { text: "CIFAR-10：", options: { bold: true, color: PRIMARY } },
    { text: "60000 张 32×32 彩色图、10 类物体。分辨率低、类间差异小（猫/狗、汽车/卡车），", options: {} },
    { text: "既能完整体验全流程，又能在普通 GPU 上快速实验。", options: {} }
  ], { x: M, y: 1.62, w: CW, h: 0.62, fontSize: 15.5, color: TEXT, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.25 });

  const goals = [
    ["01", "精度达标", "测试集 Top-1 准确率 ≥ 90%，并给出逐类精确率/召回率/F1"],
    ["02", "可交互", "网页上传图片即得 Top-3 类别与置信度，异常输入有中文提示"],
    ["03", "可交付", "模块化结构 + 接口文档 + 单元测试 + 三份文档，能交给别人运行"],
  ];
  goals.forEach(([no, t, d], i) => {
    const x = M + i * (CW / 3 + 0.02), w = CW / 3 - 0.28;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.62, w, h: 2.5, fill: { color: WHITE }, line: { color: PRI_LT, width: 1 }, rectRadius: 0.08, shadow: shadow() });
    s.addText(no, { x: x + 0.28, y: 2.86, w: 1.2, h: 0.7, fontSize: 38, bold: true, color: PRI_LT, fontFace: FONT, margin: 0 });
    s.addText(t, { x: x + 0.28, y: 3.58, w: w - 0.56, h: 0.36, fontSize: 18, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
    s.addText(d, { x: x + 0.28, y: 3.98, w: w - 0.56, h: 0.95, fontSize: 13.5, color: TEXT, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.3 });
  });

  s.addText("技术路线：paddle.vision 读取数据 → paddle.nn 搭建 ResNet18 → 训练调优 → 多测试集评估 → Streamlit 部署",
    { x: M, y: 5.62, w: CW, h: 0.42, fontSize: 13, color: MUTED, fontFace: FONT, margin: 0, valign: "middle" });
  source(s, "数据来源：CIFAR-10 官方数据集（Krizhevsky, 2009）；参考教材：邱锡鹏《神经网络与深度学习》5.5 节");
  s.addNotes("背景讲一句就够：CIFAR-10 是入门首选，因为它小到能在笔记本 GPU 上练手，但难到能暴露真问题。" +
             "三个目标分别对应课程要求里的『精度』『界面』『可交付』。");
}

// ══════════════════════════════════════════════════════════════════
// 3 · 系统架构
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  titleSlide(s, "四层架构：模块各司其职，接口清晰可联调", "系统设计");

  const layers = [
    ["用户交互层", "app.py（Streamlit 网页）　·　predict.py（命令行推理）", PRIMARY, WHITE],
    ["业务逻辑层", "train.py（训练）　·　eval.py（评估 + 鲁棒性测试 + 耗时分析）", "1B7A99", WHITE],
    ["核心组件层", "models/resnet.py　·　data/dataset.py　·　data/preprocess.py　·　utils/（日志·指标·IO）", "5E9FB5", WHITE],
    ["配置与产物", "config.py（全部超参数与路径）　·　outputs/（模型·日志·评估结果）", PRI_LT, TEXT],
  ];
  const lh = 0.82, gap = 0.2;
  layers.forEach(([name, detail, bg, fg], i) => {
    const y = 1.72 + i * (lh + gap);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y, w: CW - 3.5, h: lh, fill: { color: bg }, line: { color: bg, width: 1 }, rectRadius: 0.06 });
    s.addText(name, { x: M + 0.28, y: y + 0.06, w: 1.9, h: lh - 0.12, fontSize: 16, bold: true, color: fg, fontFace: FONT, margin: 0, valign: "middle" });
    s.addText(detail, { x: M + 2.25, y: y + 0.06, w: CW - 5.9, h: lh - 0.12, fontSize: 12.5, color: fg, fontFace: FONT, margin: 0, valign: "middle", lineSpacingMultiple: 1.2 });
    if (i < layers.length - 1) {
      s.addShape(pres.shapes.LINE, { x: M + 1.1, y: y + lh, w: 0, h: gap, line: { color: PRI_MD, width: 1.25, endArrowType: "triangle" } });
    }
  });

  // 右侧：数据流
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: W - M - 3.2, y: 1.72, w: 3.2, h: 3.3, fill: { color: WHITE }, line: { color: PRI_LT, width: 1 }, rectRadius: 0.08, shadow: shadow() });
  s.addText("一次推理的数据流", { x: W - M - 2.94, y: 1.92, w: 2.7, h: 0.32, fontSize: 14, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
  s.addText([
    { text: "图片上传", options: { breakLine: true } },
    { text: "统一转 RGB（任意格式/位深）", options: { breakLine: true } },
    { text: "缩放到 32×32", options: { breakLine: true } },
    { text: "float32 归一化（CIFAR-10 均值方差）", options: { breakLine: true } },
    { text: "ResNet18 前向 → softmax", options: { breakLine: true } },
    { text: "Top-3 类别 + 置信度", options: {} },
  ], { x: W - M - 2.94, y: 2.36, w: 2.7, h: 2.5, fontSize: 12, color: TEXT, fontFace: FONT, margin: 0, valign: "top", paraSpaceAfter: 7, bullet: bullet() });

  source(s, "调用关系详见 docs/technical_doc.md 第 2 节（6 张接口表）与第 8 节（联调说明）");
  s.addNotes("架构的重点不是分层好看，而是『共用核心』：命令行与网页共用同一个推理函数，" +
             "评估与训练共用同一套指标和模型加载逻辑，所以行为一致、不会两边跑出不同结果。");
}

// ══════════════════════════════════════════════════════════════════
// 4 · 数据组织与预处理
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  titleSlide(s, "60000 张图：固定种子划分，两种增强抑制过拟合", "数据集组织与管理");

  // 左：划分流程
  s.addText("① 数据划分（固定随机种子 SEED=42，任何机器结果一致）", { x: M, y: 1.58, w: 6.4, h: 0.32, fontSize: 15, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });

  const b1y = 2.2;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: b1y, w: 2.5, h: 1.75, fill: { color: PRIMARY }, line: { color: PRIMARY }, rectRadius: 0.06 });
  s.addText([{ text: "50000", options: { fontSize: 26, bold: true, color: WHITE, breakLine: true } }, { text: "官方训练集", options: { fontSize: 12, color: PRI_LT } }],
    { x: M, y: b1y + 0.25, w: 2.5, h: 1.3, fontFace: FONT, margin: 0, align: "center", valign: "middle" });

  s.addShape(pres.shapes.LINE, { x: M + 2.5, y: b1y + 0.87, w: 0.42, h: 0, line: { color: PRI_MD, width: 1.75, endArrowType: "triangle" } });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 2.92, y: b1y, w: 3.5, h: 0.8, fill: { color: WHITE }, line: { color: PRI_MD, width: 1 }, rectRadius: 0.06 });
  s.addText([{ text: "45000 ", options: { fontSize: 17, bold: true, color: PRIMARY } }, { text: "训练集（含增强）", options: { fontSize: 12.5, color: TEXT } }],
    { x: M + 3.06, y: b1y + 0.04, w: 3.24, h: 0.76, fontFace: FONT, margin: 0, valign: "middle" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 2.92, y: b1y + 0.95, w: 3.5, h: 0.8, fill: { color: WHITE }, line: { color: PRI_MD, width: 1 }, rectRadius: 0.06 });
  s.addText([{ text: "5000 ", options: { fontSize: 17, bold: true, color: PRIMARY } }, { text: "验证集（仅标准化）", options: { fontSize: 12.5, color: TEXT } }],
    { x: M + 3.06, y: b1y + 0.97, w: 3.24, h: 0.76, fontFace: FONT, margin: 0, valign: "middle" });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: b1y + 1.98, w: 6.42, h: 0.82, fill: { color: PRI_LT }, line: { color: PRI_LT }, rectRadius: 0.06 });
  s.addText([{ text: "10000 ", options: { fontSize: 17, bold: true, color: PRIMARY } }, { text: "官方测试集 —— 训练全程从未接触，只在最终评估时使用", options: { fontSize: 12.5, color: TEXT } }],
    { x: M + 0.14, y: b1y + 2.0, w: 6.14, h: 0.78, fontFace: FONT, margin: 0, valign: "middle" });

  // 右：增强
  s.addText("② 训练数据增强（每张图随机变换，扩充样本多样性）", { x: M + 6.9, y: 1.58, w: 5.2, h: 0.32, fontSize: 15, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
  const augs = [
    ["随机裁剪", "四周补 4 像素后随机裁回 32×32　→　提升平移不变性"],
    ["随机水平翻转", "50% 概率左右镜像　→　模拟镜像视角（语义不变）"],
    ["标准化", "按 CIFAR-10 均值/方差归一化，与教材 5.5 节一致"],
  ];
  augs.forEach(([t, d], i) => {
    const y = 2.2 + i * 1.15;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 6.9, y, w: 5.22, h: 1.0, fill: { color: WHITE }, line: { color: PRI_LT, width: 1 }, rectRadius: 0.06, shadow: shadow() });
    s.addText(t, { x: M + 7.08, y: y + 0.07, w: 4.9, h: 0.32, fontSize: 14, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
    s.addText(d, { x: M + 7.08, y: y + 0.46, w: 4.9, h: 0.5, fontSize: 12, color: TEXT, fontFace: FONT, margin: 0, valign: "top" });
  });

  source(s, "工程保障：数据首次运行自动下载 + MD5 校验 + 3 次重试 + 原子替换，损坏自动重下（data/dataset.py）");
  s.addNotes("这里要强调两点：一是划分用固定种子，所以助教在自己机器上复现得到的数据划分完全一样；" +
             "二是增强只作用于训练集，验证/测试/推理都只做标准化，保证评估公平。");
}

// ══════════════════════════════════════════════════════════════════
// 5 · 模型方案
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  titleSlide(s, "ResNet18：为 32×32 小图做三处关键适配", "模型选型与搭建");

  // 左：残差块结构示意（跳跃连接从左侧绕行，不穿过卷积框）
  s.addText("残差块 BasicBlock（4 个阶段，每阶段 2 个块）", { x: M, y: 1.58, w: 5.9, h: 0.3, fontSize: 14, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });

  const bx = M + 0.45, bw = 2.6, bh = 0.52, cx = bx + bw / 2;
  [["Conv3×3 + BN + ReLU", 2.06], ["Conv3×3 + BN", 2.82]].forEach(([t, y]) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: bx, y, w: bw, h: bh, fill: { color: WHITE }, line: { color: PRIMARY, width: 1.25 }, rectRadius: 0.06 });
    s.addText(t, { x: bx, y: y + 0.02, w: bw, h: bh - 0.04, fontSize: 12.5, color: TEXT, fontFace: FONT, margin: 0, align: "center", valign: "middle" });
  });
  s.addShape(pres.shapes.LINE, { x: cx, y: 2.58, w: 0, h: 0.24, line: { color: PRIMARY, width: 1.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.LINE, { x: cx, y: 3.34, w: 0, h: 0.26, line: { color: PRIMARY, width: 1.5 } });
  // 相加节点
  s.addShape(pres.shapes.OVAL, { x: cx - 0.13, y: 3.60, w: 0.26, h: 0.26, fill: { color: ACCENT }, line: { color: ACCENT } });
  s.addText("+", { x: cx - 0.13, y: 3.60, w: 0.26, h: 0.26, fontSize: 13, bold: true, color: WHITE, fontFace: FONT, margin: 0, align: "center", valign: "middle" });
  // 跳跃连接：从第一个卷积框左侧绕行（OOXML 的 ext cx 必须非负，故一律从左向右绘制）
  s.addShape(pres.shapes.LINE, { x: 0.70, y: 2.32, w: bx - 0.70, h: 0, line: { color: ACCENT, width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: 0.70, y: 2.32, w: 0, h: 1.41, line: { color: ACCENT, width: 1.5 } });
  s.addShape(pres.shapes.LINE, { x: 0.70, y: 3.73, w: cx - 0.83, h: 0, line: { color: ACCENT, width: 1.5, endArrowType: "triangle" } });
  s.addText([
    { text: "跳跃连接", options: { bold: true, color: ACCENT, breakLine: true } },
    { text: "恒等映射或 1×1 卷积", options: { color: TEXT } },
  ], { x: bx + bw + 0.3, y: 2.42, w: 2.6, h: 0.9, fontSize: 12, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.3 });
  s.addText("梯度可「抄近路」直接回传，避免深层网络退化 → 这是 ResNet 能稳定收敛的关键（教材 5.4 节）",
    { x: M, y: 4.02, w: 5.9, h: 0.9, fontSize: 12.5, color: TEXT, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.35 });

  // 右：三处适配
  s.addText("相对 ImageNet 版原结构的三处改造", { x: M + 6.3, y: 1.6, w: 5.8, h: 0.3, fontSize: 14, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });

  const rows = [
    ["首层卷积", "7×7, stride 2", "3×3, stride 1", "32×32 太小，大核大步长会过早丢信息"],
    ["首个池化", "MaxPool 3×3 s2", "去掉", "否则特征图到第 2 阶段就只剩 1×1"],
    ["末端池化", "AdaptiveAvgPool", "保留", "对特征图尺寸不敏感，适配小图"],
  ];
  const ty = 2.02, rh = 0.56;
  s.addShape(pres.shapes.RECTANGLE, { x: M + 6.3, y: ty, w: 5.82, h: rh, fill: { color: PRIMARY }, line: { color: PRIMARY } });
  [["改造项", 0.02, 1.1], ["ImageNet 版", 1.14, 1.5], ["本项目", 2.68, 1.3], ["原因", 4.02, 1.78]].forEach(([t, dx, dw]) => {
    s.addText(t, { x: M + 6.3 + dx + 0.1, y: ty, w: dw - 0.14, h: rh, fontSize: 12.5, bold: true, color: WHITE, fontFace: FONT, margin: 0, valign: "middle" });
  });
  rows.forEach(([a, b, c, d], i) => {
    const y = ty + rh + i * rh;
    s.addShape(pres.shapes.RECTANGLE, { x: M + 6.3, y, w: 5.82, h: rh, fill: { color: i % 2 ? WHITE : "EDF3F5" }, line: { color: PRI_LT, width: 0.75 } });
    s.addText(a, { x: M + 6.42, y, w: 1.0, h: rh, fontSize: 12, bold: true, color: TEXT, fontFace: FONT, margin: 0, valign: "middle" });
    s.addText(b, { x: M + 7.44, y, w: 1.46, h: rh, fontSize: 11, color: MUTED, fontFace: FONT, margin: 0, valign: "middle" });
    s.addText(c, { x: M + 8.98, y, w: 1.24, h: rh, fontSize: 11.5, bold: true, color: ACCENT, fontFace: FONT, margin: 0, valign: "middle" });
    s.addText(d, { x: M + 10.32, y, w: 1.76, h: rh, fontSize: 10.5, color: TEXT, fontFace: FONT, margin: 0, valign: "middle", lineSpacingMultiple: 1.15 });
  });

  // 结构输出尺寸（通道数须与阶段严格对应：64→64→128→256→512）
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 6.3, y: ty + rh * 4 + 0.16, w: 5.82, h: 0.68, fill: { color: PRI_LT }, line: { color: PRI_LT }, rectRadius: 0.06 });
  s.addText("特征图（通道×高×宽）：64×32×32 → 64×32×32 → 128×16×16 → 256×8×8 → 512×4×4 → 池化 → FC(10)",
    { x: M + 6.44, y: ty + rh * 4 + 0.18, w: 5.54, h: 0.64, fontSize: 11.5, color: TEXT, fontFace: FONT, margin: 0, valign: "middle", lineSpacingMultiple: 1.2 });

  statCard(s, M + 6.3, 5.3, 2.8, 1.18, "11.17", "M", "可训练参数量（权重 43 MB）", { numSize: 26 });
  statCard(s, M + 9.32, 5.3, 2.8, 1.18, "93%", "", "该结构公开最好水平 93~94%", { numSize: 26 });

  source(s, "模型用 paddle.nn 原生算子搭建，未使用高层封装，便于逐层验证（models/resnet.py）");
  s.addNotes("这一页是技术含量最高的部分，重点讲『为什么改』：原版 ResNet18 是给 224×224 设计的，" +
             "直接搬到 32×32 上，特征图会在第二个阶段就被压成 1×1 而无法继续。所以首层改成 3×3 小步长、去掉首个池化。");
}

// ══════════════════════════════════════════════════════════════════
// 6 · 训练策略与过程
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  titleSlide(s, "交叉熵 + Adam + 阶梯衰减 + 早停，40 轮自动收敛", "模型训练");

  s.addImage({ path: IMG("outputs/results/train_curves.png"), x: M, y: 1.6, w: 8.2, h: 4.28 });

  const items = [
    ["损失函数", "F.cross_entropy（多分类标准选择）"],
    ["优化器", "Adam，lr=1e-3，weight_decay=5e-4"],
    ["学习率衰减", "StepDecay：每 20 轮 ×0.1　→　第 20 轮出现台阶式跳升"],
    ["早停", "验证准确率连续 10 轮不提升即停 → 第 57 轮触发"],
    ["最佳模型", "验证准确率创新高才保存 → 取自第 47 轮"],
  ];
  items.forEach(([k, v], i) => {
    const y = 1.68 + i * 0.88;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 8.5, y, w: 4.22, h: 0.76, fill: { color: WHITE }, line: { color: PRI_LT, width: 1 }, rectRadius: 0.06, shadow: shadow() });
    s.addText(k, { x: M + 8.66, y: y + 0.06, w: 4.0, h: 0.28, fontSize: 13, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
    s.addText(v, { x: M + 8.66, y: y + 0.34, w: 4.0, h: 0.38, fontSize: 11, color: TEXT, fontFace: FONT, margin: 0, valign: "top" });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 8.5, y: 6.1, w: 4.22, h: 0.62, fill: { color: PRIMARY }, line: { color: PRIMARY }, rectRadius: 0.06 });
  s.addText("最佳验证准确率 92.88%（第 47 轮）", { x: M + 8.5, y: 6.12, w: 4.22, h: 0.58, fontSize: 13.5, bold: true, color: WHITE, fontFace: FONT, margin: 0, align: "center", valign: "middle" });

  source(s, "训练环境：NVIDIA RTX 5060 Laptop GPU，约 1060 img/s，全程约 40 分钟（含早停）");
  s.addNotes("看曲线讲三件事：① 第 20 轮学习率降 10 倍，验证准确率台阶式跳升；" +
             "② 第 25 轮前后验证损失见底回升而训练损失仍在降——典型过拟合信号；③ 早停在第 57 轮介入，" +
             "最佳模型取自第 47 轮，这就是『保存最佳而非最后一轮』的价值。");
}

// ══════════════════════════════════════════════════════════════════
// 7 · 测试结果总览
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  titleSlide(s, "测试集 10000 张：准确率 92.07%，各类表现均衡", "评估结果");

  statCard(s, M, 1.6, 2.9, 1.5, "92.07", "%", "测试集 Top-1 准确率（9207/10000）", { numSize: 32, accent: true });
  statCard(s, M + 3.05, 1.6, 2.9, 1.5, "0.9205", "", "宏平均 F1（精确率/召回率同为 0.917x）", { numSize: 28 });
  statCard(s, M + 6.1, 1.6, 2.9, 1.5, "0.2838", "", "测试集平均交叉熵损失", { numSize: 28 });

  s.addText("逐类 F1（最高 / 最低）", { x: M + 9.35, y: 1.62, w: 2.78, h: 0.32, fontSize: 13, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
  s.addText([{ text: "最高　", options: { color: MUTED, fontSize: 11.5 } }, { text: "汽车 0.959\n", options: { bold: true, color: PRIMARY, fontSize: 15 } },
             { text: "最低　", options: { color: MUTED, fontSize: 11.5 } }, { text: "猫 0.828", options: { bold: true, color: ACCENT, fontSize: 15 } }],
    { x: M + 9.35, y: 1.98, w: 2.78, h: 1.0, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.35 });

  // 逐类 F1 柱状图（真实数据）
  s.addChart(pres.charts.BAR, [{
    name: "F1", labels: ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
    values: [0.9160, 0.9591, 0.8971, 0.8276, 0.9235, 0.8737, 0.9387, 0.9472, 0.9474, 0.9438],
  }], {
    x: M, y: 3.32, w: CW, h: 3.05, barDir: "col",
    chartColors: ["14617C", "14617C", "14617C", "DE7A1F", "14617C", "14617C", "14617C", "14617C", "14617C", "14617C"],
    varyColors: true,
    chartArea: { fill: { color: LIGHT } },
    catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    valGridLine: { color: "D8E3E8", size: 0.5 }, catGridLine: { style: "none" },
    valAxisMinVal: 0.75, valAxisMaxVal: 1.0,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT, dataLabelFontFace: FONT, dataLabelFontSize: 10.5, dataLabelFormatCode: "0.000",
    showLegend: false, barGapWidthPct: 45,
  });

  source(s, "数据来源：outputs/results/classification_report.json（本项目实测，RTX 5060 GPU）");
  s.addNotes("92.07% 接近该结构的公开最好水平（93~94%）。逐类看，猫最低（0.835）——因为猫的姿态毛色变化大，" +
             "且与狗高度相似；汽车最高（0.959）。这类类别不平衡的结论比一个总准确率更有信息量。");
}

// ══════════════════════════════════════════════════════════════════
// 8 · 混淆矩阵
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  titleSlide(s, "错误集中在语义相近的类别之间", "混淆矩阵分析");

  s.addImage({ path: IMG("outputs/results/confusion_matrix.png"), x: M, y: 1.6, w: 7.6, h: 5.0 });

  s.addText("最易混淆的类别对 Top 5", { x: M + 8.0, y: 1.68, w: 4.72, h: 0.32, fontSize: 14, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
  const pairs = [["狗 → 猫", "92 次"], ["猫 → 狗", "63 次"], ["鸟 → 飞机", "33 次"], ["汽车 → 卡车", "32 次"], ["猫 → 鹿", "27 次"]];
  pairs.forEach(([a, b], i) => {
    const y = 2.1 + i * 0.58;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 8.0, y, w: 4.72, h: 0.48, fill: { color: i < 2 ? "FBEBDC" : WHITE }, line: { color: i < 2 ? "F0C9A0" : PRI_LT, width: 1 }, rectRadius: 0.06 });
    s.addText(a, { x: M + 8.2, y, w: 2.4, h: 0.48, fontSize: 13, bold: i < 2, color: i < 2 ? ACCENT : TEXT, fontFace: FONT, margin: 0, valign: "middle" });
    s.addText(b, { x: M + 10.6, y, w: 2.0, h: 0.48, fontSize: 13, bold: true, color: MUTED, fontFace: FONT, margin: 0, valign: "middle", align: "right" });
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 8.0, y: 5.18, w: 4.72, h: 1.42, fill: { color: WHITE }, line: { color: PRI_LT, width: 1 }, rectRadius: 0.08, shadow: shadow() });
  s.addText("结论与改进方向", { x: M + 8.18, y: 5.3, w: 4.36, h: 0.28, fontSize: 12.5, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
  s.addText("猫狗双向互混占两者错误的大头，模型抓住了「四足 + 毛茸茸」的共同特征但难以细分。" +
            "可引入 Mixup/CutMix、标签平滑或升级 ResNet34 改善。",
    { x: M + 8.18, y: 5.58, w: 4.36, h: 0.94, fontSize: 11.5, color: TEXT, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.25 });

  source(s, "对角线为各类召回率；完整矩阵见 outputs/results/confusion_matrix.png 与 .csv");
  s.addNotes("混淆矩阵是最能说明『模型学到了什么』的工具。看对角线：汽车 0.95、船 0.95 最好；" +
             "看非对角：猫狗互混最严重。这说明错误不是随机的，而是集中在语义相近的类别上——与人类直觉一致。");
}

// ══════════════════════════════════════════════════════════════════
// 9 · 鲁棒性测试
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  titleSlide(s, "5 种扰动测试：平移鲁棒性强，噪声/低分辨率是短板", "多类型测试数据集");

  s.addChart(pres.charts.BAR, [{
    name: "准确率", labels: ["标准测试集", "裁剪平移 ±4px", "中心裁剪 80%", "旋转 ±20°", "高斯噪声 σ=0.08", "低分辨率 12×12"],
    values: [92.07, 91.85, 90.30, 85.25, 32.55, 22.60],
  }], {
    x: M, y: 1.66, w: 8.5, h: 4.85, barDir: "bar",
    chartColors: ["14617C", "14617C", "14617C", "14617C", "DE7A1F", "DE7A1F"],
    varyColors: true,
    chartArea: { fill: { color: LIGHT } },
    catAxisLabelColor: TEXT, valAxisLabelColor: MUTED, catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT, catAxisLabelFontSize: 12.5, valAxisLabelFontSize: 11,
    valGridLine: { color: "D8E3E8", size: 0.5 }, catGridLine: { style: "none" },
    valAxisMinVal: 0, valAxisMaxVal: 100,
    showValue: true, dataLabelPosition: "outEnd", dataLabelColor: TEXT, dataLabelFontFace: FONT, dataLabelFontSize: 11, dataLabelFormatCode: "0.00",
    showLegend: false, barGapWidthPct: 40,
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 8.8, y: 1.72, w: 3.92, h: 2.0, fill: { color: WHITE }, line: { color: PRI_LT, width: 1 }, rectRadius: 0.08, shadow: shadow() });
  s.addText("数据增强的收益是可量化的", { x: M + 8.98, y: 1.86, w: 3.56, h: 0.3, fontSize: 13, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
  s.addText("裁剪平移仅掉 0.20 个百分点——因为训练时用了随机裁剪，" +
            "平移不变性被「练」出来了。这是增强有效性的直接证据。",
    { x: M + 8.98, y: 2.2, w: 3.56, h: 1.45, fontSize: 12, color: TEXT, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.35 });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 8.8, y: 4.16, w: 3.92, h: 2.35, fill: { color: "FBEBDC" }, line: { color: "F0C9A0", width: 1 }, rectRadius: 0.08 });
  s.addText("短板与原因", { x: M + 8.98, y: 4.3, w: 3.56, h: 0.3, fontSize: 13, bold: true, color: ACCENT, fontFace: FONT, margin: 0 });
  s.addText("噪声与低分辨率下降剧烈（−55 / −69 个百分点），根本原因是训练分布中不含这类退化，" +
            "且 32×32 小图信噪比本就低。改进：训练增强中加入噪声与 RandomRotation。",
    { x: M + 8.98, y: 4.64, w: 3.56, h: 1.75, fontSize: 12, color: TEXT, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.35 });

  source(s, "每项 2000 张，固定随机种子保证各扰动使用同一批样本；完整对比表见 outputs/results/comparison_table.md");
  s.addNotes("这一页是全场最有分析价值的部分。不要只报数字，要讲『为什么』：" +
             "裁剪平移几乎不掉分，证明数据增强真的起作用了；噪声和低分辨率掉得多，是因为训练数据里根本没有这类退化，" +
             "属于典型的分布外问题。这就把『做了测试』升级成了『发现了问题并指出改进方向』。");
}

// ══════════════════════════════════════════════════════════════════
// 10 · 推理性能
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  titleSlide(s, "单张 3.30 ms，批量吞吐 7073 img/s", "推理性能分析");

  statCard(s, M, 1.7, 3.0, 1.5, "3.30", "ms", "单张推理延迟（batch=1，含预处理）", { numSize: 32, accent: true });
  statCard(s, M + 3.25, 1.7, 3.0, 1.5, "7073", "img/s", "批量吞吐（batch=128）", { numSize: 32 });
  statCard(s, M + 6.5, 1.7, 3.0, 1.5, "43", "MB", "模型权重文件大小", { numSize: 32 });

  // 表格（右侧留出与全篇一致的 0.6" 页边距）
  const tx = M + 9.7, tw = W - M - tx, rh = 0.62;
  s.addShape(pres.shapes.RECTANGLE, { x: tx, y: 1.7, w: tw, h: 0.48, fill: { color: PRIMARY }, line: { color: PRIMARY } });
  s.addText("batch", { x: tx + 0.1, y: 1.7, w: 0.75, h: 0.48, fontSize: 12, bold: true, color: WHITE, fontFace: FONT, margin: 0, valign: "middle" });
  s.addText("吞吐 img/s", { x: tx + 0.82, y: 1.7, w: tw - 0.92, h: 0.48, fontSize: 12, bold: true, color: WHITE, fontFace: FONT, margin: 0, valign: "middle", align: "right" });
  [["1", "303"], ["64", "5351"], ["128", "7073"]].forEach(([b, t], i) => {
    const y = 2.18 + i * 0.5;
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y, w: tw, h: 0.5, fill: { color: i % 2 ? "EDF3F5" : WHITE }, line: { color: PRI_LT, width: 0.75 } });
    s.addText(b, { x: tx + 0.1, y, w: 0.75, h: 0.5, fontSize: 12, color: TEXT, fontFace: FONT, margin: 0, valign: "middle" });
    s.addText(t, { x: tx + 0.82, y, w: tw - 0.92, h: 0.5, fontSize: 12, bold: true, color: PRIMARY, fontFace: FONT, margin: 0, valign: "middle", align: "right" });
  });

  // 结论
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 3.72, w: CW, h: 2.05, fill: { color: WHITE }, line: { color: PRI_LT, width: 1 }, rectRadius: 0.08, shadow: shadow() });
  s.addText("性能解读", { x: M + 0.3, y: 3.9, w: 3.0, h: 0.32, fontSize: 14, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
  s.addText([
    { text: "小批量场景 GPU 利用率不足：", options: { bold: true, color: TEXT } },
    { text: "batch 从 1 增到 128，吞吐量提升约 23 倍（303 → 7073 img/s），说明大批量能显著摊薄单张开销。", options: { color: TEXT, breakLine: true } },
    { text: "部署建议：", options: { bold: true, color: TEXT } },
    { text: "生产环境应凑批推理而非逐张调用；界面单张 3.30 ms 已远快于人眼感知，无优化必要。", options: { color: TEXT, breakLine: true } },
    { text: "界面响应优化：", options: { bold: true, color: TEXT } },
    { text: "模型加载（43 MB）用 st.cache_resource 缓存，会话内只加载一次；推理本身不再缓存以保证每次上传都实时计算。", options: { color: TEXT } },
  ], { x: M + 0.3, y: 4.28, w: CW - 0.6, h: 1.38, fontSize: 13, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.4, paraSpaceAfter: 5 });

  source(s, "测量方法：预热 20 次后统计 100 次平均，排除 CUDA 初始化等一次性开销（eval.py::benchmark_inference）");
  s.addNotes("性能这页要讲两层：一是数字本身（3.30 ms 很快）；二是从数字看出的工程结论——" +
             "batch 增大吞吐提升 16 倍说明小批量浪费了 GPU，所以服务化部署应该凑批。这是从数据推出决策。");
}

// ══════════════════════════════════════════════════════════════════
// 11 · 界面与容错
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  titleSlide(s, "上传图片即得 Top-3 类别与置信度，异常输入不崩溃", "用户交互界面与容错性");

  s.addText("以下为系统实际识别结果（模型：第 47 轮，验证准确率 92.88%）", { x: M, y: 1.56, w: 7.8, h: 0.3, fontSize: 12.5, color: MUTED, fontFace: FONT, margin: 0 });

  const samples = [["test_imgs/0_cat.png", "猫 cat", "99.99%"], ["test_imgs/1_ship.png", "船 ship", "100.00%"], ["test_imgs/9702_horse.png", "马 horse", "100.00%"]];
  samples.forEach(([p, label, conf], i) => {
    const x = M + i * 2.62;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.94, w: 2.36, h: 3.3, fill: { color: WHITE }, line: { color: PRI_LT, width: 1 }, rectRadius: 0.08, shadow: shadow() });
    s.addImage({ path: IMG(p), x: x + 0.28, y: 2.14, w: 1.8, h: 1.8 });
    s.addText(label, { x: x + 0.14, y: 4.04, w: 2.08, h: 0.34, fontSize: 15, bold: true, color: PRIMARY, fontFace: FONT, margin: 0, align: "center" });
    s.addText("置信度 " + conf, { x: x + 0.14, y: 4.4, w: 2.08, h: 0.3, fontSize: 12, color: ACCENT, fontFace: FONT, margin: 0, align: "center" });
    s.addText(i === 2 ? "第 2 名 卡车 5.05%" : (i === 1 ? "第 2 名 汽车 0.00%" : "第 2 名 狗 0.01%"),
      { x: x + 0.14, y: 4.74, w: 2.08, h: 0.3, fontSize: 10.5, color: MUTED, fontFace: FONT, margin: 0, align: "center" });
  });

  // 右：功能与容错
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 8.1, y: 1.94, w: 4.62, h: 3.42, fill: { color: WHITE }, line: { color: PRI_LT, width: 1 }, rectRadius: 0.08, shadow: shadow() });
  s.addText("界面功能", { x: M + 8.32, y: 2.1, w: 4.2, h: 0.3, fontSize: 13.5, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
  s.addText([
    { text: "拖拽/点击上传，支持 jpg·png·bmp·webp", options: { breakLine: true } },
    { text: "Top-3 类别表格 + 置信度柱状图", options: { breakLine: true } },
    { text: "可展开查看全部 10 类概率分布", options: { breakLine: true } },
    { text: "显示推理耗时、原图尺寸、运行设备", options: {} },
  ], { x: M + 8.32, y: 2.44, w: 4.2, h: 1.3, fontSize: 12, color: TEXT, fontFace: FONT, margin: 0, valign: "top", paraSpaceAfter: 6, bullet: bullet() });

  s.addText("容错设计（异常均转为中文提示）", { x: M + 8.32, y: 3.8, w: 4.2, h: 0.3, fontSize: 13.5, bold: true, color: PRIMARY, fontFace: FONT, margin: 0 });
  s.addText([
    { text: "模型缺失/为空/结构不匹配 → 分别给出提示", options: { breakLine: true } },
    { text: "图片路径、格式、损坏三类校验", options: { breakLine: true } },
    { text: "各种图片模式与位深统一转 RGB", options: { breakLine: true } },
    { text: "未预期异常兜底捕获，堆栈写入日志", options: {} },
  ], { x: M + 8.32, y: 4.14, w: 4.2, h: 1.16, fontSize: 12, color: TEXT, fontFace: FONT, margin: 0, valign: "top", paraSpaceAfter: 5, bullet: bullet() });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.52, w: CW, h: 1.12, fill: { color: PRI_LT }, line: { color: PRI_LT }, rectRadius: 0.08 });
  s.addText("模型文件管理：训练会覆盖 best_model.pdparams，因此训练脚本在首次覆盖前自动备份旧模型，" +
            "并每轮另存 last_model.pdparams；启动时打印覆盖警告，避免误把训练好的模型覆盖成半成品。",
    { x: M + 0.28, y: 5.66, w: CW - 0.56, h: 0.86, fontSize: 12.5, color: TEXT, fontFace: FONT, margin: 0, valign: "middle", lineSpacingMultiple: 1.3 });

  source(s, "界面代码 app.py（Streamlit）；异常与修复经验记录于 docs/technical_doc.md 第 6 节（16 条）");
  s.addNotes("界面这页建议现场演示 30 秒：上传一张图，指出 Top-3、置信度柱状图和全类别概率。 " +
             "容错部分挑一条讲：模型缺失时会明确提示『请先运行 python train.py』而不是抛一堆英文堆栈。");
}

// ══════════════════════════════════════════════════════════════════
// 12 · 真实故障复盘
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: LIGHT };
  titleSlide(s, "开发中定位并修复的两个真实故障", "错误排查记录（部分）");

  const cases = [
    {
      tag: "故障一", t: "界面任何图片都报 uint8 错误",
      rows: [["现象", "命令行识别正常，但网页上传一律失败，报 in assign must be ... but received uint8"],
             ["定位", "拿到完整堆栈：ToTensor 内部把 uint8 像素数组交给 paddle.to_tensor；而飞桨在静态图模式下会走 assign() 分支拒绝 uint8（动态图模式则不会，所以 CLI 正常）"],
             ["修复", "自实现 image_to_tensor 完成缩放与归一化，完全绕开 ToTensor；并在入口与每次推理前自动把模式切回动态图"]],
    },
    {
      tag: "故障二", t: "第二次上传仍显示第一次的结果",
      rows: [["现象", "首张图识别正确，之后上传任何图片都返回同一结果"],
             ["定位", "推理函数被 @st.cache_data 装饰，而 Streamlit 会忽略名称以下划线开头的参数——图片参数被命名为 _img_bytes，缓存键退化成只有 (模型路径, topk) 恒定不变"],
             ["修复", "移除该缓存（推理仅 3 ms，无需缓存；模型加载仍由 cache_resource 缓存）；新增界面端到端测试，连续上传 3 张图断言结果各不相同"]],
    },
  ];

  cases.forEach((c, i) => {
    const x = M + i * (CW / 2 + 0.05), w = CW / 2 - 0.28;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.66, w, h: 4.7, fill: { color: WHITE }, line: { color: PRI_LT, width: 1 }, rectRadius: 0.08, shadow: shadow() });
    s.addText(c.tag, { x: x + 0.28, y: 1.82, w: 1.0, h: 0.3, fontSize: 12, bold: true, color: ACCENT, fontFace: FONT, margin: 0, charSpacing: 2 });
    s.addText(c.t, { x: x + 0.28, y: 2.14, w: w - 0.56, h: 0.6, fontSize: 16, bold: true, color: PRIMARY, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.2 });
    c.rows.forEach(([k, v], j) => {
      const y = 2.86 + j * 1.16;
      s.addShape(pres.shapes.LINE, { x: x + 0.28, y: y - 0.1, w: w - 0.56, h: 0, line: { color: "E2EAEE", width: 1 } });
      s.addText(k, { x: x + 0.28, y, w: 0.72, h: 0.3, fontSize: 12, bold: true, color: MUTED, fontFace: FONT, margin: 0 });
      s.addText(v, { x: x + 1.02, y, w: w - 1.3, h: 1.0, fontSize: 11.5, color: TEXT, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.3 });
    });
  });

  source(s, "另记录 13 例（路径、维度、显存、模型加载、中文乱码等），完整清单见 docs/technical_doc.md 第 6 节");
  s.addNotes("这一页最能体现工程能力，建议重点讲。两个故障的共同点：命令行正常、只有特定宿主环境才暴露——" +
             "所以关键是拿到完整堆栈（我在界面错误处理里加了把堆栈写进日志），而不是靠猜。" +
             "而且两个 bug 都补了回归测试，避免复发。");
}

// ══════════════════════════════════════════════════════════════════
// 13 · 总结与展望
// ══════════════════════════════════════════════════════════════════
{
  const s = pres.addSlide();
  s.background = { color: DARK };

  s.addText("总结与展望", { x: M, y: 0.72, w: CW, h: 0.6, fontSize: 30, bold: true, color: WHITE, fontFace: FONT, margin: 0 });

  s.addText("已完成", { x: M, y: 1.6, w: 6.0, h: 0.32, fontSize: 15, bold: true, color: PRI_MD, fontFace: FONT, margin: 0, charSpacing: 2 });
  s.addText([
    { text: "全流程闭环：", options: { bold: true, color: WHITE } },
    { text: "数据自动管理 → 模型搭建 → 训练调优 → 多维评估 → 网页部署 → 三份文档", options: { color: PRI_LT, breakLine: true } },
    { text: "精度达标：", options: { bold: true, color: WHITE } },
    { text: "测试集 92.07%，宏平均 F1 0.9205，各类表现均衡", options: { color: PRI_LT, breakLine: true } },
    { text: "工程完备：", options: { bold: true, color: WHITE } },
    { text: "42 项自动化测试（含界面端到端）、模块化结构、全链路中文异常提示", options: { color: PRI_LT, breakLine: true } },
    { text: "有分析深度：", options: { bold: true, color: WHITE } },
    { text: "不仅报数字，还量化了数据增强收益、定位了噪声与低分辨率短板", options: { color: PRI_LT } },
  ], { x: M, y: 2.0, w: 6.4, h: 3.4, fontSize: 13.5, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.5, paraSpaceAfter: 8 });

  s.addText("下一步", { x: M + 7.1, y: 1.6, w: 5.0, h: 0.32, fontSize: 15, bold: true, color: PRI_MD, fontFace: FONT, margin: 0, charSpacing: 2 });
  s.addText([
    { text: "提升精度：", options: { bold: true, color: WHITE } },
    { text: "Mixup / CutMix / AutoAugment / 标签平滑，或升级 ResNet34", options: { color: PRI_LT, breakLine: true } },
    { text: "增强鲁棒性：", options: { bold: true, color: WHITE } },
    { text: "训练增强中加入高斯噪声与随机旋转，针对性补齐短板", options: { color: PRI_LT, breakLine: true } },
    { text: "工程扩展：", options: { bold: true, color: WHITE } },
    { text: "批量图片推理、导出 ONNX 跨框架部署、支持自定义数据集", options: { color: PRI_LT, breakLine: true } },
    { text: "模型集成：", options: { bold: true, color: WHITE } },
    { text: "多模型投票或测试时增强（TTA）进一步提升稳定性", options: { color: PRI_LT } },
  ], { x: M + 7.1, y: 2.0, w: 4.9, h: 3.4, fontSize: 13.5, fontFace: FONT, margin: 0, valign: "top", lineSpacingMultiple: 1.5, paraSpaceAfter: 8 });

  s.addShape(pres.shapes.LINE, { x: M + 6.75, y: 1.72, w: 0, h: 3.7, line: { color: DARK2, width: 1.25 } });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.72, w: CW, h: 1.02, fill: { color: DARK2 }, line: { color: DARK2 }, rectRadius: 0.08 });
  s.addText([{ text: "谢谢，请老师批评指正　　", options: { fontSize: 15, bold: true, color: WHITE } },
             { text: "项目代码、Notebook、技术文档与课程设计报告随附提交", options: { fontSize: 12.5, color: "9DC0CC" } }],
    { x: M + 0.4, y: 5.86, w: CW - 0.8, h: 0.74, fontFace: FONT, margin: 0, valign: "middle" });

  s.addNotes("总结一句话：这是一次完整的小型 AI 应用系统实践，不只是跑通模型。" +
             "如果老师问不足，坦诚说：准确率距该结构公开最好水平（93~94%）仍有差距，鲁棒性两项是短板，并已给出改进路径。");
}

pres.writeFile({ fileName: OUT }).then(() => console.log("已生成: " + OUT));
