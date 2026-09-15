# -*- coding: utf-8 -*-
"""输出 PPT 的布局 QA：文字溢出 / 越界 / 重叠检查"""
import sys, math
from pptx import Presentation
from pptx.util import Emu

def emu2in(v): return v / 914400.0

def est_text_h(tf, box_w_in, default_pt=14.0):
    """估算文本所需高度（英寸）：CJK 按 1.0em 宽，拉丁按 0.55em 宽。"""
    total = 0.0
    for p in tf.paragraphs:
        runs = p.runs
        if not runs:
            total += default_pt * 1.4 / 72.0
            continue
        pt = max([(r.font.size.pt if r.font.size else default_pt) for r in runs])
        text = "".join(r.text for r in runs)
        if not text.strip():
            total += pt * 0.6 / 72.0
            continue
        cjk = sum(1 for ch in text if '\u2e80' <= ch <= '\u9fff' or '\uff00' <= ch <= '\uffef')
        latin = len(text) - cjk
        w_pt = cjk * pt * 1.0 + latin * pt * 0.55
        lines = max(1, math.ceil(w_pt / max(box_w_in * 72.0, 1.0)))
        total += lines * pt * 1.45 / 72.0
    return total

def est_text_w(tf, default_pt=14.0):
    widest = 0.0
    for p in tf.paragraphs:
        runs = p.runs
        if not runs: continue
        pt = max([(r.font.size.pt if r.font.size else default_pt) for r in runs])
        text = "".join(r.text for r in p.runs)
        cjk = sum(1 for ch in text if '\u2e80' <= ch <= '\u9fff' or '\uff00' <= ch <= '\uffef')
        latin = len(text) - cjk
        widest = max(widest, (cjk * pt + latin * pt * 0.55) / 72.0)
    return widest

prs = Presentation(sys.argv[1])
SW, SH = emu2in(prs.slide_width), emu2in(prs.slide_height)
print("画布: %.2f x %.2f 英寸 | 共 %d 页\n" % (SW, SH, len(prs.slides)))

# ── 0) XML 合法性：负尺寸/负偏移会让 PowerPoint 拒绝打开整个文件 ──
import zipfile, re
issues = 0
with zipfile.ZipFile(sys.argv[1]) as z:
    for n in sorted(z.namelist()):
        if not re.match(r'ppt/slides/slide\d+\.xml$', n):
            continue
        x = z.read(n).decode("utf-8")
        bad = re.findall(r'<a:ext cx="(-\d+)"', x) + re.findall(r'<a:ext cy="(-\d+)"', x)
        if bad:
            print("  [%s] 非法负尺寸 ext: %s（PowerPoint 会拒绝打开！）" % (n, bad))
            issues += 1

for idx, slide in enumerate(prs.slides, 1):
    boxes = []
    for sh in slide.shapes:
        try:
            x, y = emu2in(sh.left), emu2in(sh.top)
            w, h = emu2in(sh.width), emu2in(sh.height)
        except Exception:
            continue
        has_text = sh.has_text_frame and sh.text_frame.text.strip()
        if has_text:
            boxes.append((sh, x, y, w, h, sh.text_frame))
        # 越界
        if x < -0.02 or y < -0.02 or x + w > SW + 0.02 or y + h > SH + 0.02:
            print("  [P%d] 越界: %-28s x=%.2f y=%.2f w=%.2f h=%.2f" %
                  (idx, (sh.text_frame.text[:24] if has_text else sh.shape_type), x, y, w, h))
            issues += 1

    for sh, x, y, w, h, tf in boxes:
        need_h = est_text_h(tf, w)
        # 允许 8% 容差（行距/字距估算误差）
        if need_h > h * 1.08 + 0.06:
            print("  [P%d] 可能溢出: %-30s 需%.2f\" 实%.2f\"  | %s" %
                  (idx, tf.text[:26].replace("\n", "⏎"), need_h, h, "字号偏大或文字偏多"))
            issues += 1

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            _, x1, y1, w1, h1, tf1 = boxes[i]
            _, x2, y2, w2, h2, tf2 = boxes[j]
            ox = min(x1 + w1, x2 + w2) - max(x1, x2)
            oy = min(y1 + h1, y2 + h2) - max(y1, y2)
            if ox > 0.06 and oy > 0.06:
                area = ox * oy
                if area > 0.05:
                    print("  [P%d] 文本重叠 %.2f in²: 「%s」 × 「%s」" %
                          (idx, area, tf1.text[:18].replace("\n", "⏎"), tf2.text[:18].replace("\n", "⏎")))
                    issues += 1

print("\n检查完毕：共发现 %d 处待确认问题" % issues)
