"""
translate_tex.py
将 main.tex 的正文英文内容翻译成中文，生成 main-zh.tex。
- 保留所有 LaTeX 命令、数学公式、引用、标签等不变
- 只翻译自然语言文本
- 使用 OpenAI 兼容 API（默认用 deepseek，可改成其他兼容接口）
"""

import re
import os
import sys
import time

# ── 配置 ────────────────────────────────────────────────────────────────────
API_KEY  = "ollama"          # Ollama 不需要真实 key，随便填
BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL    = os.environ.get("TRANSLATE_MODEL", "qwen3:14b")

INPUT_FILE  = os.path.join(os.path.dirname(__file__), "main.tex")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "main-zh.tex")
# ────────────────────────────────────────────────────────────────────────────

try:
    from openai import OpenAI
except ImportError:
    print("请先安装 openai 包：pip install openai")
    sys.exit(1)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ── 需要跳过（整体不翻译）的环境 ───────────────────────────────────────────
SKIP_ENVS = {
    "equation", "equation*", "align", "align*",
    "algorithmic", "algorithm",
    "tabular", "thebibliography",
}

# ── 需要保护的 LaTeX 命令模式（翻译时占位，翻译后还原）─────────────────────
# 顺序：先匹配较长/复杂的模式，再匹配短的
PROTECT_PATTERNS = [
    # 数学模式 $...$ 和 $$...$$（非贪婪）
    r'\$\$.*?\$\$',
    r'\$[^\$\n]+?\$',
    # \cite{...}
    r'\\cite\{[^}]*\}',
    # \ref{...} \eqref{...} \label{...}
    r'\\(?:ref|eqref|label)\{[^}]*\}',
    # \emph{...} \textbf{...} \textit{...} \text..{...} — 保护命令本身，内容仍翻译
    # （不在此处保护，让翻译模型看到内容）
    # \texttt \url
    r'\\(?:texttt|url)\{[^}]*\}',
    # \~  连字符-波浪号（nbsp）
    r'~',
    # \\  换行
    r'\\\\',
    # \, \; \ （间距命令）
    r'\\[,;!: ]',
    # \% \$ \& \# \_ \{ \}
    r'\\[%$&#_{}]',
    # 数字+单位连写，如 9.4\,h  0.047\%
    r'\d+\.?\d*\\[,;]\\?[a-zA-Z%]+',
]


def protect_latex(text: str):
    """用占位符替换需要保护的 LaTeX 片段，返回 (新文本, 占位符->原文 的字典)。"""
    placeholders = {}
    counter = [0]

    combined = re.compile(
        '|'.join(f'(?:{p})' for p in PROTECT_PATTERNS),
        re.DOTALL
    )

    def replace(m):
        key = f"LTXPH{counter[0]:04d}LTXPH"
        placeholders[key] = m.group(0)
        counter[0] += 1
        return key

    return combined.sub(replace, text), placeholders


def restore_latex(text: str, placeholders: dict) -> str:
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return text


SYSTEM_PROMPT = """你是一名专业的学术论文翻译专家，专注于电子工程和人工智能领域。
请将用户提供的 LaTeX 论文片段从英文翻译成中文。

严格规则：
1. 只翻译自然语言文本，不修改任何 LaTeX 命令、环境、数学公式。
2. 形如 LTXPH0001LTXPH 的占位符原样保留，不得修改。
3. 专业术语翻译参考：
   - RF filter → 射频滤波器
   - fine-tuning → 微调
   - large language model / LLM → 大语言模型（LLM）
   - QLoRA → QLoRA（保留英文）
   - SFT → SFT（保留英文）
   - perturbation-induced reflection / PIR → 扰动诱导反思（PIR）
   - bidirectional performance prediction / BPP → 双向性能预测（BPP）
   - sensitivity-guided → 灵敏度引导
   - curriculum learning → 课程学习
   - passband / stopband → 通带 / 阻带
   - cutoff frequency → 截止频率
   - ripple → 纹波
   - attenuation → 衰减
   - order → 阶数
   - lowpass / highpass / bandpass → 低通 / 高通 / 带通
   - LPF / HPF / BPF → 低通滤波器（LPF）/ 高通滤波器（HPF）/ 带通滤波器（BPF）
   - ablation study → 消融实验
   - closed-loop → 闭环
   - reflective agent → 反思智能体
   - EDA → EDA（保留英文）
4. 句子风格保持学术严谨，不要口语化。
5. 直接输出翻译后的 LaTeX 片段，不要任何解释或说明。
"""


def translate_chunk(text: str, retries: int = 3) -> str:
    """调用 LLM 翻译一段文本（已做占位符保护）。"""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": text},
                ],
                temperature=0.1,
                max_tokens=4096,
            )
            raw = resp.choices[0].message.content or ""
            # 过滤 qwen3 思维链 <think>...</think>
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
            return raw.strip()
        except Exception as e:
            print(f"  [重试 {attempt+1}/{retries}] 错误: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"翻译失败，已重试 {retries} 次")


# ── LaTeX 文档解析与分段 ───────────────────────────────────────────────────

def split_into_translatable_blocks(content: str):
    """
    将 .tex 文件内容分割成若干 block，每个 block 是 (text, should_translate) 二元组。
    - 注释行、跳过环境、preamble（\\begin{document} 之前）、\\begin{thebibliography}
      标记为 should_translate=False
    - 其余段落标记为 should_translate=True
    """
    blocks = []

    # 先按行处理，找到 \begin{document}
    lines = content.split('\n')
    doc_start = 0
    for i, line in enumerate(lines):
        if r'\begin{document}' in line:
            doc_start = i
            break

    # preamble 不翻译
    preamble = '\n'.join(lines[:doc_start + 1])
    blocks.append((preamble + '\n', False))

    body = '\n'.join(lines[doc_start + 1:])

    # 用正则提取需要整体保护的环境
    env_pattern = re.compile(
        r'(\\begin\{(' + '|'.join(re.escape(e) for e in SKIP_ENVS) + r')\}.*?\\end\{\2\})',
        re.DOTALL
    )

    last = 0
    for m in env_pattern.finditer(body):
        # 环境前的文本 → 可翻译
        before = body[last:m.start()]
        if before:
            blocks.append((before, True))
        # 环境本身 → 不翻译
        blocks.append((m.group(0), False))
        last = m.end()

    # 剩余文本
    if last < len(body):
        blocks.append((body[last:], True))

    return blocks


def translate_translatable_block(text: str) -> str:
    """
    对可翻译的 block，进一步按"段落"（空行分隔）拆分，
    每段独立占位保护 → 翻译 → 还原。
    注释行直接保留。
    """
    # 按段落分割（保留空行）
    paragraphs = re.split(r'(\n{2,})', text)
    result = []
    for para in paragraphs:
        # 空白段 / 纯空行分隔符直接保留
        if not para.strip():
            result.append(para)
            continue
        # 全是注释
        if all(l.strip().startswith('%') or l.strip() == '' for l in para.split('\n')):
            result.append(para)
            continue
        # 仅含 LaTeX 命令行（如 \section、\label、\begin、\end、\item 等）
        # 如果段落里没有任何"纯英文单词序列"，就跳过翻译
        # 用简单启发式：去掉所有 LaTeX 命令和数学后，剩余英文字符 >= 10 个才翻译
        stripped = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', '', para)
        stripped = re.sub(r'\$[^\$]*\$', '', stripped)
        stripped = re.sub(r'[^a-zA-Z]', '', stripped)
        if len(stripped) < 10:
            result.append(para)
            continue

        # 保护
        protected, placeholders = protect_latex(para)
        # 翻译
        print(f"  翻译段落（{len(para)} 字符）...")
        translated = translate_chunk(protected)
        # 还原
        translated = restore_latex(translated, placeholders)
        result.append(translated)

    return ''.join(result)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"找不到输入文件：{INPUT_FILE}")
        sys.exit(1)

    print(f"读取：{INPUT_FILE}")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    print("分析文档结构...")
    blocks = split_into_translatable_blocks(content)
    print(f"共 {len(blocks)} 个 block，其中 {sum(1 for _, t in blocks if t)} 个需要翻译")

    translated_parts = []
    for i, (block_text, should_translate) in enumerate(blocks):
        if not should_translate:
            translated_parts.append(block_text)
        else:
            print(f"[Block {i+1}/{len(blocks)}] 翻译中...")
            translated_parts.append(translate_translatable_block(block_text))

    result = ''.join(translated_parts)

    # 在 preamble 加入中文支持包（如果还没有）
    if r'\usepackage{ctex}' not in result and r'\usepackage[UTF8]{ctex}' not in result:
        result = result.replace(
            r'\documentclass[conference]{IEEEtran}',
            r'\documentclass[conference]{IEEEtran}' + '\n' + r'\usepackage[UTF8]{ctex}',
            1
        )

    print(f"\n写出：{OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(result)

    print("完成！")


if __name__ == '__main__':
    main()
