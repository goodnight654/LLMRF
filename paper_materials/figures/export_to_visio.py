#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键将 figures/ 目录下所有 PDF 图表转为 Visio 可编辑的矢量格式

输出格式（按优先级）:
  1. SVG  — 推荐。Visio 2013+ 可直接导入，解组后可编辑每个元素。
  2. EMF  — Windows Enhanced Metafile，Visio 双击即可打开。

依赖安装（若未安装）:
  pip install cairosvg        # PDF/PNG -> SVG
  pip install matplotlib      # EMF 备用方案（Windows 仅）

用法:
  python export_to_visio.py            # 转换全部，生成 SVG + EMF
  python export_to_visio.py --svg      # 仅生成 SVG
  python export_to_visio.py --emf      # 仅生成 EMF
  python export_to_visio.py --check    # 检查依赖是否安装
"""

import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).parent  # paper_materials/figures/
ROOT = HERE.parent.parent      # G:/wenlong/llmrf/


# ─────────────────────────────────────────────
#  依赖检查
# ─────────────────────────────────────────────

def check_deps():
    results = {}

    # PyMuPDF (fitz) — 首选，自带 MuPDF，Windows 无需额外 DLL
    try:
        import fitz
        results['pymupdf'] = fitz.__doc__.split('\n')[0] if fitz.__doc__ else fitz.version[0]
    except Exception:
        results['pymupdf'] = None

    # cairosvg — 备选（Linux/Mac 更顺畅，Windows 需 libcairo.dll）
    try:
        import cairosvg
        results['cairosvg'] = cairosvg.__version__
    except Exception:
        results['cairosvg'] = None

    # inkscape 命令行
    try:
        r = subprocess.run(['inkscape', '--version'], capture_output=True, timeout=5)
        results['inkscape'] = r.stdout.decode(errors='ignore').split('\n')[0]
    except Exception:
        results['inkscape'] = None

    # matplotlib
    try:
        import matplotlib
        results['matplotlib'] = matplotlib.__version__
    except ImportError:
        results['matplotlib'] = None

    return results


# ─────────────────────────────────────────────
#  方法 1: PyMuPDF — PDF → SVG（首选，Windows 兼容）
# ─────────────────────────────────────────────

def convert_pdf_to_svg_pymupdf(pdf_path: Path, svg_path: Path) -> bool:
    """使用 PyMuPDF (fitz) 将 PDF 第一页转为 SVG。"""
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        # 放大 2x 提高精度（SVG 是矢量，但字体等细节更好）
        svg_content = page.get_svg_image(matrix=fitz.Matrix(2, 2))
        svg_path.write_text(svg_content, encoding='utf-8')
        doc.close()
        return True
    except Exception as e:
        print(f'  [pymupdf 失败] {pdf_path.name}: {e}')
        return False


# ─────────────────────────────────────────────
#  方法 2: cairosvg — PDF → SVG（备选）
# ─────────────────────────────────────────────

def convert_pdf_to_svg_cairosvg(pdf_path: Path, svg_path: Path) -> bool:
    try:
        import cairosvg
        cairosvg.pdf2svg(url=str(pdf_path), write_to=str(svg_path))
        return True
    except Exception as e:
        print(f'  [cairosvg 失败] {pdf_path.name}: {e}')
        return False


# ─────────────────────────────────────────────
#  方法 2: Inkscape 命令行 — PDF → SVG
# ─────────────────────────────────────────────

def convert_pdf_to_svg_inkscape(pdf_path: Path, svg_path: Path) -> bool:
    try:
        # Inkscape 1.x 语法
        r = subprocess.run(
            ['inkscape', '--pdf-poppler', str(pdf_path), '--export-type=svg',
             f'--export-filename={svg_path}'],
            capture_output=True, timeout=30
        )
        if svg_path.exists():
            return True
        # Inkscape 0.91 语法备用
        r = subprocess.run(
            ['inkscape', str(pdf_path), f'--export-plain-svg={svg_path}'],
            capture_output=True, timeout=30
        )
        return svg_path.exists()
    except Exception as e:
        print(f'  [inkscape 失败] {pdf_path.name}: {e}')
        return False


# ─────────────────────────────────────────────
#  方法 3: pdf2svg 命令行（Linux/Mac 常用）
# ─────────────────────────────────────────────

def convert_pdf_to_svg_pdf2svg(pdf_path: Path, svg_path: Path) -> bool:
    try:
        r = subprocess.run(['pdf2svg', str(pdf_path), str(svg_path)],
                           capture_output=True, timeout=30)
        return svg_path.exists()
    except Exception:
        return False


# ─────────────────────────────────────────────
#  SVG 转换主函数
# ─────────────────────────────────────────────

def export_all_svg(deps: dict):
    pdfs = sorted(HERE.glob('*.pdf'))
    if not pdfs:
        print('[警告] 未找到 PDF 文件')
        return

    svg_dir = HERE / 'svg'
    svg_dir.mkdir(exist_ok=True)
    print(f'\n[SVG] 输出目录: {svg_dir}')
    print(f'      共 {len(pdfs)} 个 PDF 文件\n')

    ok_count = 0
    for pdf in pdfs:
        svg = svg_dir / (pdf.stem + '.svg')
        if svg.exists():
            print(f'  [跳过] {pdf.name} (已存在)')
            ok_count += 1
            continue

        success = False
        if deps['pymupdf']:
            success = convert_pdf_to_svg_pymupdf(pdf, svg)
        if not success and deps['cairosvg']:
            success = convert_pdf_to_svg_cairosvg(pdf, svg)
        if not success and deps['inkscape']:
            success = convert_pdf_to_svg_inkscape(pdf, svg)
        if not success:
            success = convert_pdf_to_svg_pdf2svg(pdf, svg)

        if success:
            size_kb = svg.stat().st_size / 1024
            print(f'  [OK] {pdf.name:40s} -> {svg.name} ({size_kb:.1f} KB)')
            ok_count += 1
        else:
            print(f'  [失败] {pdf.name} — 无可用转换工具')

    print(f'\n  完成: {ok_count}/{len(pdfs)} 个文件转换成功')
    if ok_count < len(pdfs):
        print('\n  [提示] 安装 cairosvg 可解决大部分失败:')
        print('         pip install cairosvg')


# ─────────────────────────────────────────────
#  EMF 生成（通过重新调用 matplotlib 生成脚本）
# ─────────────────────────────────────────────

def export_all_emf(deps: dict):
    """
    matplotlib 在 Windows 上支持 .emf 后缀输出（需要 pyemf 或 Windows GDI 后端）。
    这里通过动态 patch 各生成脚本的 savefig 调用来输出 EMF。
    """
    if not deps['matplotlib']:
        print('[EMF] matplotlib 未安装，跳过')
        return

    import matplotlib
    import matplotlib.pyplot as plt

    emf_dir = HERE / 'emf'
    emf_dir.mkdir(exist_ok=True)
    print(f'\n[EMF] 输出目录: {emf_dir}')

    # 找出所有 PDF（已知 matplotlib 生成的图）
    pdfs = sorted(HERE.glob('*.pdf'))
    print(f'      共 {len(pdfs)} 个目标文件\n')

    # matplotlib 的 EMF 后端需要在 Windows 上且安装了 pyemf
    # 做一次可用性测试
    emf_available = False
    try:
        import io
        fig_test, ax_test = plt.subplots()
        ax_test.plot([1, 2], [1, 2])
        buf = io.BytesIO()
        fig_test.savefig(buf, format='emf')
        plt.close(fig_test)
        emf_available = True
        print('  [INFO] matplotlib EMF 后端可用')
    except Exception as e:
        print(f'  [INFO] matplotlib EMF 后端不可用: {e}')
        print('  [INFO] 尝试使用 SVG 绕过：Visio 可接受 SVG，推荐使用 --svg 选项')
        return

    if not emf_available:
        return

    # 直接将现有 PDF 通过 cairosvg + svglib 转 EMF 暂不支持
    # 推荐方案: 修改各生成脚本追加 .emf 输出
    print('\n  [提示] EMF 直接生成需重新运行各可视化脚本。')
    print('         请改用 SVG 格式——Visio 对 SVG 的支持同样完整。')
    print('         如需 EMF，可在各生成脚本中追加:')
    print('           fig.savefig("paper_materials/figures/emf/xxx.emf")')


# ─────────────────────────────────────────────
#  在所有生成脚本中追加 SVG 输出（可选补丁）
# ─────────────────────────────────────────────

def patch_generators_for_svg():
    """
    在 ROOT 目录下的图表生成脚本中追加 SVG 保存语句。
    注意：此函数只做展示，不实际修改文件，避免误操作。
    """
    scripts = [
        ROOT / 'generate_paper_figures.py',
        ROOT / 'generate_sparam_figures.py',
        ROOT / 'generate_ablation_figure.py',
    ]
    print('\n[提示] 若要让生成脚本直接输出 SVG，可在各脚本的 savefig 块后追加:')
    print('       fig.savefig(SAVE_DIR / "xxx.svg")  # 或 svg_dir / "xxx.svg"')
    print('\n涉及脚本:')
    for s in scripts:
        print(f'  {s}')


# ─────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    print('=' * 60)
    print('Figures → Visio 可编辑格式 转换工具')
    print('=' * 60)

    deps = check_deps()

    # --check 只显示依赖状态
    if '--check' in args:
        print('\n[依赖状态]')
        for k, v in deps.items():
            status = f'✓ {v}' if v else '✗ 未安装'
            print(f'  {k:<15} {status}')
        print()
        if not any([deps.get('pymupdf'), deps.get('cairosvg'), deps.get('inkscape')]):
            print('[建议] 运行:  pip install pymupdf')
        return

    do_svg = '--emf' not in args or '--svg' in args
    do_emf = '--emf' in args

    # 默认同时生成 SVG
    if not args:
        do_svg = True
        do_emf = False  # EMF 通常需要额外配置，默认跳过

    print('\n[依赖]')
    for k, v in deps.items():
        status = f'✓ {v}' if v else '✗ 未安装'
        print(f'  {k:<15} {status}')

    if do_svg:
        export_all_svg(deps)

    if do_emf:
        export_all_emf(deps)

    print('\n─────────────────────────────────────────')
    print('[在 Visio 中打开 SVG 的步骤]')
    print('  1. Visio → 文件 → 打开 → 选择 svg/ 目录下的 .svg 文件')
    print('  2. 或拖拽 .svg 文件到 Visio 画布')
    print('  3. 右键 → 取消组合 (Ungroup) → 即可单独编辑每个元素')
    print('─────────────────────────────────────────\n')


if __name__ == '__main__':
    main()
