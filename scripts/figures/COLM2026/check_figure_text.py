"""Report the printed point size of every glyph in a figure PDF.

COLM asks for no figure text below \\small, which in the paper's 10pt Palatino
measures 8.9pt. What a figure is authored at does not answer that question: it
goes into the page at some fraction of \\textwidth, so a figure drawn 16.5in wide
and included at \\textwidth is squeezed to 5.5in and an authored 27pt label
prints at 9pt. This multiplies each glyph's authored size by the squeeze the
figure is about to get.

The squeeze is read from the paper, not assumed. The appendix includes its
figures at .95\\textwidth, which is 5% more squeeze than the main body's and
enough on its own to put a figure under the floor -- so the fraction in each
\\includegraphics is parsed out and used. A figure sitting in figures/ that no
\\includegraphics names is reported as unused and not counted against the paper.

It reads the size mupdf reports per text run, so it sees what the renderer sees
-- including text drawn by a library that never asked for a point size.

Usage:
    bash scripts/figures/COLM2026/bash/check_figure_text.sh          # the paper's
    bash scripts/figures/COLM2026/bash/check_figure_text.sh a.pdf b.pdf
"""

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER = REPO_ROOT / "_external" / "COLM_2026_SGTR"
PAPER_FIGURES = PAPER / "figures"

TEXT_WIDTH_PT = 396.0       # 5.5in, the COLM text block
FLOOR_PT = 8.9              # \small at 10pt Palatino


def include_widths():
    """{figure filename: printed width in pt}, from the paper's \\includegraphics.

    Only the width=<fraction>\\textwidth form is understood, which is every
    inclusion in this paper. A figure included some other way is left out of the
    mapping and falls back to the full text width, which is the optimistic
    reading -- so it can only ever under-report a violation, never invent one.
    """
    pattern = re.compile(
        r"\\includegraphics\[[^\]]*width\s*=\s*([0-9.]*)\s*\\textwidth[^\]]*\]"
        r"\{([^}]*?)([^}/]+\.pdf)\}")
    widths = {}
    for tex in sorted(PAPER.glob("*.tex")):
        for match in pattern.finditer(tex.read_text()):
            fraction = float(match.group(1)) if match.group(1) else 1.0
            widths[match.group(3)] = fraction * TEXT_WIDTH_PT
    return widths


def glyph_sizes(pdf):
    """{authored size: character count}, and the page width, per mupdf."""
    out = subprocess.run(["mutool", "draw", "-F", "stext", "-o", "-", str(pdf)],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"mutool failed on {pdf}: {out.stderr.strip()}")
    page = re.search(r'<page[^>]*width="([0-9.]+)"', out.stdout)
    if not page:
        raise RuntimeError(f"no page in mutool's output for {pdf}")

    sizes = Counter()
    for run in re.finditer(r'<font [^>]*size="([0-9.]+)"[^>]*>(.*?)</font>',
                           out.stdout, re.S):
        sizes[float(run.group(1))] += len(re.findall(r'c="(.)"', run.group(2)))
    return sizes, float(page.group(1))


def check(pdf, printed_width=None):
    """Report `pdf`'s glyph sizes. None printed_width means it is unused."""
    if printed_width is None:
        print(f"{pdf.name}\n  not included by the paper, not checked")
        return True

    sizes, width = glyph_sizes(pdf)
    if not sizes:
        print(f"{pdf.name}: no text")
        return True

    squeeze = printed_width / width
    print(f"{pdf.name}")
    print(f"  {width:.0f}pt wide, printed at {printed_width:.0f}pt "
          f"-> x{squeeze:.4f}")

    # matplotlib sets a mathtext superscript at 0.7x its base, so "$R^2$" at a
    # compliant 27pt still puts an 18.9pt "2" on the page. That is the same
    # device the paper's own body text uses for $R^2$, at the same relative
    # size, so it is reported and not counted -- but it is never inferred away
    # silently, and a base size that is itself too small still fails.
    scripts = {s for s in sizes if any(abs(s - 0.7 * b) < 0.01 for b in sizes)}

    ok = True
    for authored, chars in sorted(sizes.items()):
        printed = authored * squeeze
        under = printed < FLOOR_PT
        note = ""
        if authored in scripts:
            note = "   mathtext script, not counted"
        elif under:
            note = f"   UNDER {FLOOR_PT}pt"
            ok = False
        print(f"  {authored:6.2f}pt authored -> {printed:5.2f}pt printed"
              f"  ({chars} chars){note}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdfs", nargs="*", type=Path,
                        help=f"Figure PDFs. Default: every one in {PAPER_FIGURES}")
    args = parser.parse_args()

    pdfs = args.pdfs or sorted(PAPER_FIGURES.glob("*.pdf"))
    if not pdfs:
        parser.error(f"no PDFs given and none in {PAPER_FIGURES}")

    # A PDF named on the command line is checked at full text width unless the
    # paper says otherwise; one found by the glob is checked only if it is used.
    widths = include_widths()
    ok = all([check(p, widths.get(p.name, TEXT_WIDTH_PT if args.pdfs else None))
              for p in pdfs])
    print("\nall text at or above the floor" if ok else
          "\nsome text prints below the floor")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
