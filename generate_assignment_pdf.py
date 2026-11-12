#!/usr/bin/env python3
"""Generate DSP_Assignment_Report_Naman_Gupta.pdf from the repository's real results.

Usage:  python generate_assignment_pdf.py [--skip-tests]

Theme: white pages with black serif text (STIX General, shipped with Matplotlib); only the figures and the dark code blocks use colour.
All numbers come from outputs/analysis_results.json or are recomputed from the original recordings with the
project's own functions; all code listings are extracted from the real source files; figures are regenerated in
colour from the real recordings (report_assets/figures/). Unit tests are run during generation unless
--skip-tests is given, so the reported test result is real.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from pygments import lex  # noqa: E402
from pygments.lexers import PythonLexer  # noqa: E402
from pygments.token import Token  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak,  # noqa: E402
                                PageTemplate, Paragraph, Spacer, Table, TableStyle, XPreformatted)
from reportlab.platypus.tableofcontents import TableOfContents  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import assignment1 as a  # noqa: E402

ASSETS = ROOT / "report_assets"
FIGDIR = ASSETS / "figures"
OUT_PDF = ROOT / "DSP_Assignment_Report_Naman_Gupta.pdf"
GITHUB = "https://github.com/Naman21036/voice-signal-analysis"
SUBMISSION_DATE = "17 October 2026"  # fixed so the report does not change with the day it is regenerated
BLACK, DGRAY, MGRAY = colors.black, colors.HexColor("#333333"), colors.HexColor("#666666")
CODE_BG = colors.HexColor("#F2F2F2")
TEXT_W = A4[0] - 4.2 * cm

# plot palette (only the figures are coloured)
C_WAVE, C_SPEECH, C_SILENCE, C_AMBIG = "#2563EB", "#22C55E", "#EF4444", "#F59E0B"
C_TEAL, C_PURPLE, C_SLATE, C_GRID, C_INK = "#0D9488", "#8B5CF6", "#475569", "#E5E7EB", "#222222"
CMAP = "magma"


# --------------------------------------------------------------------------- fonts
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for fam, files in (("STIX", ("STIXGeneral.ttf", "STIXGeneralBol.ttf", "STIXGeneralItalic.ttf", "STIXGeneralBolIta.ttf")),
                   ("Mono", ("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf", "DejaVuSansMono-Oblique.ttf", "DejaVuSansMono-BoldOblique.ttf"))):
    names = [fam, fam + "-Bold", fam + "-Italic", fam + "-BoldItalic"]
    for n, f in zip(names, files):
        pdfmetrics.registerFont(TTFont(n, str(FONT_DIR / f)))
    pdfmetrics.registerFontFamily(fam, normal=names[0], bold=names[1], italic=names[2], boldItalic=names[3])
SERIF, SERIF_B, SERIF_I, MONO = "STIX", "STIX-Bold", "STIX-Italic", "Mono"

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 9,
                     "axes.titlesize": 9.5, "axes.labelsize": 9, "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#222222", "axes.labelcolor": "#222222", "text.color": "#222222", "xtick.color": "#222222",
                     "ytick.color": "#222222", "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})


def S_(name, **kw):
    return ParagraphStyle(name, **kw)


S = {
    "body": S_("body", fontName=SERIF, fontSize=10.2, leading=12.8, alignment=TA_JUSTIFY, textColor=BLACK, spaceAfter=3.8),
    "small": S_("small", fontName=SERIF, fontSize=9, leading=11, textColor=DGRAY, alignment=TA_LEFT),
    "cap": S_("cap", fontName=SERIF, fontSize=9.3, leading=11.4, textColor=BLACK, alignment=TA_JUSTIFY, spaceBefore=2, spaceAfter=7),
    "tcap": S_("tcap", fontName=SERIF, fontSize=9.3, leading=11.4, textColor=BLACK, alignment=TA_LEFT, spaceBefore=3, spaceAfter=3, keepWithNext=1),
    "H1": S_("H1", fontName=SERIF_B, fontSize=15.5, leading=19, textColor=BLACK, spaceBefore=9, spaceAfter=4, keepWithNext=1),
    "H2": S_("H2", fontName=SERIF_B, fontSize=12, leading=15, textColor=BLACK, spaceBefore=5, spaceAfter=2.5, keepWithNext=1),
    "bullet": S_("bullet", bulletFontName=SERIF, fontName=SERIF, fontSize=10.2, leading=12.6, leftIndent=14, bulletIndent=3, alignment=TA_JUSTIFY, spaceAfter=2),
    "cell": S_("cell", fontName=SERIF, fontSize=9.3, leading=11.2, alignment=TA_LEFT),
    "cellb": S_("cellb", fontName=SERIF_B, fontSize=9.3, leading=11.2, alignment=TA_LEFT),
    "code": S_("code", fontName=MONO, fontSize=8.5, leading=10.7, textColor=colors.HexColor("#F8FAFC"), backColor=colors.HexColor("#111111"),
               borderColor=colors.HexColor("#374151"), borderWidth=0.5, borderPadding=(5, 6, 5, 6), leftIndent=6, rightIndent=6,
               spaceBefore=5, spaceAfter=9),
    "title": S_("title", fontName=SERIF_B, fontSize=24, leading=30, textColor=BLACK, alignment=TA_CENTER),
    "subtitle": S_("subtitle", fontName=SERIF_I, fontSize=14, leading=18, textColor=DGRAY, alignment=TA_CENTER),
    "center": S_("center", fontName=SERIF, fontSize=11, leading=14, alignment=TA_CENTER),
    "ref": S_("ref", bulletFontName=SERIF, fontName=SERIF, fontSize=9.6, leading=11.8, leftIndent=18, firstLineIndent=-18, spaceAfter=2, alignment=TA_LEFT),
}


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def bullets(items: list[str], glyph: str = "•") -> list:
    return [Paragraph(t, S["bullet"], bulletText=glyph) for t in items]


def mono(t: str) -> str:
    return f"<font face='{MONO}' size='8.6'>{t}</font>"


def hrule(width=TEXT_W, thickness=0.8) -> Table:
    t = Table([[""]], colWidths=[width], rowHeights=[2])
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), thickness, BLACK)]))
    return t


def table(rows: list[list], widths: list[float]) -> Table:
    """Booktabs-style table (rules only, no vertical lines, no colour)."""
    data = [[Paragraph(c, S["cellb"] if i == 0 else S["cell"]) if isinstance(c, str) else c for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1.0, BLACK), ("LINEBELOW", (0, 0), (-1, 0), 0.5, BLACK),
                           ("LINEBELOW", (0, -1), (-1, -1), 1.0, BLACK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


_N = {"fig": 0, "tab": 0}


def tcap(text: str) -> Paragraph:
    _N["tab"] += 1
    return Paragraph(f"<b>Table {_N['tab']}.</b> {text}", S["tcap"])


def fig(path: Path, caption: str, width_cm: float):
    _N["fig"] += 1
    w, h = PILImage.open(path).size
    img = Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)
    img.hAlign = "CENTER"
    return KeepTogether([img, Paragraph(f"<b>Figure {_N['fig']}.</b> {caption}", S["cap"])])


def eq(tex: str, name: str, number: int, size: int = 13) -> Table:
    (ASSETS / "equations").mkdir(parents=True, exist_ok=True)
    path = ASSETS / "equations" / f"{name}.png"
    f_ = plt.figure(figsize=(0.1, 0.1), facecolor="white")
    f_.text(0, 0, f"${tex}$", fontsize=size, color="black")
    f_.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(f_)
    w, h = PILImage.open(path).size
    k = 72 / 300 * 0.95
    img = Image(str(path), width=w * k, height=h * k)
    t = Table([["", img, Paragraph(f"({number})", S["cell"])]], colWidths=[0.5 * cm, TEXT_W - 2.4 * cm, 1.4 * cm])
    t.setStyle(TableStyle([("ALIGN", (1, 0), (1, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (2, 0), (2, 0), "RIGHT"),
                           ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return t


# --------------------------------------------------------------------------- code listings (real source, grayscale highlighting)
def get_lines(path: str, start: str, end: str | None = None, count: int = 1) -> list[str]:
    src = (ROOT / path).read_text(encoding="utf-8").split("\n")
    i0 = next(i for i, l in enumerate(src) if start in l)
    i1 = next(i for i in range(i0, len(src)) if end in src[i]) if end else i0 + count - 1
    return textwrap.dedent("\n".join(src[i0:i1 + 1])).split("\n")


# dark code-block theme (only the code blocks are dark; everything else stays black on white)
CODE_BG_DARK = "#111111"
CODE_FG = "#F8FAFC"
CODE_COL = {"keyword": "#C084FC", "string": "#86EFAC", "number": "#FDBA74", "function": "#7DD3FC", "comment": "#9CA3AF",
            "builtin": "#FDE68A", "punct": "#D1D5DB"}
CODE_WRAP = 86


def _wrap_code(lines: list[str], width: int = CODE_WRAP) -> list[str]:
    """Wrap over-long lines at spaces with a hanging indent (display only; code is not altered otherwise)."""
    # re-indent aligned continuation lines (e.g. arguments aligned to an opening bracket) to a 4-space hanging indent
    tidy: list[str] = []
    for l in lines:
        if tidy and tidy[-1].rstrip().endswith((",", "(")) and l.strip() and not l.lstrip().startswith("#"):
            prev_indent = len(tidy[-1]) - len(tidy[-1].lstrip())
            cur = len(l) - len(l.lstrip())
            if cur > prev_indent + 8:
                l = " " * (prev_indent + 4) + l.lstrip()
        tidy.append(l)
    lines = tidy
    out: list[str] = []
    for l in lines:
        if len(l) <= width:
            out.append(l)
            continue
        indent = len(l) - len(l.lstrip())
        out.extend(textwrap.wrap(l, width=width, subsequent_indent=" " * (indent + 4), break_long_words=False,
                                 break_on_hyphens=False, replace_whitespace=False))
    return out


def listing(*blocks: list[str]) -> KeepTogether:
    lines: list[str] = []
    for i, b in enumerate(blocks):
        if i:
            lines.append("# ...")
        lines += b
    lines = _wrap_code(lines)
    toks = list(lex("\n".join(lines), PythonLexer()))
    out = []
    for idx, (tok, val) in enumerate(toks):
        esc = val.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        color = None
        if val.strip():
            nxt = toks[idx + 1][1] if idx + 1 < len(toks) else ""
            if tok in Token.Comment:
                color = CODE_COL["comment"]
                esc = f"<i>{esc}</i>"
            elif tok in Token.Literal.String:
                color = CODE_COL["string"]
            elif tok in Token.Literal.Number or tok in Token.Keyword.Constant:
                color = CODE_COL["number"]
            elif tok in Token.Keyword or tok in Token.Operator.Word:
                color = CODE_COL["keyword"]
            elif tok in Token.Name.Builtin:
                color = CODE_COL["builtin"]
            elif tok in Token.Name.Function or (tok is Token.Name and nxt.startswith("(")):
                color = CODE_COL["function"]
            elif tok in Token.Operator or tok in Token.Punctuation:
                color = CODE_COL["punct"]
        out.append(f"<font color='{color}'>{esc}</font>" if color else esc)
    return KeepTogether([XPreformatted("".join(out).rstrip("\n"), S["code"])])


# --------------------------------------------------------------------------- document template
class Doc(BaseDocTemplate):
    def __init__(self, filename, **kw):
        super().__init__(filename, pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm, topMargin=2.1 * cm, bottomMargin=1.9 * cm,
                         title="Digital Signal Processing of Speech Signals", author="Naman Gupta",
                         subject="Voice Recording, Silence Detection, Audio Preprocessing and Spectral Analysis", **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f", leftPadding=0, rightPadding=0, topPadding=0,
                      bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="first", frames=[frame], onPage=lambda c, d: None, autoNextPageTemplate="rest"),
                               PageTemplate(id="rest", frames=[frame], onPage=self._decorate)])

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setFont(SERIF_I, 9)
        canvas.setFillColor(MGRAY)
        canvas.drawString(self.leftMargin, A4[1] - 1.25 * cm, "Digital Signal Processing of Speech Signals")
        canvas.drawRightString(A4[0] - self.rightMargin, A4[1] - 1.25 * cm, "Naman Gupta")
        canvas.setStrokeColor(MGRAY)
        canvas.setLineWidth(0.4)
        canvas.line(self.leftMargin, A4[1] - 1.4 * cm, A4[0] - self.rightMargin, A4[1] - 1.4 * cm)
        canvas.setFont(SERIF, 9.5)
        canvas.setFillColor(BLACK)
        canvas.drawCentredString(A4[0] / 2, 1.0 * cm, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and fl.style.name in ("H1", "H2"):
            lvl = 0 if fl.style.name == "H1" else 1
            text = re.sub("<[^>]+>", "", fl.text)
            HEADINGS.append(text)
            # page numbers come from the rendered PDF of the previous pass when available (see build())
            self.notify("TOCEntry", (lvl, text, PAGE_MAP.get(text, self.page)))


HEADINGS: list[str] = []
PAGE_MAP: dict[str, int] = {}


def scan_heading_pages(pdf: Path, headings: list[str]) -> dict[str, int]:
    """Find the page on which each heading is actually rendered (pages after the contents page)."""
    import pymupdf
    doc = pymupdf.open(str(pdf))
    norm = lambda t: re.sub(r"\s+", " ", t).strip()  # noqa: E731
    found: dict[str, int] = {}
    for h in headings:
        for i in range(2, len(doc)):
            if any(norm(l) == norm(h) for l in doc[i].get_text().splitlines()):
                found[h] = i + 1
                break
    return found


def H1(t):
    return Paragraph(t, S["H1"])


def H2(t):
    return Paragraph(t, S["H2"])


# --------------------------------------------------------------------------- data and grayscale figures
def load_results() -> dict:
    return json.loads((ROOT / "outputs" / "analysis_results.json").read_text(encoding="utf-8"))


def run_tests(skip: bool) -> str:
    if skip:
        return "The automated tests were not run during PDF generation."
    res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"(\d+) passed", res.stdout)
    if res.returncode == 0 and m:
        return f"All {m.group(1)} automated tests passed (executed while generating this report)."
    return "The automated tests were executed while generating this report and did NOT all pass; see the repository."


def leading_zero_s(data, fs) -> float:
    nz = np.flatnonzero(np.any(data != 0, axis=1))
    return float(nz[0] / fs) if nz.size else data.shape[0] / fs


def _channels(ax, d, fs, t0=0.0, t1=None, left="#2563EB", right="#475569"):
    i0, i1 = int(t0 * fs), (d.shape[0] if t1 is None else int(t1 * fs))
    t = np.arange(i0, i1) / fs
    ax.plot(t, d[i0:i1, 0], color=left, lw=0.5, label="Left", zorder=3)
    ax.plot(t, d[i0:i1, 1], color=right, lw=0.4, alpha=0.65, label="Right", zorder=2)
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t0, (d.shape[0] / fs) if t1 is None else t1)


def make_figures(rec, data, cfg, R=None) -> dict[str, Path]:
    """Full-colour scientific figures (only the plots are coloured; the document itself stays black on white)."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = {}
    names = {"bit": "BIT (Naman_BIT.wav)", "name": "Naman Gupta (Naman_Name.wav)"}
    dets, procs, infos = {}, {}, {}
    for k in rec:
        d, fs = data[k]
        dets[k] = a.detect_silence(d, fs, cfg)
        procs[k], infos[k] = a.trim_silence(d, fs, dets[k], cfg)
    ymax = {k: 1.05 * np.max(np.abs(data[k][0])) for k in rec}

    def shade(ax, det, alpha=0.20):
        for r in det["speech"]:
            ax.axvspan(r["start_s"], r["end_s"], facecolor=C_SPEECH, alpha=alpha, lw=0, zorder=0)
        for r in det["silence"]:
            ax.axvspan(r["start_s"], r["end_s"], facecolor=C_SILENCE, alpha=alpha, lw=0, zorder=0)
        for r in det["ambiguous"]:
            ax.axvspan(r["start_s"], r["end_s"], facecolor=C_AMBIG, alpha=alpha + 0.08, lw=0, zorder=0)

    region_handles = [Patch(facecolor=C_SPEECH, alpha=0.35, label="Speech"), Patch(facecolor=C_SILENCE, alpha=0.35, label="Silence"),
                      Patch(facecolor=C_AMBIG, alpha=0.45, label="Ambiguous")]

    def fig_title(ax, text):
        ax.set_title(text, loc="left", fontsize=9.5)

    # Fig 1: waveforms with detected regions
    f_, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), constrained_layout=True)
    for ax, k, tag in zip(axes, rec, "ab"):
        d, fs = data[k]
        _channels(ax, d, fs, left=C_WAVE, right=C_SLATE)
        shade(ax, dets[k])
        ax.set_ylim(-ymax[k], ymax[k])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig_title(ax, f"({tag}) {names[k]}, f$_s$ = {fs} Hz")
        if tag == "a":
            h, _ = ax.get_legend_handles_labels()
            ax.legend(handles=h + region_handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7.2, ncol=5, frameon=True, framealpha=0.95, edgecolor="#E5E7EB")
    out["waveforms"] = FIGDIR / "fig_waveforms.png"
    f_.savefig(out["waveforms"], dpi=220)
    plt.close(f_)

    # Fig 2: detection (waveform + frame level and thresholds)
    f_, axes = plt.subplots(4, 1, figsize=(7.2, 7.0), constrained_layout=True, gridspec_kw={"height_ratios": [2, 1.2, 2, 1.2]})
    for j, (k, tag) in enumerate(zip(rec, "ab")):
        d, fs = data[k]
        det = dets[k]
        ax, ax2 = axes[2 * j], axes[2 * j + 1]
        _channels(ax, d, fs, left=C_WAVE, right=C_SLATE)
        shade(ax, det)
        ax.set_ylim(-ymax[k], ymax[k])
        ax.set_ylabel("Amplitude")
        fig_title(ax, f"({tag}) {names[k]}: detected regions")
        ax2.plot(det["frame_times_s"], det["frame_rms_db"], color=C_WAVE, lw=1.0, label="Frame level $L_m$")
        ax2.axhline(det["high_threshold_db"], color=C_PURPLE, ls="--", lw=1.1, label=f"Onset {det['high_threshold_db']:.1f} dB")
        ax2.axhline(det["threshold_db"], color=C_SLATE, ls="-.", lw=1.1, label=f"Hold {det['threshold_db']:.1f} dB")
        ax2.axhline(det["noise_floor_db"], color="#9CA3AF", ls=":", lw=1.4, label=f"Noise floor {det['noise_floor_db']:.1f} dB")
        shade(ax2, det, alpha=0.10)
        ax2.set_ylim(-70, 0)
        ax2.set_xlim(0, d.shape[0] / fs)
        ax2.set_ylabel("Level (dBFS)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc="lower right", fontsize=7, ncol=2, frameon=True, framealpha=0.95, edgecolor="#E5E7EB")
        if j == 0:
            ax.legend(handles=region_handles + [Line2D([0], [0], color=C_WAVE, lw=1, label="Left"), Line2D([0], [0], color=C_SLATE, lw=1, label="Right")],
                      loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=7.2, ncol=5, frameon=True, framealpha=0.95, edgecolor="#E5E7EB")
    out["detection"] = FIGDIR / "fig_detection.png"
    f_.savefig(out["detection"], dpi=220)
    plt.close(f_)

    # Fig 3: before/after trimming (blue original, teal processed)
    f_, axes = plt.subplots(4, 1, figsize=(7.2, 6.2), constrained_layout=True)
    for j, (k, tag) in enumerate(zip(rec, "ab")):
        d, fs = data[k]
        t0 = infos[k]["kept_segments"][0][0]
        t1 = infos[k]["kept_segments"][-1][1]
        dur = d.shape[0] / fs
        ax, bx = axes[2 * j], axes[2 * j + 1]
        ax.plot(np.arange(d.shape[0]) / fs, d[:, 0], color=C_WAVE, lw=0.5, label="Original (left)")
        ax.plot(np.arange(d.shape[0]) / fs, d[:, 1], color=C_SLATE, lw=0.4, alpha=0.6, label="Original (right)")
        ax.axvspan(0, t0, facecolor=C_SILENCE, alpha=0.15, lw=0, zorder=0, label="Removed")
        ax.axvspan(t1, dur, facecolor=C_SILENCE, alpha=0.15, lw=0, zorder=0)
        ax.axvline(t0, color=C_PURPLE, ls="--", lw=1.1, label="Cut points")
        ax.axvline(t1, color=C_PURPLE, ls="--", lw=1.1)
        ax.set_xlim(0, dur)
        ax.set_ylim(-ymax[k], ymax[k])
        ax.set_ylabel("Amplitude")
        fig_title(ax, f"({tag}) {names[k]}: original, {dur:.3f} s")
        ax.legend(loc="upper right", fontsize=7.2, ncol=4, frameon=True, framealpha=0.95, edgecolor="#E5E7EB")
        p = procs[k]
        bx.plot(np.arange(p.shape[0]) / fs, p[:, 0], color=C_TEAL, lw=0.5, label="Processed (left)")
        bx.plot(np.arange(p.shape[0]) / fs, p[:, 1], color=C_SLATE, lw=0.4, alpha=0.6, label="Processed (right)")
        bx.set_xlim(0, dur)
        bx.set_ylim(-ymax[k], ymax[k])
        bx.set_ylabel("Amplitude")
        bx.set_xlabel("Time (s)")
        fig_title(bx, f"processed (trimmed), {p.shape[0] / fs:.3f} s, same axes")
        bx.legend(loc="upper right", fontsize=7.2, ncol=2, frameon=True, framealpha=0.95, edgecolor="#E5E7EB")
    out["trim"] = FIGDIR / "fig_trim.png"
    f_.savefig(out["trim"], dpi=220)
    plt.close(f_)

    # Fig 4: spectrograms (magma, identical dB scale, 0 dB = peak of the original recording)
    f_, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), constrained_layout=True)
    for ax, k, tag in zip(axes, rec, "ab"):
        d, fs = data[k]
        ref = a.spectrogram_reference(d, fs, cfg)
        f, t, sxx = a.compute_spectrogram(a.to_mono(d), fs, cfg)
        m = ax.pcolormesh(t, f, 10 * np.log10(sxx / ref + a.EPS), shading="auto", cmap=CMAP, vmin=-cfg.spec_range_db, vmax=0)
        ax.set_ylim(0, cfg.spec_max_hz)
        ax.set_xlim(0, d.shape[0] / fs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig_title(ax, f"({tag}) {names[k]}: spectrogram")
        ax.grid(False)
        cb = f_.colorbar(m, ax=ax, pad=0.01)
        cb.set_label("Power (dB re. peak)")
        cb.outline.set_edgecolor("#222222")
    out["spec"] = FIGDIR / "fig_spectrograms.png"
    f_.savefig(out["spec"], dpi=220)
    plt.close(f_)

    # Fig 5: FFT size comparison (BIT)
    d, fs = data["bit"]
    ref = a.spectrogram_reference(d, fs, cfg)
    f_, axes = plt.subplots(3, 1, figsize=(7.2, 4.6), sharex=True, constrained_layout=True)
    for ax, nfft in zip(axes, (512, 1024, 2048)):
        f, t, sxx = a.compute_spectrogram(a.to_mono(d), fs, cfg, nfft=nfft, hop=nfft // 8)
        m = ax.pcolormesh(t, f, 10 * np.log10(sxx / ref + a.EPS), shading="auto", cmap=CMAP, vmin=-cfg.spec_range_db, vmax=0)
        ax.set_ylim(0, 4000)
        ax.set_ylabel("Frequency (Hz)")
        fig_title(ax, f"N = {nfft}: window {1000 * nfft / fs:.1f} ms, bin spacing {fs / nfft:.1f} Hz, hop N/8")
        ax.grid(False)
        cb = f_.colorbar(m, ax=ax, pad=0.01)
        cb.set_label("Power (dB re. peak)")
        cb.outline.set_edgecolor("#222222")
    axes[-1].set_xlabel("Time (s)")
    out["fft"] = FIGDIR / "fig_fft_sizes.png"
    f_.savefig(out["fft"], dpi=220)
    plt.close(f_)

    # cover: small waveform
    f_, ax = plt.subplots(figsize=(7.0, 2.2), constrained_layout=True)
    d, fs = data["bit"]
    _channels(ax, d, fs, left=C_WAVE, right=C_SLATE)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True, framealpha=0.95, edgecolor="#E5E7EB")
    out["cover"] = FIGDIR / "fig_cover.png"
    f_.savefig(out["cover"], dpi=220)
    plt.close(f_)

    # Fig 6: synthetic evaluation, baseline versus improved detector (frame-level F1 and speech preserved)
    if R is not None:
        se = R["synthetic_evaluation"]["scenarios"]
        keys = ["clean", "snr_20dB", "snr_10dB", "little_silence_(speech_fills_file)", "stereo_opposite_polarity", "noise_burst_300ms_(-34dBFS)_in_silence"]
        labels = ["clean", "SNR 20 dB", "SNR 10 dB", "little\nsilence", "opposite\npolarity", "noise\nburst"]
        f_, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True, sharey=True)
        x = np.arange(len(keys))
        for ax, metric, title in zip(axes, ("f1", "speech_preserved_fraction"), ("Frame-level F1", "True speech preserved by trimming")):
            b = [se[k]["baseline"][metric] for k in keys]
            i_ = [se[k]["improved"][metric] for k in keys]
            ax.bar(x - 0.19, b, 0.36, color=C_SLATE, label="Baseline detector", hatch="//", edgecolor="white", lw=0.5)
            ax.bar(x + 0.19, i_, 0.36, color=C_WAVE, label="Improved detector", edgecolor="white", lw=0.5)
            for xi, v in zip(x - 0.19, b):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.3, color="#222222")
            for xi, v in zip(x + 0.19, i_):
                ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=6.3, color="#222222")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_ylim(0, 1.15)
            fig_title(ax, title)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("Score (0 to 1)")
        h_, l_ = axes[0].get_legend_handles_labels()
        f_.legend(h_, l_, loc="outside upper center", ncol=2, fontsize=7.5, frameon=False)
        out["synthetic"] = FIGDIR / "fig_synthetic.png"
        f_.savefig(out["synthetic"], dpi=220)
        plt.close(f_)
    return out


def rt(x, d=3):
    return f"{x:.{d}f}"


# --------------------------------------------------------------------------- report body
def build(skip_tests: bool) -> None:
    R = load_results()
    cfg = a.Config(**R["config"])
    rec = R["recordings"]
    bit, name = rec["bit"], rec["name"]
    cv = R.get("content_verification") or {"recordings": {}}
    test_line = run_tests(skip_tests)
    today = SUBMISSION_DATE
    data = {k: a.load_audio(ROOT / "audio" / rec[k]["metadata"]["source_file"]) for k in rec}
    fs = bit["metadata"]["sampling_frequency_hz"]
    zero = {k: leading_zero_s(*data[k]) for k in rec}
    cons = {}
    for k in rec:
        d, f_ = data[k]
        det = a.detect_silence(d, f_, replace(cfg, trim_ambiguous=False))
        p, _ = a.trim_silence(d, f_, det, replace(cfg, trim_ambiguous=False))
        cons[k] = (p.shape[0] / f_, (1 - p.shape[0] / d.shape[0]) * 100)
    amb_peak = {k: [float(np.max(np.abs(data[k][0][int(e["region"]["start_s"] * fs):int(e["region"]["end_s"] * fs)]))) for e in rec[k]["region_analysis"]]
                for k in rec}
    figs = make_figures(rec, data, cfg, R)
    sdb, sdn = bit["silence_detection"], name["silence_detection"]

    st: list = []
    A = st.append

    # ============ Cover
    A(Spacer(1, 2.2 * cm))
    A(P("DEPARTMENT OF ELECTRONICS AND COMMUNICATION ENGINEERING", "center"))
    A(P("Birla Institute of Technology, Mesra", "center"))
    A(Spacer(1, 1.4 * cm))
    A(Paragraph("Digital Signal Processing of Speech Signals", S["title"]))
    A(Spacer(1, 0.4 * cm))
    A(Paragraph("Voice Recording, Silence Detection, Audio Preprocessing and Spectral Analysis", S["subtitle"]))
    A(Spacer(1, 0.8 * cm))
    A(hrule(thickness=1.2))
    A(Spacer(1, 0.8 * cm))
    cw, ch = PILImage.open(figs["cover"]).size
    im = Image(str(figs["cover"]), width=14.5 * cm, height=14.5 * cm * ch / cw)
    im.hAlign = "CENTER"
    A(im)
    A(Paragraph("Waveform of the recording <i>Naman_BIT.wav</i> (left and right channels), computed from the original file.", ParagraphStyle("cc", parent=S["small"], alignment=TA_CENTER)))
    A(Spacer(1, 1.3 * cm))
    info = [["Student", "Naman Gupta"], ["Roll number", "BTECH/10887/24"], ["Programme", "B.Tech, Electronics and Communication Engineering, Section C"],
            ["Course", "EC24313 Artificial Intelligence and Machine Learning, Assignment 1 (Digital Signal Processing of voice recordings)"],
            ["Submission date", today]]
    t = Table([[Paragraph(f"<b>{k}</b>", S["cell"]), Paragraph(v, S["cell"])] for k, v in info], colWidths=[3.6 * cm, 12.0 * cm])
    t.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, BLACK), ("LINEBELOW", (0, -1), (-1, -1), 0.8, BLACK),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    t.hAlign = "CENTER"
    A(t)
    A(PageBreak())

    # ============ Abstract + contents
    A(Paragraph("<b>Abstract</b>", ParagraphStyle("abs", fontName=SERIF_B, fontSize=13, leading=16, spaceAfter=3)))
    A(P(f"This report analyses two voice recordings, one of the word &ldquo;BIT&rdquo; and one of the name &ldquo;Naman Gupta&rdquo;, with a reproducible Python "
        f"pipeline. Both files are 16-bit stereo WAV recordings with a sampling frequency of {fs:,} Hz read from the file headers; their durations are "
        f"{rt(bit['metadata']['duration_s'])} s and {rt(name['metadata']['duration_s'])} s. Time-domain waveforms were plotted, silent and low-energy regions were located "
        "with an adaptive frame-based energy detector that separates confident speech, silence and ambiguous low-level activity, the recordings were trimmed with padding and fades "
        f"(to {rt(bit['reduction']['processed_duration_s'])} s and {rt(name['reduction']['processed_duration_s'])} s, i.e. {bit['reduction']['percent_removed']:.1f} % and "
        f"{name['reduction']['percent_removed']:.1f} % removed), and short-time Fourier transform spectrograms were computed. The processed audio passed all applicable integrity "
        "checks and the code is covered by automated tests and by controlled synthetic experiments. Speech intelligibility of the processed audio, and the spoken content of the files, were "
        "not verified by listening; these and other limitations are stated in Section 10."))
    A(Spacer(1, 4))
    A(Paragraph("<b>Contents</b>", ParagraphStyle("cont", fontName=SERIF_B, fontSize=13, leading=16, spaceAfter=3)))
    toc = TableOfContents()
    toc.dotsMinLevel = 0
    toc.tableStyle = TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                 ("TOPPADDING", (0, 0), (-1, -1), 0.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 0.4)])
    toc.levelStyles = [ParagraphStyle("t0", fontName=SERIF_B, fontSize=11, leading=13, leftIndent=18, firstLineIndent=-18, spaceBefore=3, spaceAfter=0),
                       ParagraphStyle("t1", fontName=SERIF, fontSize=10, leading=11.3, leftIndent=40, firstLineIndent=-22, spaceBefore=0, spaceAfter=0)]
    A(toc)
    A(PageBreak())

    # ============ 1 Introduction
    A(H1("1  Introduction and Objectives"))
    A(H2("1.1  Purpose"))
    A(P("Speech is a non-stationary signal: its amplitude and its spectrum both change within tens of milliseconds, so it is analysed in the time domain (when energy is present) and in the "
        "time-frequency domain (which frequencies are present). A recorded file is only a sequence of amplitude samples; the sampling frequency fixes the time scale and the highest frequency that "
        "can be represented, silence at the start and end adds duration without information, and a spectrogram reveals voiced sounds, bursts and noise that a waveform cannot separate. "
        "This assignment applies these ideas to two recordings of the author's voice."))
    A(H2("1.2  Assignment objectives"))
    A(table([["No.", "Objective", "Section"],
             ["1", "Record &ldquo;BIT&rdquo; and the student's name (<i>YOURNAME_BIT.wav</i>, <i>YOURNAME_NAME.wav</i>)", "3"],
             ["2", "Plot the time-domain waveforms and determine the sampling frequency", "3, 4"],
             ["3", "Identify and mark silent or unvoiced regions in the waveforms", "5"],
             ["4", "Trim or minimise silent and flat regions without removing speech", "6"],
             ["5", "Apply a time-frequency transformation and plot the spectrograms", "7"]], [1.2, 12.6, 1.8]))
    A(Spacer(1, 4))
    A(H2("1.3  Tools"))
    A(P(f"Python 3 with NumPy (frame computations), SoundFile (WAV input and header metadata), SciPy ({mono('scipy.signal.spectrogram')} for the STFT), Matplotlib (plots), "
        "pytest (tests) and ReportLab (this document). Everything runs on a CPU without external services."))

    # ============ 2 Theory
    A(H1("2  Theoretical Background"))
    A(H2("2.1  Sampling and discrete-time signals"))
    A(P("A continuous-time speech signal <i>x<sub>c</sub></i>(<i>t</i>) is measured every <i>T<sub>s</sub></i> = 1/<i>f<sub>s</sub></i> seconds, giving the discrete-time signal "
        "<i>x</i>[<i>n</i>] of Eq. (1). Sample <i>n</i> corresponds to the time of Eq. (2), where <i>N</i> is the number of samples per channel and <i>T</i> the duration. "
        "To avoid aliasing, <i>f<sub>s</sub></i> must exceed twice the highest frequency present (Nyquist criterion); frequencies up to <i>f<sub>s</sub></i>/2 can be represented. "
        "The sampling frequency is a property of the recording and must not be confused with the pitch (fundamental frequency) of the voice, which is roughly 85&ndash;255 Hz for adults."))
    A(eq(r"x[n]=x_c(nT_s),\qquad T_s=\frac{1}{f_s}", "eq1", 1))
    A(eq(r"t_n=\frac{n}{f_s},\qquad T=\frac{N}{f_s}", "eq2", 2))
    A(H2("2.2  Amplitude measures"))
    A(P("Amplitudes are relative to digital full scale (range &minus;1 to +1). The peak and the root-mean-square (RMS) amplitude are"))
    A(eq(r"A_{pk}=\max_n |x[n]|,\qquad A_{rms}=\sqrt{\frac{1}{N}\sum_{n=0}^{N-1}x^2[n]}", "eq3", 3))
    A(H2("2.3  Silence, unvoiced speech and background noise"))
    A(P("Exact digital silence (all-zero samples) is rare; a pause in speech still contains background noise, so &ldquo;silence&rdquo; here means a stretch whose short-time energy stays near the "
        "recording's noise floor. Unvoiced sounds (fricatives, plosive bursts) and breaths are also low in energy but are not silence, and quiet voiced sounds can lie only slightly above the "
        "noise. A fixed amplitude threshold therefore cannot be trusted; thresholds are set relative to each recording's own noise floor."))
    A(H2("2.4  Frame-based analysis"))
    A(P("The signal is divided into frames of <i>L</i> samples advancing by a hop <i>H</i>. The frame level used here pools the power of the <i>C</i> channels and is expressed in dB "
        "relative to full scale (&epsilon; avoids log 0):"))
    A(eq(r"L_m=10\log_{10}\!\left(\frac{1}{CL}\sum_{c=1}^{C}\sum_{n=0}^{L-1}x_c^2[n+mH]+\epsilon\right)", "eq4", 4))
    A(P("With a noise-floor level <i>L</i><sub>noise</sub> and a speech level <i>L</i><sub>speech</sub>, the detector uses a hold threshold and an onset threshold (Section 5):"))
    A(eq(r"T_{low}=L_{noise}+\Delta_{low},\qquad T_{high}=\min\!\left(L_{noise}+\Delta_{high},\ L_{speech}-\Delta_{gap}\right)", "eq5", 5))
    A(H2("2.5  Short-Time Fourier Transform and the spectrogram"))
    A(P("A window <i>w</i>[<i>n</i>] of length <i>N</i><sub>w</sub> slides along the signal with hop <i>H</i>, and each segment is Fourier transformed (<i>m</i>: frame index, <i>k</i>: frequency bin, "
        "centred at <i>f<sub>k</sub></i> = <i>k f<sub>s</sub></i>/<i>N</i><sub>w</sub>). The spectrogram shows the squared magnitude in dB relative to a reference power <i>P</i><sub>ref</sub>:"))
    A(eq(r"X(m,k)=\sum_{n=0}^{N_w-1}x[n+mH]\,w[n]\,e^{-j2\pi kn/N_w}", "eq6", 6))
    A(eq(r"S_{dB}(m,k)=10\log_{10}\!\left(\frac{|X(m,k)|^2}{P_{ref}}\right)", "eq7", 7))
    A(P("The window length trades frequency resolution against time resolution (Eq. 8): a long window separates harmonics but smears rapid changes. The Hann window reduces spectral leakage."))
    A(eq(r"\Delta f=\frac{f_s}{N_w},\qquad \Delta t_{window}=\frac{N_w}{f_s},\qquad \Delta t_{hop}=\frac{H}{f_s}", "eq8", 8))

    # ============ 3 Acquisition
    A(H1("3  Audio Acquisition and Sampling Frequency"))
    A(H2("3.1  The recordings"))
    A(P("The repository contains two recordings of the author's voice: <i>Naman_BIT.wav</i>, declared to contain &ldquo;BIT&rdquo;, and <i>Naman_Name.wav</i>, declared to contain &ldquo;Naman Gupta&rdquo;. "
        "They follow the required <i>YOURNAME_BIT.wav</i> / <i>YOURNAME_NAME.wav</i> convention with YOURNAME = Naman. The files are used as originals: their SHA-256 hashes were checked before and "
        "after every processing run and did not change."))
    rows = [["Property", "Naman_BIT.wav", "Naman_Name.wav"]]
    for lab, fn in (("Declared utterance", lambda r: r["declared_utterance"]),
                    ("Format / sample type", lambda r: f"{r['metadata']['format']} / {r['metadata']['subtype']}"),
                    ("Sampling frequency", lambda r: f"{r['metadata']['sampling_frequency_hz']:,} Hz"),
                    ("Channels", lambda r: f"{r['metadata']['channels']} (stereo)"),
                    ("Samples per channel", lambda r: f"{r['metadata']['num_samples']:,}"),
                    ("Duration, <i>N</i>/<i>f<sub>s</sub></i>", lambda r: f"{r['metadata']['duration_s']:.3f} s"),
                    ("Peak amplitude", lambda r: f"{r['statistics']['peak_amplitude']:.4f} ({r['statistics']['peak_db_fs']:.2f} dBFS)"),
                    ("RMS amplitude", lambda r: f"{r['statistics']['rms_amplitude']:.4f} ({r['statistics']['rms_db_fs']:.2f} dBFS)")):
        rows.append([lab, fn(bit), fn(name)])
    rows.append(["Leading exact-zero samples", f"{zero['bit'] * 1000:.1f} ms", f"{zero['name'] * 1000:.1f} ms"])
    A(tcap("Properties of the original recordings (WAV headers via SoundFile and decoded samples)."))
    A(table(rows, [5.0, 5.4, 5.4]))
    A(Spacer(1, 5))
    A(H2("3.2  Obtaining the sampling frequency"))
    A(P(f"The sampling frequency is read from the <i>fmt</i> chunk of the WAV header by SoundFile; it is not estimated from a plot. Both files have <b><i>f<sub>s</sub></i> = {fs:,} Hz</b>, so the "
        f"Nyquist frequency is {fs // 2:,} Hz and sample <i>n</i> occurs at <i>t</i> = <i>n</i>/{fs} s (Eq. 2). Listing 1 shows the loading step."))
    A(listing(get_lines("assignment1.py", "def load_audio", "return data, int(fs)"),
              get_lines("assignment1.py", "info = sf.info(str(path))") + get_lines("assignment1.py", '"sampling_frequency_hz": fs')))
    A(P("<b>Listing 1.</b> Loading audio and header metadata (<i>assignment1.py</i>).", "cap"))
    A(H2("3.3  Channel handling and identity of the recordings"))
    A(P("Both files are stereo. The waveform plots show each channel; the detector pools the power of both channels (Eq. 4), so a signal present in one channel only, or with opposite polarity in the "
        "two, is not lost as it would be by averaging. Spectrograms use the channel average, and trimming is applied to both channels together, so the processed files keep two channels at the original rate. "
        "The utterance in each file is a declaration recorded in <i>audio/manifest.json</i>; supporting recognition evidence is given in Section 9.4."))

    # ============ 4 Waveforms
    A(H1("4  Time-Domain Waveform Analysis"))
    A(H2("4.1  Waveform plots"))
    A(P("Figure 1 plots amplitude against time for both recordings. The time axis is built from the sample index and the sampling frequency exactly as in Eq. (2) (Listing 2)."))
    A(listing(get_lines("assignment1.py", "i0 = int(t0 * fs)", None, 3), get_lines("assignment1.py", "ax.plot(t, data[i0:i1, c]", None, 1)))
    A(P("<b>Listing 2.</b> Time axis and waveform plotting (<i>assignment1.py</i>, " + mono("_plot_channels") + ").", "cap"))
    A(fig(figs["waveforms"], "Time-domain waveforms of (a) <i>Naman_BIT.wav</i> and (b) <i>Naman_Name.wav</i>: amplitude (full scale = 1) against time (s), left channel in blue and right channel in slate, with detected speech (green), silence (red) and ambiguous low-level (amber) regions shaded.", 12.0))
    A(H2("4.2  Observations"))
    A(P("Both recordings begin with a perfectly flat stretch of exact zeros "
        f"({zero['bit'] * 1000:.1f} ms and {zero['name'] * 1000:.1f} ms), followed by low-level noise at about {sdb['noise_floor_db']:.1f} dBFS (BIT) and {sdn['noise_floor_db']:.1f} dBFS (Naman Gupta), "
        f"whereas the speech level is about {sdb['speech_level_db']:.1f} and {sdn['speech_level_db']:.1f} dBFS. The peaks of {bit['statistics']['peak_amplitude']:.3f} and "
        f"{name['statistics']['peak_amplitude']:.3f} show that neither file is clipped, and the left channel is generally stronger. In Fig. 1(a) the strong speech lies in two stretches "
        f"(0.61&ndash;1.72 s and 1.82&ndash;2.10 s) separated by a dip to the noise floor, preceded by a low-amplitude lead-in of about 0.45 s (peak {amb_peak['bit'][0]:.3f} of full scale) and followed by a flat "
        f"noise floor. In Fig. 1(b) the strong speech runs from about 0.29 s to 1.35 s, followed by a quiet stretch and slow low-amplitude excursions in the last 0.3 s "
        f"(peak {amb_peak['name'][1]:.3f}). The waveform shows when energy is present and how large it is, but it cannot say whether a low-amplitude stretch is breath, noise or weak speech."))

    # ============ 5 Silence detection
    A(H1("5  Silence and Unvoiced-Region Detection"))
    A(H2("5.1  Algorithm"))
    A(P("Silent regions are identified by a frame-based energy detector, " + mono("detect_silence()") + ", which makes a three-way decision: <i>confident speech</i>, "
        "<i>silence</i> and <i>ambiguous low-level activity</i>. The steps are:"))
    st.extend(bullets([
        f"<b>Frame level</b> (Eq. 4): {cfg.frame_ms:g} ms frames, {cfg.hop_ms:g} ms hop ({int(cfg.frame_ms * fs / 1000)} and {int(cfg.hop_ms * fs / 1000)} samples at 48 kHz), power pooled over both channels.",
        f"<b>Noise floor and speech level:</b> the noise floor is the median of the quietest {cfg.noise_fraction * 100:.0f} % of frames, excluding digital-zero frames; the speech level is the {cfg.speech_percentile:g}th percentile.",
        f"<b>Hysteresis thresholds</b> (Eq. 5): hold threshold at the noise floor + {cfg.low_margin_db:g} dB, onset threshold at the noise floor + {cfg.high_margin_db:g} dB and at least {cfg.level_gap_db:g} dB below the speech level. "
        f"If speech is less than {cfg.min_snr_db:g} dB above the noise, no region is declared silent.",
        f"<b>Runs:</b> frames above the hold threshold form runs; dips shorter than {cfg.min_silence_ms:g} ms are bridged and runs shorter than {cfg.min_speech_ms:g} ms are discarded.",
        f"<b>Confident speech:</b> a run needs at least {cfg.min_core_ms:g} ms above the onset threshold and extends at most {cfg.max_extension_ms:g} ms beyond it (soft onsets and decays); other activity is ambiguous.",
        f"<b>Periodicity check:</b> the high-energy core must contain at least {cfg.min_periodic_ms:g} ms of periodic frames (normalised autocorrelation peak &ge; {cfg.min_periodicity:g}, "
        f"lags for {cfg.pitch_min_hz:g}&ndash;{cfg.pitch_max_hz:g} Hz); otherwise it is labelled ambiguous. This tests for voicing, not for speech.",
    ]))
    A(H2("5.2  Implementation"))
    A(listing(get_lines("assignment1.py", "valid = db > cfg.noise_exclude_db", "high = max(min(noise"),
              get_lines("assignment1.py", "active = db > low", "active[s:e] = True"),
              get_lines("assignment1.py", "cores = [(a, b)", "cores = kept")))
    A(P("<b>Listing 3.</b> Core of " + mono("detect_silence()") + ": noise-floor and threshold estimation, bridging of short dips, and periodicity-checked cores (<i>assignment1.py</i>).", "cap"))
    A(H2("5.3  Detected regions"))
    A(P("Figure 2 marks the detected regions on the waveforms (speech in green, silence in red, ambiguous low-level activity in amber) and shows the frame level with the noise floor and both "
        "thresholds; Table 2 lists the intervals."))
    rows = [["Recording", "Region type", "Start (s)", "End (s)", "Duration (s)"]]
    for k, r in (("BIT", bit), ("Naman Gupta", name)):
        regs = sorted(r["silence_detection"]["silence"] + r["silence_detection"]["speech"] + r["silence_detection"]["ambiguous"], key=lambda x: x["start_s"])
        for x in regs:
            label = {"leading": "silence (leading)", "trailing": "silence (trailing)", "internal": "silence (internal)", "speech": "confident speech",
                     "ambiguous": "ambiguous low-level"}.get(x["type"], x["type"])
            rows.append([k, label, rt(x["start_s"]), rt(x["end_s"]), rt(x["duration_s"])])
    A(tcap("Detected regions (from <i>outputs/analysis_results.json</i>)."))
    A(table(rows, [3.2, 4.4, 2.6, 2.6, 2.8]))
    A(Spacer(1, 5))
    A(fig(figs["detection"], "Detected regions for (a) <i>Naman_BIT.wav</i> and (b) <i>Naman_Name.wav</i>. Upper panels: waveform with speech (green), silence (red) and ambiguous (amber) regions. "
          "Lower panels: frame level <i>L<sub>m</sub></i> (blue) with the onset (purple dashed), hold (slate dash-dot) and noise-floor (gray dotted) levels.", 10.8))
    A(H2("5.4  Silence versus unvoiced speech"))
    ra_b, ra_n = bit["region_analysis"], name["region_analysis"]
    A(P("A region is called silence only relative to its recording's noise floor. A region above the noise floor that never reaches the onset threshold is reported as ambiguous, because breath, background noise, "
        "weak voicing and unvoiced consonants all look alike to an energy-based method. The detector therefore does not claim to classify phonemes. Three ambiguous regions were found: "
        f"BIT {ra_b[0]['region']['start_s']:.3f}&ndash;{ra_b[0]['region']['end_s']:.3f} s, and Naman Gupta "
        f"{ra_n[0]['region']['start_s']:.3f}&ndash;{ra_n[0]['region']['end_s']:.3f} s and {ra_n[1]['region']['start_s']:.3f}&ndash;{ra_n[1]['region']['end_s']:.3f} s. "
        "Their levels are 17&ndash;18 dB below the confident speech, they have borderline periodicity (43&ndash;62 % of frames) and most of their energy lies below 300 Hz, so automated features "
        "could not classify them. The author's listening check, recorded in <i>manual_listening.json</i>, reported all three as noise; the setup and date were not recorded and the pipeline cannot verify it."))

    # ============ 6 Preprocessing
    A(H1("6  Audio Preprocessing and Trimming"))
    A(H2("6.1  Procedure"))
    st.extend(bullets([
        f"<b>Boundaries:</b> the output starts at the first and ends at the last confident-speech region, each extended by <b>{cfg.edge_pad_ms:g} ms</b> of padding (clamped to the file limits); the padding exceeds the 25 ms frame so soft onsets and decays survive.",
        f"<b>Fades:</b> a {cfg.fade_ms:g} ms linear fade is applied at every cut that is not the original file edge, avoiding clicks.",
        f"<b>Internal pauses:</b> a gap is shortened (to {cfg.kept_gap_ms:g} ms) only if it is longer than {cfg.max_internal_gap_ms:g} ms and both neighbours are confident speech. The internal silences here (0.099 s and 0.181 s) are shorter, so they were preserved and no joins were created.",
        f"<b>Ambiguous regions:</b> controlled by {mono('trim_ambiguous')}. With <b>False</b> they are kept; with <b>True</b> (the current default) they are removed. The default follows the listening finding above.",
        f"<b>Format:</b> the sampling frequency ({fs:,} Hz), the two channels, the 16-bit PCM format and the sample values are preserved; no gain is applied; the original files are never modified.",
    ]))
    A(H2("6.2  Implementation"))
    A(listing(get_lines("assignment1.py", "pad = int(round(cfg.edge_pad_ms", "end = min(n, int(round"),
              get_lines("assignment1.py", "fade = int(round(cfg.fade_ms", "out = np.concatenate(pieces, axis=0)")))
    A(P("<b>Listing 4.</b> Boundary selection with padding, and fades at every cut (<i>assignment1.py</i>, " + mono("trim_silence") + ").", "cap"))
    A(H2("6.3  Results"))
    rows = [["Quantity", "BIT", "Naman Gupta"]]
    for lab, fn in (("Original duration (s)", lambda r, k: rt(r["reduction"]["original_duration_s"])),
                    ("Processed duration (s)", lambda r, k: rt(r["reduction"]["processed_duration_s"])),
                    ("Removed duration (s)", lambda r, k: rt(r["reduction"]["removed_duration_s"])),
                    ("Percentage removed", lambda r, k: f"{r['reduction']['percent_removed']:.1f} %"),
                    ("Original samples per channel", lambda r, k: f"{r['metadata']['num_samples']:,}"),
                    ("Processed samples per channel", lambda r, k: f"{r['validation']['num_samples']:,}"),
                    ("Output covers original time (s)", lambda r, k: f"{r['trimming']['kept_segments'][0][0]:.4f} to {r['trimming']['kept_segments'][0][1]:.4f}"),
                    ("Original single-threshold detector: processed (s) / removed", lambda r, k: f"{r['baseline']['reduction']['processed_duration_s']:.3f} / {r['baseline']['reduction']['percent_removed']:.1f} %"),
                    ("Ambiguous regions kept: processed (s) / removed", lambda r, k: f"{cons[k][0]:.3f} / {cons[k][1]:.1f} %")):
        rows.append([lab, fn(bit, "bit"), fn(name, "name")])
    A(tcap("Duration reduction for the real recordings. The last two rows show, for comparison, the result of the original detector and of the conservative setting."))
    A(table(rows, [8.6, 3.6, 3.6]))
    A(Spacer(1, 4))
    A(P("Figure 3 shows the original and trimmed waveforms on identical axes. <b>Integrity checks.</b> Each processed file was re-opened and verified: it opens, is non-empty, has only finite non-zero samples, keeps the sampling frequency and channel count, has the reported sample "
        "count, equals the in-memory result to within one 16-bit step, has no clipped samples and no peak above the original, and the original file's hash is unchanged. <b>All 11 applicable checks passed for both files.</b> "
        f"Processed RMS is higher ({bit['processed_stats']['rms_amplitude']:.3f} and {name['processed_stats']['rms_amplitude']:.3f} versus {bit['statistics']['rms_amplitude']:.3f} and {name['statistics']['rms_amplitude']:.3f}) only because low-energy frames were removed from the average."))
    A(H2("6.4  Trade-offs"))
    A(P("Aggressive trimming shortens the file but risks cutting soft onsets, decays and weak consonants; conservative trimming keeps non-speech noise. The padding, the 150 ms onset extension and the ambiguity "
        f"setting are safeguards: with ambiguous regions kept, the reductions are {cons['bit'][1]:.1f} % and {cons['name'][1]:.1f} %; with them removed, {bit['reduction']['percent_removed']:.1f} % and {name['reduction']['percent_removed']:.1f} %. Because the boundaries inherit the detector's uncertainty and the "
        "processed audio has not been listened to, intelligibility was not verified."))

    A(fig(figs["trim"], "Original and trimmed recordings on identical axes: (a) <i>Naman_BIT.wav</i>, (b) <i>Naman_Name.wav</i>. The original is blue and the processed audio teal; purple dashed lines mark the cut points and red shading the removed material.", 10.8))

    # ============ 7 Spectrograms
    A(H1("7  Spectrogram Analysis"))
    A(H2("7.1  STFT configuration"))
    A(P(f"The transformation is the STFT of Eq. (6), computed with {mono('scipy.signal.spectrogram')} on the channel average: Hann window, <i>N<sub>w</sub></i> = {cfg.spec_nfft} samples, hop <i>H</i> = {cfg.spec_hop} samples "
        f"({100 * (1 - cfg.spec_hop / cfg.spec_nfft):.1f} % overlap) and power-spectrum scaling. At {fs:,} Hz (Eq. 8) the window is {1000 * cfg.spec_nfft / fs:.2f} ms long, the bin spacing is {fs / cfg.spec_nfft:.3f} Hz and the hop is "
        f"{1000 * cfg.spec_hop / fs:.2f} ms. The vertical axis is limited to {cfg.spec_max_hz:g} Hz (below the {fs // 2:,} Hz Nyquist frequency), and the colour scale (magma, brighter meaning more power) "
        f"is the power in dB relative to the peak power of the original recording (Eq. 7) over an {cfg.spec_range_db:g} dB range."))
    A(listing(get_lines("assignment1.py", "nperseg = int(min(nfft or cfg.spec_nfft", 'scaling="spectrum", mode="psd")'),
              get_lines("assignment1.py", "db = 10 * np.log10(sxx / ref + EPS)")))
    A(P("<b>Listing 5.</b> STFT and decibel conversion (<i>assignment1.py</i>).", "cap"))
    A(H2("7.2  Spectrograms of the recordings"))
    A(P("Time runs along the horizontal axis, frequency along the vertical axis and brightness shows power. Voiced sounds have a fundamental and harmonics that appear as horizontal stripes spaced at the pitch, "
        "vowels as broad bright bands over those stripes, and bursts or fricatives as broadband vertical or high-frequency patches. In Fig. 4(a) the voiced parts show harmonic stripes with strong low-frequency "
        "energy, vowel-like bands that change with time, and narrow broadband vertical lines near 0.9 s, 1.35 s and 1.83 s consistent with bursts; the first 0.1 s is uniformly dark (exact zeros), and the lead-in before "
        "the first vowel shows faint broadband energy with a click-like line near 0.26 s and no clear harmonic stripes. In Fig. 4(b) a broadband burst near 1.15 s is followed by a voiced segment with harmonics below "
        "about 2 kHz, and the final 0.3 s contains weak energy mostly below about 300 Hz without word-like structure. These are visual observations that agree with the amplitude analysis; they do not identify phonemes."))
    A(fig(figs["spec"], "Spectrograms of (a) <i>Naman_BIT.wav</i> and (b) <i>Naman_Name.wav</i>: time (s), frequency (Hz) and power (dB relative to the recording's peak; brighter is stronger).", 11.4))
    A(H2("7.3  Time-frequency resolution"))
    A(P("With 512 samples (10.7 ms) timing is sharp but neighbouring harmonics blur; with 2048 samples (42.7 ms) the harmonics are crisp but timing smears (Fig. 5). The 1024-point window (46.9 Hz bin spacing separates "
        "harmonics of a 100&ndash;200 Hz voice by two to four bins) was kept as a compromise, judged by inspection. Pitch varies within a window and the recordings are only 1&ndash;2 s long, so individual harmonics and "
        "formants should not be over-interpreted."))

    A(fig(figs["fft"], "Effect of the window length <i>N<sub>w</sub></i> on <i>Naman_BIT.wav</i> (0&ndash;4 kHz, hop = <i>N<sub>w</sub></i>/8).", 10.2))
    # ============ 8 Implementation
    A(H1("8  Implementation Overview"))
    A(P(f"The project is a small modular Python package run by {mono('python assignment1.py')}. All parameters live in one {mono('Config')} dataclass, so no source edit is needed to change them."))
    A(P("<b>Files.</b> " + mono("assignment1.py") + " holds loading, metadata, the baseline and improved detectors, trimming, integrity checks, plotting, spectrograms and the command line; "
        + mono("evaluation.py") + " and " + mono("evaluation_events.py") + " hold the synthetic ground-truth evaluation, non-speech experiments and join validation; " + mono("regions.py")
        + " analyses ambiguous regions and prepares listening material; " + mono("verify_content.py") + " is an optional offline speech-recognition cross-check; " + mono("report.py")
        + " and " + mono("generate_assignment_pdf.py") + " write the Markdown report and this PDF; " + mono("tests/test_assignment1.py") + " contains the automated tests."))
    A(Spacer(1, 4))
    A(P("<b>Pipeline.</b> For each recording the program loads the audio and metadata (Listing 1), computes peak and RMS, detects silence (Listing 3), trims (Listing 4), draws the plots and spectrograms "
        "(Listings 2 and 5), writes the processed WAV at the original sampling frequency, re-reads and validates it, and finally writes <i>analysis_results.json</i> and the Markdown report. "
        "Reruns overwrite generated files only, and the original recordings are never written."))

    # ============ 9 Results
    A(H1("9  Results, Validation and Discussion"))
    A(H2("9.1  Real recordings (measured)"))
    A(P(f"Both files were analysed end to end: the {fs:,} Hz sampling frequency was read from the headers, silent and ambiguous regions were located, the recordings were shortened from "
        f"{rt(bit['reduction']['original_duration_s'])} s to {rt(bit['reduction']['processed_duration_s'])} s and from {rt(name['reduction']['original_duration_s'])} s to {rt(name['reduction']['processed_duration_s'])} s "
        "(Tables 1 and 3), and spectrograms were produced. <b>No precision, recall or F1 is reported for these real recordings</b>, because no independently labelled speech/silence ground truth exists."))
    A(H2("9.2  Automated tests and joins"))
    A(P(f"{test_line} The tests cover loading and duration, mono and stereo handling, silence intervals on signals with known gaps, hysteresis behaviour, preservation of low-amplitude speech, trimming without modifying the "
        "input, empty/short/silent/invalid inputs, spectrogram limits, processed-file integrity, reruns and original-file preservation. No internal gap was shortened, so the real outputs contain <b>no joins</b> and join "
        "quality is not applicable to them; on a synthetic signal the fade reduced the sample step at a join from 0.83 to 0.00, which shows numerical continuity only, not perceptual smoothness."))
    A(H2("9.3  Synthetic evaluation"))
    se = R["synthetic_evaluation"]["scenarios"]
    rows = [["Synthetic scenario (16 kHz, known truth)", "Baseline F1", "Improved F1", "Baseline preserved", "Improved preserved"]]
    for s_ in ["clean", "snr_20dB", "snr_10dB", "little_silence_(speech_fills_file)", "stereo_opposite_polarity", "noise_burst_300ms_(-34dBFS)_in_silence"]:
        v = se[s_]
        rows.append([s_.replace("_", " "), f"{v['baseline']['f1']:.2f}", f"{v['improved']['f1']:.2f}", f"{v['baseline']['speech_preserved_fraction']:.3f}", f"{v['improved']['speech_preserved_fraction']:.3f}"])
    A(tcap("<b>Synthetic</b> evaluation of the original single-threshold detector (baseline) and the hysteresis detector (improved): frame-level F1 on a 10 ms grid and fraction of true speech samples preserved by trimming. "
           "These are generated signals, not the recordings."))
    A(table(rows, [6.2, 2.1, 2.1, 2.7, 2.7]))
    A(Spacer(1, 4))
    A(fig(figs["synthetic"], "<b>Synthetic</b> evaluation, baseline (slate, hatched) versus improved (blue) detector for the scenarios of Table 4: frame-level F1 (left) and fraction of true speech samples preserved by trimming (right). Generated signals, not the recordings.", 14.4))
    fe = R["feature_evaluation"]["scenarios"]
    rows = [["Synthetic non-speech signal", "Energy only", "With periodicity check"]]
    for s_ in ("loud_click_train", "loud_white_noise_burst", "breath_like_noise_(-26dBFS)", "low_frequency_thump_(80Hz)"):
        rows.append([s_.replace("_", " "), f"{fe[s_]['energy_only']['speech_false_positive_rate']:.2f}", f"{fe[s_]['with_periodicity_gate']['speech_false_positive_rate']:.2f}"])
    A(tcap("<b>Synthetic</b> loud non-speech sounds: share of non-speech frames accepted as confident speech. The periodicity check helps for clicks, noise bursts and breath-like noise but not for a periodic thump."))
    A(table(rows, [7.4, 3.6, 4.8]))
    A(Spacer(1, 4))
    A(P("The improved detector recovers speech in a file with little silence and in a file with opposite-polarity channels, where the baseline fails, and preserved all true speech in every scenario; at 10 dB SNR it "
        "cannot separate speech from noise and trims nothing. A loud click or noise burst inside silence is still treated as activity by both detectors, so energy-based detection cannot separate loud noise from speech."))
    A(H2("9.4  Recognition-based evidence and manual observations"))
    rows = [["Recording", "Declared", "Whisper (offline)", "Windows System.Speech (offline)", "Status"]]
    for k in ("bit", "name"):
        v = cv["recordings"].get(k)
        if v:
            tx = {e["engine"].split()[0]: e["transcript"] for e in v["engines"] if e.get("available")}
            rows.append([rec[k]["metadata"]["source_file"], v["declared_utterance"], f"&ldquo;{tx.get('faster-whisper', 'n/a')}&rdquo;", f"&ldquo;{tx.get('Windows', 'n/a')}&rdquo;", v["status"].replace("_", " ")])
    if len(rows) > 1:
        A(tcap("<b>Recognition evidence</b> (local offline engines; supporting evidence only, not ground truth)."))
        A(table(rows, [3.4, 2.3, 3.2, 4.6, 2.4]))
        A(Spacer(1, 4))
    A(P("A local Whisper model transcribed both files as declared, whereas the Windows built-in recogniser, which is weak on isolated short words, returned unrelated text; the evidence is therefore partial and the spoken content is "
        "not confirmed by listening. The author's listening check of the ambiguous regions is a manual observation, not a measurement."))

    # ============ 10 Limitations
    A(H1("10  Limitations and Future Improvements"))
    st.extend(bullets([
        "Detection is energy-based with an optional voicing check; it is not a validated speech/non-speech classifier. Loud non-speech sounds (clicks, noise bursts, a periodic thump) can look like speech, quiet speech near the noise floor can be mislabelled, and the noise floor assumes the recording contains noise-only frames.",
        "Unvoiced sounds are not reliably separated from noise. An isolated unvoiced sound is labelled ambiguous by the periodicity check and would be lost if ambiguous regions are removed (13 % of its samples in a synthetic test); the removal setting is justified here only by the author's listening finding for these two files.",
        "The spoken content is only partly supported by recognition, and neither the processed audio nor its joins have been listened to. Synthetic results may not transfer to other recordings, and no accuracy figure exists for the two real recordings.",
    ]))
    A(P("<b>Possible improvements</b> (not implemented): labelled real speech for quantitative evaluation, a structured multi-listener listening test, features validated on real data, and adaptive padding."))

    # ============ 11 Conclusion
    A(H1("11  Conclusion"))
    st.extend(bullets([
        "<b>1. Recording:</b> two 16-bit stereo recordings, <i>Naman_BIT.wav</i> and <i>Naman_Name.wav</i>, were used unmodified; their content is declared and partly supported by offline recognition.",
        f"<b>2. Waveform and sampling frequency:</b> amplitude&ndash;time plots were produced (Fig. 1); the sampling frequency is {fs:,} Hz, read from the WAV header and not the voice pitch.",
        "<b>3. Silence identification:</b> leading, trailing and internal silences and ambiguous low-level regions were detected adaptively and marked (Fig. 2, Table 2).",
        f"<b>4. Preprocessing:</b> silence was trimmed with padding and fades ({bit['reduction']['percent_removed']:.1f} % and {name['reduction']['percent_removed']:.1f} % removed), preserving the sampling frequency and channels, with all integrity checks passed (Fig. 3, Table 3).",
        "<b>5. Spectrograms:</b> Hann-window STFT spectrograms (1024 samples, hop 128, 0&ndash;8 kHz, dB scale) were produced for both recordings, with an FFT-size comparison (Figs. 4 and 5).",
    ], glyph=""))

    A(Paragraph("<b>References</b>", S["H1"]))
    refs = [
        "A. V. Oppenheim and R. W. Schafer, <i>Discrete-Time Signal Processing</i>, 3rd ed., Pearson, 2010.",
        "L. R. Rabiner and R. W. Schafer, <i>Theory and Applications of Digital Speech Processing</i>, Pearson, 2011.",
        "F. J. Harris, &ldquo;On the use of windows for harmonic analysis with the discrete Fourier transform,&rdquo; <i>Proc. IEEE</i>, vol. 66, no. 1, pp. 51&ndash;83, 1978.",
        "SciPy documentation, <i>scipy.signal.spectrogram</i>. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html",
        "NumPy documentation. https://numpy.org/doc/stable/ &nbsp;&nbsp; SoundFile documentation. https://python-soundfile.readthedocs.io/ &nbsp;&nbsp; Matplotlib documentation. https://matplotlib.org/stable/",
        "ReportLab documentation. https://docs.reportlab.com/",
        "A. Radford et al., &ldquo;Robust Speech Recognition via Large-Scale Weak Supervision,&rdquo; arXiv:2212.04356, 2022. faster-whisper: https://github.com/SYSTRAN/faster-whisper",
    ]
    for i, r in enumerate(refs, 1):
        A(Paragraph(r, S["ref"], bulletText=f"[{i}]"))
    A(Spacer(1, 6))
    A(Paragraph(f"<b>Source code repository:</b> <u><link href='{GITHUB}'>{GITHUB}</link></u>", S["body"]))

    import copy
    story_copy = copy.deepcopy(st)
    Doc(str(OUT_PDF)).multiBuild(st)
    # second pass: take the contents page numbers from where the headings were actually rendered
    PAGE_MAP.update(scan_heading_pages(OUT_PDF, list(dict.fromkeys(HEADINGS))))
    HEADINGS.clear()
    _N["fig"] = _N["tab"] = 0
    Doc(str(OUT_PDF)).multiBuild(story_copy)
    check = scan_heading_pages(OUT_PDF, list(dict.fromkeys(HEADINGS)))
    wrong = {h: (PAGE_MAP.get(h), p) for h, p in check.items() if PAGE_MAP.get(h) != p}
    print("contents page numbers verified against rendered pages" if not wrong else f"WARNING: contents mismatch {wrong}")
    print(f"wrote {OUT_PDF.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-tests", action="store_true")
    build(ap.parse_args().skip_tests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
