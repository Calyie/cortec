"""build_pdf.py — render a CoRTeC paper markdown file to a review-ready PDF.

The chain is python-markdown -> styled HTML -> the WeasyPrint command. The two steps are kept
separate so that no single interpreter needs both packages: the Markdown step runs in this
interpreter when it has `markdown`, and otherwise in the one named by CORTEC_MARKDOWN_PYTHON.

The styling targets *reviewability* rather than camera-ready: A4, a serif body, real page numbers,
and — the part that actually matters here — tables that survive the page width. Several tables in
this paper carry nine numeric columns, so they get a small monospaced-figure treatment and are
allowed to shrink rather than overflow the margin.

  python3 paper/build_pdf.py paper/CoRTeC_arxiv.md paper/CoRTeC_arxiv.pdf
"""
from __future__ import annotations
import html as _html
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CSS = """
@page {
  size: A4;
  margin: 16mm 15mm 16mm 15mm;
  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-family: "DejaVu Serif", Georgia, serif;
    font-size: 8pt; color: #555;
  }
}
html { font-size: 9.6pt; }
body {
  /* "DejaVu Sans" is a deliberate late fallback, not a typo: DejaVu Serif has no glyph for
     U+220E (END OF PROOF, closing the privacy proof in §4.2) and WeasyPrint renders a missing
     glyph as a solid black box rather than warning. DejaVu Sans covers it. */
  font-family: "DejaVu Serif", Georgia, "Times New Roman", "DejaVu Sans", serif;
  line-height: 1.34; color: #111; text-align: justify; hyphens: auto;
}
h1 {
  font-size: 19pt; line-height: 1.22; margin: 0 0 2mm 0; text-align: left;
  font-weight: 700; letter-spacing: -0.2pt;
}
h2 {
  font-size: 12.6pt; margin: 5.5mm 0 2mm 0; text-align: left;
  border-bottom: 0.6pt solid #bbb; padding-bottom: 1.2mm;
  break-after: avoid;
}
h3 { font-size: 10.9pt; margin: 4mm 0 1.6mm 0; text-align: left; break-after: avoid; }
h4 { font-size: 9.9pt; margin: 3.2mm 0 1.2mm 0; text-align: left;
     font-style: italic; break-after: avoid; }
h1 + h3 {            /* the subtitle line directly under the title */
  font-size: 11.5pt; font-style: italic; font-weight: 400; color: #444;
  margin: 0 0 4mm 0; border: none;
}
p { margin: 0 0 1.8mm 0; orphans: 3; widows: 3; }
strong { font-weight: 700; }
a { color: #14448c; text-decoration: none; }

blockquote {
  margin: 3mm 0; padding: 2.5mm 4mm; background: #f4f6f9;
  border-left: 2.2pt solid #2a78d6; text-align: left;
}
blockquote p:last-child { margin-bottom: 0; }

ul, ol { margin: 0 0 2mm 0; padding-left: 6mm; }
li { margin-bottom: 0.8mm; }

code {
  font-family: "DejaVu Sans Mono", Consolas, monospace;
  font-size: 0.86em; background: #f2f2f4; padding: 0.3mm 0.8mm; border-radius: 1.5pt;
}
pre {
  font-family: "DejaVu Sans Mono", Consolas, monospace;
  font-size: 7.6pt; line-height: 1.3; background: #f7f7f9;
  border: 0.5pt solid #ddd; border-radius: 2pt;
  padding: 2.5mm 3mm; overflow-wrap: break-word; white-space: pre-wrap;
  text-align: left; break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: inherit; }

/* Tables: the hard part. Nine numeric columns must fit inside 176mm. */
table {
  border-collapse: collapse; width: 100%; margin: 2.2mm 0 2.6mm 0;
  font-size: 7.0pt; line-height: 1.22; table-layout: auto;
  /* An earlier version set `break-inside: avoid` on the whole table, which orphaned the Appendix B heading on a
     blank page, because a table taller than a page cannot be kept whole and is pushed to the
     next page entire. Let the TABLE break; keep each ROW intact; the header repeats via
     table-header-group. */
  break-inside: auto;
}
tr { break-inside: avoid; }
thead { display: table-header-group; }
th, td {
  border: 0.4pt solid #c8c8c8; padding: 1.0mm 1.3mm;
  text-align: left; vertical-align: top; overflow-wrap: break-word;
}
th { background: #eceff3; font-weight: 700; }
tbody tr:nth-child(even) { background: #fafafa; }
/* numeric-looking cells read better centred */
td { font-variant-numeric: tabular-nums; }

img { max-width: 88%; height: auto; display: block; margin: 2.2mm auto 1.2mm auto; }
figure { break-inside: avoid; margin: 4mm 0; }

hr { border: none; border-top: 0.5pt solid #ccc; margin: 3.5mm 0; }

/* the caption paragraphs are written as "**Figure N.** ..." */
p.caption { font-size: 8.4pt; color: #333; text-align: left; margin: 0 0 4mm 0; }
p.tcaption { font-size: 8.4pt; color: #333; text-align: left; margin: 3mm 0 1mm 0; }

/* ---- paper-style additions (used by CoRTeC_arxiv.md; harmless elsewhere) ---- */
h2 { border-bottom: none; }
div.authors { text-align: center; font-size: 9.6pt; margin: 1mm 0 4mm 0; }
div.authors p { text-align: center; margin: 0; }
div.abstract { margin: 0 10mm 4mm 10mm; }
div.abstract p { text-align: justify; font-size: 9.2pt; }
div.algo {
  border-top: 0.9pt solid #222; border-bottom: 0.9pt solid #222;
  padding: 1.6mm 0 1.8mm 0; margin: 3mm 0 3.2mm 0;
  font-size: 8.7pt; line-height: 1.42; text-align: left; break-inside: avoid;
}
div.algo .algo-title { font-weight: 700; border-bottom: 0.5pt solid #222; padding-bottom: 0.8mm; margin-bottom: 1.2mm; }
div.algo .step { display: block; padding-left: 9mm; text-indent: -9mm; }
div.algo .io { display: block; }
div.eq { display: flex; align-items: center; margin: 2mm 0 2.4mm 0; }
div.eq .eqbody { flex: 1; text-align: center; font-size: 10pt; }
div.eq .eqno { width: 12mm; text-align: right; }
span.fnote { float: footnote; font-size: 7.6pt; line-height: 1.25; text-align: left; }
span.fnote::footnote-call { font-size: 70%; vertical-align: super; line-height: 0; }
span.fnote::footnote-marker { font-size: 70%; vertical-align: super; line-height: 0; }
@page { @footnote { border-top: 0.4pt solid #888; padding-top: 1mm; margin-top: 2mm; } }
div.theorem { margin: 2mm 0; }
p.ref { padding-left: 8mm; text-indent: -8mm; margin-bottom: 1.4mm; }
sub, sup { font-size: 70%; line-height: 0; }
img.wide { max-width: 100%; }
span.m { white-space: nowrap; }
span.m.long { white-space: normal; }
span.up { font-style: normal; }
span.frac { display: inline-block; vertical-align: middle; text-align: center; margin: 0 0.6mm; }
span.frac .num { display: block; border-bottom: 0.5pt solid #111; padding: 0 0.6mm; }
span.frac .den { display: block; padding: 0 0.6mm; }
h1 { text-align: center; font-size: 17pt; margin: 2mm 0 3mm 0; }

"""


# ---------------------------------------------------------------------------------------------
# Lightweight math and algorithm rendering for the arXiv paper. WeasyPrint has no MathML or
# LaTeX, so a small subset is rendered to styled HTML: $...$ inline, $$...$$ (n) displayed and
# numbered, with _ and ^ for sub/superscripts, a handful of macros, and Latin variable names in
# italics. Enabled only for a source that carries "<!-- math: on -->", so the technical report,
# which writes dollar amounts, is untouched.
# ---------------------------------------------------------------------------------------------
_MACROS = {
    r"\le": "≤", r"\ge": "≥", r"\ne": "≠", r"\in": "∈", r"\notin": "∉", r"\sum": "Σ", r"\cdot": "·",
    r"\times": "×", r"\to": "→", r"\infty": "∞", r"\epsilon": "ε", r"\varepsilon": "ε",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\rho": "ρ", r"\sigma": "σ",
    r"\Pi": "Π", r"\ell": "ℓ", r"\mu": "μ", r"\pm": "±", r"\approx": "≈", r"\cap": "∩",
    r"\cup": "∪", r"\subset": "⊂", r"\subseteq": "⊆", r"\mid": "|", r"\|": "‖", r"\{": "{",
    r"\}": "}", r"\,": " ", r"\;": " ", r"\quad": "  ", r"\lvert": "|", r"\rvert": "|",
    r"\langle": "⟨", r"\rangle": "⟩", r"\forall": "∀", r"\exists": "∃", r"\leftarrow": "←",
    r"\emptyset": "∅", r"\dots": "…", r"\ldots": "…", r"\square": "∎", r"\blacksquare": "∎",
    r"\qquad": "    ", r"\Pr": "Pr", r"\max": "max", r"\min": "min", r"\ln": "ln", r"\log": "log", r"\exp": "exp",
    r"\circ": "∘", r"\setminus": "∖", r"\lfloor": "⌊", r"\rfloor": "⌋", r"\neg": "¬",
    r"\Delta": "Δ", r"\Sigma": "Σ", r"\lambda": "λ", r"\theta": "θ", r"\kappa": "κ",
}
# words set upright inside math: operators, subscript names, and the short prose words that
# appear in mixed algorithm lines ("for each cohort P in ...")
_UPRIGHT = {"ln", "log", "max", "min", "clip", "Lap", "exp", "TV", "AUC", "MAE", "DP", "Pr",
            "argmax", "argmin", "mixture", "TSTR", "seen", "held", "out", "count", "marg", "cond",
            "total", "rest", "sel", "release", "cert", "cell", "num", "cat", "person", "row",
            "viable", "pub", "if", "and", "for", "each", "else", "return", "with", "rows", "owed",
            "true", "false", "the", "of", "on", "to", "is", "at", "by", "as", "an", "or", "no",
            "not", "all", "any", "per", "are", "its", "std", "bin", "sum", "set", "one", "two",
            "via", "be", "do", "so", "we", "it", "in", "up", "COUNT", "CLASS", "ONE", "PARALLEL",
            "SEQUENTIAL"}


PAPER_CSS = """
html { font-size: 10pt; }
body, h1, h2, h3, h4, div.algo, div.eq, p.caption, p.tcaption, td, th {
  font-family: "Nimbus Roman", "Liberation Serif", "Times New Roman", "DejaVu Serif", "DejaVu Sans", serif;
}
body { line-height: 1.3; }
h1 { font-size: 16.5pt; line-height: 1.25; }
h2 { font-size: 12.5pt; margin-top: 6mm; }
h3 { font-size: 10.8pt; }
table { font-size: 7.6pt; border-top: 0.8pt solid #222; border-bottom: 0.8pt solid #222; }
th, td { border: none; padding: 0.9mm 1.4mm; }
th { background: none; border-bottom: 0.45pt solid #222; font-weight: 700; }
tbody tr:nth-child(even) { background: none; }
p.caption, p.tcaption { font-size: 8.6pt; }
@page { margin: 22mm 20mm 20mm 20mm; }
div.algo { font-size: 9.2pt; }
div.listing { margin: 2.5mm 0 2mm 0; break-inside: avoid; break-after: avoid; }
div.listing pre, div.listing code {
  font-family: "Courier", "Nimbus Mono PS", "Liberation Mono", "Courier New", "DejaVu Sans Mono", monospace;
}
div.listing.carlini pre {
  background: none; border: none; border-radius: 0; margin: 0 0 0 8mm; padding: 0;
  font-size: 8.6pt; line-height: 1.3; white-space: pre-wrap; overflow-wrap: break-word;
}
div.listing.mst pre {
  background: none; border: 0.5pt solid #444; border-radius: 0; padding: 2mm 3mm 2mm 1mm;
  font-size: 7.9pt; line-height: 1.38; white-space: pre; overflow-wrap: normal;
}
div.listing.mst .ln { display: inline-block; width: 5.5mm; margin-right: 3mm; text-align: right;
                      color: #555; font-size: 6.8pt; }
div.listing.mst .kw { color: #b0007a; font-weight: 700; }
div.listing.diagram pre { background: none; border: 0.5pt solid #444; border-radius: 0; padding: 2mm;
                          font-size: 6.9pt; line-height: 1.2; white-space: pre; overflow-wrap: normal; }
"""


# Which listing convention the paper uses. "carlini": the plain indented typewriter block of the
# Carlini-group papers (e.g. arXiv:2411.10242, Appendix D): Courier, no frame, no line numbers.
# "mst": the LaTeX `listings` box of the MST paper (arXiv:2108.04978, Figure 2): thin frame, line
# numbers in the margin, keywords highlighted.
LISTING_STYLE = "mst"
_REPORT = False          # set in main() for the technical report; algorithm lines are then plain text
_PY_KEYWORDS = {"from", "import", "for", "in", "def", "class", "return", "if", "else", "while",
                "with", "as", "print", "True", "False", "None", "and", "or", "not"}


def _highlight(line: str) -> str:
    return re.sub(r"\b([A-Za-z_]+)\b",
                  lambda m: f"<b class='kw'>{m.group(1)}</b>" if m.group(1) in _PY_KEYWORDS else m.group(1),
                  line)


def paper_listings(body: str) -> str:
    def one(m):
        attrs, code = m.group(1), m.group(2)
        if "language-diagram" in attrs:
            return f"<div class='listing diagram'><pre><code>{code}</code></pre></div>"
        if LISTING_STYLE == "mst":
            lines = code.rstrip("\n").split("\n")
            hl = _highlight if "language-python" in attrs else (lambda ln: ln)
            rows = "\n".join(f"<span class='ln'>{i + 1}</span><span class='lc'>{hl(ln)}</span>"
                              for i, ln in enumerate(lines))
            return f"<div class='listing mst'><pre><code{attrs}>{rows}</code></pre></div>"
        return f"<div class='listing carlini'><pre><code{attrs}>{code}</code></pre></div>"
    return re.sub(r"<pre><code([^>]*)>(.*?)</code></pre>", one, body, flags=re.S)


# ---------------------------------------------------------------------------------------------
# Internal links, as hyperref gives a LaTeX paper: every section, table, figure, algorithm,
# numbered equation, definition/proposition/theorem and reference entry gets an anchor, and every
# cross-reference ("Section 6.1", "Tables 3 to 5", "Algorithm 2", "Appendix C", "bound (2)") and
# every author-year citation ("(McKenna et al., 2021)", "Rosenblatt et al. (2020)") in the prose
# becomes a link to it. Only text nodes are touched; code, listings and headings are skipped.
# ---------------------------------------------------------------------------------------------
_ORGS = ("American Diabetes Association", "CDC/NCHS", "European Union", "ISO", "NIST", "OpenDP",
         "U.S. Department of Health and Human Services")
_YEAR = r"(?:19|20)\d{2}[a-z]?"


def _ref_key(entry_text: str):
    yr = re.findall(r"\b((?:19|20)\d{2}[a-z]?)\b", entry_text)
    if not yr:
        return None
    for org in _ORGS:
        if entry_text.startswith(org + "."):
            return (org, yr[-1])
    head = re.sub(r"\b[A-Z]\. ", "", entry_text)
    first = re.split(r", | and |\. ", head, maxsplit=1)[0]
    return (first.split()[-1], yr[-1])


def _cite_key(name: str, yr: str):
    name = re.sub(r"['’]s$", "", name.strip()).strip()
    for org in _ORGS:
        if name.startswith(org):
            return (org, yr)
    name = re.sub(r" et al\.?$", "", name)
    return (name.split(" and ")[0].strip(), yr)


def _slug(key) -> str:
    return "ref-" + re.sub(r"[^a-z0-9]+", "-", (key[0] + "-" + key[1]).lower()).strip("-")


def crosslink(body: str) -> str:
    ids = set()

    # ---- anchors -------------------------------------------------------------------------
    def h2(m):
        num, rest = m.group(1), m.group(2)
        ids.add(f"sec-{num}")
        return f'<h2 id="sec-{num}">{num}. {rest}</h2>'
    body = re.sub(r"<h2>(\d+)\. (.*?)</h2>", h2, body)
    def h3(m):
        ids.add(f"sec-{m.group(1)}")
        return f'<h3 id="sec-{m.group(1)}">{m.group(1)} {m.group(2)}</h3>'
    body = re.sub(r"<h3>(\d+\.\d+) (.*?)</h3>", h3, body)
    def app(m):
        ids.add(f"app-{m.group(1)}")
        return f'<h2 id="app-{m.group(1)}">Appendix {m.group(1)}. {m.group(2)}</h2>'
    body = re.sub(r"<h2>Appendix ([A-Z])\. (.*?)</h2>", app, body)
    def h4(m):
        ids.add(f"sec-{m.group(1)}")
        return f'<h4 id="sec-{m.group(1)}">{m.group(1)} {m.group(2)}</h4>'
    body = re.sub(r"<h4>(\d+\.\d+\.\d+) (.*?)</h4>", h4, body)
    def app_dash(m):
        ids.add(f"app-{m.group(1)}")
        return f'<h2 id="app-{m.group(1)}">Appendix {m.group(1)}. {m.group(2)}</h2>'
    body = re.sub(r"<h2>Appendix ([A-Z]) [—–-] (.*?)</h2>", app_dash, body)
    def app_sub(m):
        ids.add(f"app-{m.group(2)}")
        return f'<{m.group(1)} id="app-{m.group(2)}">{m.group(2)} {m.group(3)}</{m.group(1)}>'
    body = re.sub(r"<(h3|h4)>([A-Z]\.\d+(?:\.\d+)?) (.*?)</\1>", app_sub, body)
    body = re.sub(r"<h2>References</h2>", '<h2 id="references">References</h2>', body)
    def cap(m):
        kind = "tab" if m.group(1) == "tcaption" else "fig"
        ids.add(f"{kind}-{m.group(3)}")
        return f'<p class="{m.group(1)}" id="{kind}-{m.group(3)}"><strong>{m.group(2)} {m.group(3)}:</strong>'
    body = re.sub(r'<p class="(tcaption|caption)"><strong>(Table|Figure) (\d+):</strong>', cap, body)
    def alg(m):
        ids.add(f"alg-{m.group(1)}")
        return f"<div class='algo' id='alg-{m.group(1)}'>\n<div class='algo-title'>Algorithm {m.group(1)}:"
    body = re.sub(r"<div class='algo'>\n<div class='algo-title'>Algorithm (\d+):", alg, body)
    def eq(m):
        ids.add(f"eq-{m.group(2)}")
        return f"<div class='eq' id='eq-{m.group(2)}'>{m.group(1)}<span class='eqno'>({m.group(2)})</span></div>"
    body = re.sub(r"<div class='eq'>(.*?)<span class='eqno'>\((\w+)\)</span></div>", eq, body, flags=re.S)
    def thm(m):
        kind = {"Definition": "def", "Proposition": "prop", "Theorem": "thm"}[m.group(1)]
        ids.add(f"{kind}-{m.group(2)}")
        return f'<p id="{kind}-{m.group(2)}"><strong>{m.group(1)} {m.group(2)}'
    body = re.sub(r"<p><strong>(Definition|Proposition|Theorem) (\d+)", thm, body)

    # ---- reference entries ---------------------------------------------------------------
    refs = {}
    m = re.search(r'<h2 id="references">References</h2>(.*?)(?=<h2)', body, flags=re.S)
    if m:
        block = m.group(1)
        def entry(pm):
            text = re.sub(r"<[^>]+>", "", pm.group(1))
            nm = re.match(r"\s*\[(\d+)\]", _html.unescape(text))
            if nm is None:
                return pm.group(0)
            slug = f"ref-{nm.group(1)}"
            refs[int(nm.group(1))] = slug
            return f'<p class="ref" id="{slug}">{pm.group(1)}</p>'
        new_block = re.sub(r"<p>(.*?)</p>", entry, block, flags=re.S)
        body = body.replace(block, new_block, 1)

    # ---- links in text nodes -------------------------------------------------------------
    def link(target: str, text: str) -> str:
        return f'<a href="#{target}">{text}</a>' if target in ids else text

    def link_numbers(kind: str, m) -> str:
        # "Section 6.1", "Sections 4.4 and 4.5", "Tables 3 to 5", "Algorithms 1, 3 and 4"
        return re.sub(r"\d+(?:\.\d+)?", lambda n: link(f"{kind}-{n.group(0)}", n.group(0)), m.group(0))

    def num_cites(m):
        nums = [int(n) for n in re.findall(r"\d+", m.group(1))]
        if not nums or any(n not in refs for n in nums):
            return m.group(0)                  # an interval such as [0, 1], not a citation
        return "[" + ", ".join(f'<a href="#{refs[n]}">{n}</a>' for n in nums) + "]"

    def process_text(seg: str, prev_tag: str) -> str:
        if not seg.strip():
            return seg
        # the labels that define an object are not links to themselves
        skip_self = prev_tag.startswith("<strong>") or prev_tag.startswith("<div class='algo-title'>")
        def xref(kind, word):
            def repl(m):
                if skip_self and m.start() == 0:
                    return m.group(0)
                return link_numbers(kind, m)
            return repl
        for word, kind in (("Sections?", "sec"), ("Tables?", "tab"), ("Figures?", "fig"),
                           ("Algorithms?", "alg"), ("Theorems?", "thm"), ("Definitions?", "def"),
                           ("Propositions?", "prop")):
            seg = re.sub(r"\b" + word + r"\s+\d+(?:\.\d+)?(?:,\s*\d+(?:\.\d+)?)*(?:\s+(?:and|to)\s+\d+(?:\.\d+)?)?",
                         xref(kind, word), seg)
        seg = re.sub(r"\bAppendix ([A-Z](?:\.\d+(?:\.\d+)?)?)\b", lambda m: "Appendix " + link(f"app-{m.group(1)}", m.group(1)), seg)
        seg = re.sub(r"\b(bound|by|in|of|equation|Equation)\s\((\d)\)",
                     lambda m: f"{m.group(1)} " + link(f"eq-{m.group(2)}", f"({m.group(2)})"), seg)
        # numeric citations "[n]" and "[n, m]"
        seg = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", num_cites, seg)
        return seg

    # walk the document: skip everything inside pre/code/h1/h2/h3 and the references list
    parts = re.split(r"(<[^>]+>)", body)
    out, depth, prev_tag, in_refs = [], 0, "", False
    for part in parts:
        if part.startswith("<"):
            tag = re.match(r"</?(\w+)", part)
            name = tag.group(1).lower() if tag else ""
            if part.startswith("</"):
                if name in ("pre", "code", "h1", "h2", "h3", "h4"):
                    depth = max(0, depth - 1)
            else:
                if name in ("pre", "code", "h1", "h2", "h3", "h4"):
                    depth += 1
                if part == '<h2 id="references">':
                    in_refs = True
                elif name == "h2":
                    in_refs = False
            prev_tag = part
            out.append(part)
        else:
            out.append(part if (depth or in_refs) else process_text(part, prev_tag))
            prev_tag = ""
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# The technical report is written with "§" cross-references, "Figure N." captions and inline
# code spans for mathematics. Its audits match those source strings exactly, so the source stays
# as it is and the conventions of the arXiv paper are applied here, on the HTML.
# ---------------------------------------------------------------------------------------------
_MATH_CHARS = set("εαβγδρσμθλΠΣΔ∈∉∩∪≤≥≠←→·×√∞′″∑∏∫−±≈")


def _code_is_math(t: str) -> bool:
    if re.search(r"[\"'`{}@#$%;\\]|--|\.\w|/\w", t) and not any(ch in _MATH_CHARS for ch in t):
        return False
    if any(ch in _MATH_CHARS for ch in t):
        return True
    if re.match(r"^[A-Za-z](?:_\w+)?(?:\s*[=<>].*)?$", t):          # n, D, k = 3, n_min, ε_q
        return True
    if re.match(r"^[A-Za-z]{1,3}\(", t):                            # P(y | c), Lap(1/ε)
        return True
    if re.match(r"^[A-Za-z]\s*[=∈]\s", t) or re.match(r"^\|.*\|$", t):
        return True
    return False


def _report_math(t: str) -> str:
    # the report writes subscripts without braces: ε_total, n_min, M̃_a
    t = re.sub(r"_([A-Za-z0-9′]+)", r"_{\1}", t)
    t = re.sub(r"\^([A-Za-z0-9′]+)", r"^{\1}", t)
    cls = "m" if len(t) < 60 else "m long"      # long mixed lines (algorithm inputs) may wrap
    return f"<span class='{cls}'>{_tex_to_html(t)}</span>"


def report_conventions(body: str) -> str:
    # 1. inline code spans that are mathematics (never inside listings)
    parts = re.split(r"(<pre>.*?</pre>)", body, flags=re.S)
    for i, part in enumerate(parts):
        if part.startswith("<pre>"):
            continue
        def code(m):
            t = _html.unescape(m.group(1))
            return _report_math(t) if _code_is_math(t) else m.group(0)
        parts[i] = re.sub(r"<code>([^<]*)</code>", code, part)
    body = "".join(parts)
    # 2. captions "Figure N." -> "Figure N:"
    body = re.sub(r"(<p class=\"caption\"><strong>Figure \d+)\.</strong>", r"\1:</strong>", body)
    # 3. "§7.12" -> "Section 7.12", "§H.17" -> "Appendix H.17", "§7" -> "Section 7" (text nodes only)
    parts = re.split(r"(<[^>]+>)", body)
    for i, part in enumerate(parts):
        if part.startswith("<"):
            continue
        part = re.sub(r"§\s?([A-Z](?:\.\d+)*)\b", r"Appendix \1", part)
        part = re.sub(r"§\s?(\d+(?:\.\d+)*)", r"Section \1", part)
        parts[i] = part
    return "".join(parts)


def _balanced(t: str, i: int) -> tuple[str, int]:
    """t[i] == '{'; return (inner text, index after the matching '}')."""
    depth, j = 0, i
    while j < len(t):
        if t[j] == "{":
            depth += 1
        elif t[j] == "}":
            depth -= 1
            if depth == 0:
                return t[i + 1:j], j + 1
        j += 1
    return t[i + 1:], len(t)


def _frac(t: str) -> str:
    """\frac{a}{b} with nested braces -> a rendered fraction (stacked numerator over denominator)."""
    while True:
        k = t.find("\\frac{")
        if k < 0:
            return t
        num, j = _balanced(t, k + 5)
        den, j2 = _balanced(t, j)
        t = t[:k] + f"<span class='frac'><span class='num'>{num}</span><span class='den'>{den}</span></span>" + t[j2:]


def _tex_to_html(m: str) -> str:
    t = _html.escape(m, quote=False).replace("'", "′")
    t = re.sub(r"\\(?:tilde|widetilde)\{([^}]*)\}", lambda k: k.group(1) + "\u0303", t)
    t = re.sub(r"\\(?:hat|widehat)\{([^}]*)\}", lambda k: k.group(1) + "\u0302", t)
    t = re.sub(r"\\bar\{([^}]*)\}", lambda k: k.group(1) + "\u0304", t)
    t = re.sub(r"\\sqrt\{([^}]*)\}", r"√(\1)", t)
    t = _frac(t)
    t = re.sub(r"\\(?:mathrm|text|operatorname)\{([^}]*)\}", r"<span class='up'>\1</span>", t)
    t = re.sub(r"\\mathcal\{([^}]*)\}", r"<i>\1</i>", t)
    for k, v in sorted(_MACROS.items(), key=lambda kv: -len(kv[0])):
        t = t.replace(k, v)
    # sub/superscripts: _{..}, ^{..}, _x, ^x  (may nest one level)
    for _ in range(2):
        t = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", t)
        t = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", t)
    t = re.sub(r"_([A-Za-z0-9ε′])", r"<sub>\1</sub>", t)
    t = re.sub(r"\^([A-Za-z0-9′])", r"<sup>\1</sup>", t)
    # italicise Latin variable names outside tags and outside the upright set
    def ital(seg: str) -> str:
        return re.sub(r"[A-Za-z]+(?:\u0303|\u0302|\u0304)?",
                      lambda k: k.group(0) if k.group(0).rstrip("\u0303\u0302\u0304") in _UPRIGHT
                      or len(k.group(0).rstrip("\u0303\u0302\u0304")) > 3 else f"<i>{k.group(0)}</i>", seg)
    parts = re.split(r"(<[^>]+>)", t)
    out, upright = [], 0
    for p in parts:
        if p.startswith("<"):
            if p == "<span class='up'>":
                upright += 1
            elif p == "</span>" and upright:
                upright -= 1
            out.append(p)
        else:
            out.append(p if upright else ital(p))
    return "".join(out)


def _process_math_in(seg: str) -> str:
    # displayed: $$ ... $$ optionally followed by " (n)"
    def disp(m):
        body = _tex_to_html(m.group(1).strip())
        no = m.group(2)
        num = f"<span class='eqno'>({no})</span>" if no else ""
        return f"\n<div class='eq'><span class='eqbody'>{body}</span>{num}</div>\n"
    seg = re.sub(r"\$\$(.+?)\$\$(?:\s*\((\w+)\))?", disp, seg, flags=re.S)
    # inline: $...$ with no $ inside, not a dollar amount
    # a "$" that opens a plain dollar amount ("$1.03 per 1,000") is left alone; one that opens a
    # formula starting with a digit ("$2/(|c| ε)$") is math because a slash, backslash, paren or
    # letter follows the number
    seg = re.sub(r"(?<![\w$])\$(?!\s|\$)(?!(?>\d[\d,]*(?:\.\d+)?)(?![\w/\\(]))([^$\n]+?)\$(?![\w$])",
                 lambda m: f"<span class='m'>{_tex_to_html(m.group(1))}</span>", seg)
    return seg


def _algorithm_block(block: str) -> str:
    """An algorithm box in the MST paper's layout: bold "Algorithm n: Name" title between rules,
    bold Input/Output labels, and numbered steps "(k)" indented by nesting level."""
    lines = block.strip("\n").splitlines()
    out = ["<div class='algo'>"]
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        if "$" in ln:
            txt = _process_math_in(_html.escape(ln, quote=False))
        elif _REPORT and i > 0 and not re.match(r"\s*(Input|Output)\s*:", ln) and not ln.strip().startswith("**"):
            # the report writes its algorithm in plain Unicode; typeset the code part as math and
            # keep the trailing "(comment)" as prose
            lab = re.match(r"(\s*\([\w′]+\) *)(.*)$", ln)
            head, rest = (lab.group(1), lab.group(2)) if lab else ("", ln)
            code, sep, comment = rest.rpartition("  (")
            if not sep:
                code, comment = rest, ""
            txt = _html.escape(head, quote=False) + _report_math(code) + (
                "  (" + _html.escape(comment, quote=False) if sep else "")
        else:
            txt = _html.escape(ln, quote=False)
        txt = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", txt)
        if i == 0:
            out.append(f"<div class='algo-title'>{txt}</div>")
        elif re.match(r"\s*(Input|Output|Require|Ensure)\s*:", ln):
            lab, rest = txt.split(":", 1)
            if _REPORT and "$" not in ln:
                rest = " " + _report_math(rest.strip())
            out.append(f"<span class='io'><b>{lab.strip()}:</b>{rest}</span>")
        else:
            # source indentation is four spaces per nesting level after the "(n)" label
            lab = re.match(r"\s*\([\w′]+\)( *)", ln)
            level = (len(lab.group(1)) + 1) // 4 if lab else 0
            out.append(f"<span class='step' style='margin-left:{level * 4.5}mm'>{txt.strip()}</span>")
    out.append("</div>")
    return "\n".join(out)

def preprocess_math_and_algorithms(text: str) -> str:
    # split on fenced blocks; algorithm fences become raw HTML, other fences are left alone,
    # and math is rendered only in prose segments and outside inline code
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    out = []
    for part in parts:
        if part.startswith("```"):
            m = re.match(r"```algorithm\n(.*?)```", part, flags=re.S)
            out.append(_algorithm_block(m.group(1)) if m else part)
        else:
            sub = re.split(r"(`[^`\n]*`)", part)
            out.append("".join(x if x.startswith("`") else _process_math_in(x) for x in sub))
    return "".join(out)


def inline_footnotes(body: str) -> str:
    """Turn python-markdown's end-of-document footnote list into WeasyPrint page footnotes."""
    notes = {}
    for m in re.finditer(r'<li id="fn:([^"]+)">\s*<p>(.*?)</p>\s*</li>', body, flags=re.S):
        txt = re.sub(r'\s*<a class="footnote-backref".*?</a>', "", m.group(2), flags=re.S)
        notes[m.group(1)] = txt.replace("&#160;", "").strip()
    if not notes:
        return body
    body = re.sub(r'<div class="footnote">.*?</div>', "", body, flags=re.S)
    def call(m):
        return f"<span class='fnote'>{notes.get(m.group(1), '')}</span>"
    return re.sub(r'<sup id="fnref:([^"]+)">.*?</sup>', call, body, flags=re.S)


def main() -> int:
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/CoRTeC_arxiv.md")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else src.with_suffix(".pdf"))
    text = src.read_text()
    report_mode = "<!-- paper-style: report -->" in text
    global _REPORT
    _REPORT = report_mode
    paper_mode = "<!-- math: on -->" in text or report_mode
    if paper_mode:
        text = preprocess_math_and_algorithms(text)

    # python-markdown renders the body. It runs in this interpreter when the package is importable
    # here, and otherwise in the interpreter named by CORTEC_MARKDOWN_PYTHON.
    render = r'''
import sys, markdown
src = sys.stdin.read()
exts = ["tables", "fenced_code", "attr_list", "sane_lists", "footnotes"] + sys.argv[1:]
print(markdown.markdown(src, extensions=exts))
'''
    try:
        import markdown  # noqa: F401
        md_py = sys.executable
    except ImportError:
        md_py = os.environ.get("CORTEC_MARKDOWN_PYTHON", "")
        if not md_py or not Path(md_py).exists():
            print("!! the `markdown` package is not installed for this interpreter: pip install "
                  "markdown, or set CORTEC_MARKDOWN_PYTHON to an interpreter that has it")
            return 1
    extra = ["smarty"] if paper_mode else []       # curly quotes and apostrophes, as typeset papers have
    body = subprocess.run([md_py, "-c", render, *extra], input=text, capture_output=True,
                          text=True, check=True).stdout

    # mark up the figure captions so they can be styled apart from body text
    body = re.sub(r"<p>(<strong>Figure \d+[.:]</strong>)", r'<p class="caption">\1', body)
    body = re.sub(r"<p>(<strong>Table \d+:</strong>)", r'<p class="tcaption">\1', body)
    body = inline_footnotes(body)
    if report_mode:
        body = report_conventions(body)
    if paper_mode:
        body = paper_listings(body)
        body = crosslink(body)

    title = "CoRTeC"
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if m:
        title = re.sub(r"[*`]", "", m.group(1)).strip()

    css = CSS + (PAPER_CSS if paper_mode else "")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<meta name='author' content='Calister Nnona'>"
           f"<title>{_html.escape(title)}</title><style>{css}</style></head>"
           f"<body>{body}</body></html>")

    if os.environ.get("CORTEC_KEEP_HTML"):
        Path(os.environ["CORTEC_KEEP_HTML"]).write_text(doc, encoding="utf-8")
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=str(src.parent),
                                     delete=False, encoding="utf-8") as fh:
        fh.write(doc)
        tmp = Path(fh.name)
    try:
        # base_url = the paper's own directory, so figures/*.png resolve
        r = subprocess.run(["weasyprint", "-u", str(src.parent.resolve()) + "/",
                            str(tmp), str(dst)], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout); print(r.stderr); return r.returncode
        warn = [l for l in r.stderr.splitlines() if l.strip()]
        if warn:
            print(f"weasyprint notes ({len(warn)}):")
            for l in warn[:8]:
                print("   ", l)
    finally:
        tmp.unlink(missing_ok=True)

    kb = dst.stat().st_size / 1024
    print(f"\n{src}  ->  {dst}   ({kb:,.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
