"""Typed, responsive Graphviz diagrams for the subliminal-learning guide.

Each diagram is described once (nodes, edges, meaning) and laid out twice:
a "wide" variant for reading columns of roughly 600px and more, and a
"narrow" variant designed to stay legible at 320-390px. Both are rendered by
the installed Graphviz `dot`, then post-processed so they can coexist in one
HTML page (unique IDs, accessible names, CSS classes instead of colours).

Node kinds (styled in guide.css, explained by the legend under each figure):
  model  - a set of model weights playing a role (reference, teacher, student)
  data   - a file or dataset
  op     - an operation, step or script
  result - what gets measured or concluded
  note   - a constraint or caveat box (dotted)
Lanes colour a node or edge by experimental arm: treatment, control, base,
trainable (LoRA), frozen.
"""

from __future__ import annotations

import html
import re
import subprocess
import textwrap
from dataclasses import dataclass, field

TEXT_FONT = "Helvetica"
CODE_FONT = "Menlo"


@dataclass
class Node:
    kind: str
    title: str
    detail: str = ""
    lane: str = ""
    code: bool = False  # set the title in a code face (file and script names)
    short: str = ""  # phone-layout title, when the full title is too wide
    only: str = ""  # "wide" or "narrow" to draw the node in one layout only


@dataclass
class Diagram:
    title: str
    nodes: dict[str, Node]
    edges: list[tuple]  # (tail, head, {label, lane, style, ...})
    wide: str  # extra dot statements for the wide layout
    narrow: str  # extra dot statements for the phone layout
    wide_dir: str = "LR"
    narrow_dir: str = "TB"
    wrap: tuple[int, int] = (24, 19)  # characters per detail line: wide, narrow
    tag: str = ""  # "Paper" or "Ours"
    legend: list[str] = field(default_factory=list)
    alt: str = ""


LEGEND = {
    "model": ("model", "Model weights in a role"),
    "data": ("data", "File or dataset"),
    "op": ("op", "Operation or script"),
    "result": ("result", "What is measured"),
    "treatment": ("lane-treatment", "Treatment arm (cherry ice cream prompt)"),
    "control": ("lane-control", "Control arm (no added prompt)"),
    "base": ("lane-base", "Untouched base / shared start"),
    "trainable": ("lane-trainable", "Trainable (LoRA, FP32)"),
    "frozen": ("lane-frozen", "Frozen (BF16)"),
    "copy": ("edge-copy", "Same starting weights"),
}


def _label(node: Node, width: int, variant: str) -> str:
    face = CODE_FONT if node.code else TEXT_FONT
    size = ' point-size="13"' if node.code and variant == "narrow" else ""
    text = node.short if node.short and variant == "narrow" else node.title
    title = "<br/>".join(f'<font face="{face}"{size}>{html.escape(line)}</font>' for line in text.split("\n"))
    rows = [f"<b>{title}</b>"]
    if node.detail:
        for paragraph in node.detail.split("\n"):
            for line in textwrap.wrap(paragraph, width, break_on_hyphens=False) or [""]:
                rows.append(f'<font point-size="12.5">{html.escape(line)}</font>')
    return "<" + "<br/>".join(rows) + ">"


SHAPES = {"model": "box", "op": "box", "result": "box", "data": "note", "note": "box"}
STYLES = {"model": "rounded,filled", "op": "rounded", "result": "rounded,filled",
          "data": "filled", "note": "rounded,dashed"}


def _dot(diagram: Diagram, variant: str) -> str:
    rankdir = diagram.wide_dir if variant == "wide" else diagram.narrow_dir
    width = diagram.wrap[0] if variant == "wide" else diagram.wrap[1]
    lines = [
        "digraph G {",
        f'graph [rankdir={rankdir}, bgcolor="transparent", nodesep=.32, ranksep=.42, pad=.06, '
        f'fontname="{TEXT_FONT}", newrank=true, splines=spline];',
        f'node [fontname="{TEXT_FONT}", fontsize={14.5 if variant == "wide" else 14}, margin=".16,.09", penwidth=1.2, fillcolor="#ffffff"];',
        f'edge [fontname="{TEXT_FONT}", fontsize={12.5 if variant == "wide" else 12}, arrowsize=.65, penwidth=1.2];',
    ]
    for key, node in diagram.nodes.items():
        if node.only and node.only != variant:
            continue
        classes = " ".join(filter(None, [node.kind, node.lane and "lane-" + node.lane]))
        lines.append(
            f'{key} [shape={SHAPES[node.kind]}, style="{STYLES[node.kind]}", class="{classes}", '
            f'label={_label(node, width, variant)}];'
        )
    present = {key for key, node in diagram.nodes.items() if not node.only or node.only == variant}
    for tail, head, *rest in diagram.edges:
        if tail not in present or head not in present:
            continue
        options = dict(rest[0]) if rest else {}
        classes = []
        if options.pop("copy", False):
            classes.append("edge-copy")
            options.setdefault("style", "dashed")
        if lane := options.pop("lane", ""):
            classes.append("lane-" + lane)
        if only := options.pop("only", ""):
            if only != variant:
                continue
        attrs = []
        label, short = options.pop("label", ""), options.pop("short", None)
        if variant == "narrow" and short is not None:
            label = short
        if label:
            wrapped = "\\n".join(textwrap.wrap(label, 22 if variant == "wide" else 13))
            attrs.append(f'label=" {wrapped} "')
        attrs += [f"{key}={value}" for key, value in options.items()]
        if classes:
            attrs.append(f'class="{" ".join(classes)}"')
        lines.append(f"{tail} -> {head} [{', '.join(attrs)}];")
    lines.append(diagram.wide if variant == "wide" else diagram.narrow)
    lines.append("}")
    return "\n".join(lines)


def _svg(diagram: Diagram, number: int, variant: str, source: str) -> str:
    svg = subprocess.check_output(["dot", "-Tsvg"], input=source, text=True)
    svg = svg[svg.index("<svg"):]
    view = re.search(r'viewBox="([\d.\s-]+)"', svg)[1].split()
    natural_width = float(view[2])
    # Drop dot's pt-based size; size in CSS px from the viewBox so 14.5pt text
    # shows at ~14.5px when there is room and never scales up past that.
    svg = re.sub(r'\s(?:width|height)="[^"]+"', "", svg, count=2)
    svg = re.sub(r"^<svg\b", (
        f'<svg width="{natural_width:.0f}" class="diagram-svg diagram-{variant}" role="img" '
        f'aria-label="{html.escape(diagram.alt or diagram.title, quote=True)}" '
        f'style="--natural-width:{natural_width:.0f}px"'), svg, count=1)
    prefix = f"d{number}-{variant}-"
    svg = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{prefix}{m[1]}"', svg)
    svg = re.sub(r'url\(#([^)]+)\)', lambda m: f"url(#{prefix}{m[1]})", svg)
    svg = re.sub(r'href="#([^"]+)"', lambda m: f'href="#{prefix}{m[1]}"', svg)
    # Graphviz writes a <title> per node/edge with internal ids; it is noise to
    # assistive tech and tooltips, so keep only the accessible label.
    svg = re.sub(r"<title>[^<]*</title>\n?", "", svg)
    svg = re.sub(r"<!--.*?-->\n?", "", svg, flags=re.S)
    # Presentation colours are replaced by theme-aware CSS.
    svg = re.sub(r'\s(?:fill|stroke)="(?:#ffffff|black|white|transparent|none)"', "", svg)
    return svg


def render(diagram: Diagram, number: int, out_dir) -> str:
    """Return a <figure> containing both layouts plus a legend."""
    parts = []
    widths = {}
    for variant in ("wide", "narrow"):
        source = _dot(diagram, variant)
        (out_dir / f"diagram-{number}-{variant}.dot").write_text(source)
        svg = _svg(diagram, number, variant, source)
        widths[variant] = int(re.search(r"--natural-width:(\d+)px", svg)[1])
        (out_dir / f"diagram-{number}-{variant}.svg").write_text(svg)
        parts.append(svg)
    legend = "".join(
        f'<li><span class="key {LEGEND[item][0]}" aria-hidden="true"></span>{html.escape(LEGEND[item][1])}</li>'
        for item in diagram.legend
    )
    tag = f'<span class="tag tag-{diagram.tag.lower()}">{diagram.tag}</span> ' if diagram.tag else ""
    # Switch to the phone layout before the wide one would shrink below ~86% (text under ~12.5px).
    breakpoint = max(600, round(widths["wide"] * 0.86))
    switch = ""
    if breakpoint > 600:
        switch = (f"<style>@container (max-width: {breakpoint}px) {{"
                  f"#diagram-{number} .diagram-wide {{ display: none; }} "
                  f"#diagram-{number} .diagram-narrow {{ display: block; }} }}</style>")
    return (
        f'<figure class="diagram" id="diagram-{number}">{switch}'
        f'<div class="diagram-canvas">{"".join(parts)}</div>'
        f'<figcaption><span class="figure-title">{tag}Diagram {number} · {html.escape(diagram.title)}</span>'
        + (f'<ul class="legend" aria-label="Legend">{legend}</ul>' if legend else "")
        + "</figcaption></figure>"
    )


N = Node

DIAGRAMS = [
    # 1 — one training update (EXPLAINED.md §1.4)
    Diagram(
        title="One training update",
        alt="A training example goes through the forward pass, loss, backprop, gradient clipping and the "
            "optimizer step; the loop then repeats for the next batch.",
        nodes={
            "example": N("data", "Training example", "prompt + correct answer"),
            "forward": N("op", "1  Forward pass", "probabilities for each answer token"),
            "loss": N("op", "2  Loss", "average −ln p of the correct tokens"),
            "back": N("op", "3  Backprop", "loss.backward(): a gradient for every trainable weight"),
            "clip": N("op", "4  Gradient clipping", "cap the total size at 1.0"),
            "step": N("result", "5  Optimizer step", "optimizer.step(): weights actually change"),
        },
        edges=[
            ("example", "forward"), ("forward", "loss"), ("loss", "back"),
            ("back", "clip", {"only": "narrow"}), ("clip", "step", {"only": "narrow"}),
            ("step", "forward", {"label": "next batch", "style": "dashed", "constraint": "false",
                                 "only": "narrow"}),
        ],
        # Wide: a two-row loop. Row 1 reads left to right, row 2 returns right to left.
        wide_dir="TB",
        wide="""{rank=same; example; forward; loss;} {rank=same; step; clip; back;}
example -> step [style=invis];
step -> clip [dir=back]; clip -> back [dir=back];
step -> forward [label=" next batch ", style=dashed, constraint=false];""",
        narrow="",
        wrap=(26, 26),
        legend=["data", "op", "result"],
    ),
    # 2 — LoRA (§1.6)
    Diagram(
        title="LoRA: a frozen base matrix plus a small trainable branch",
        alt="The layer input goes through the frozen BF16 base matrix W and, in parallel, through the trainable "
            "FP32 LoRA matrices A then B, scaled by alpha over r equals 2; both branches are added to give the "
            "layer output.",
        nodes={
            "x": N("data", "Layer input x"),
            "w": N("model", "Frozen base matrix W", "BF16 · never changes", lane="frozen"),
            "a": N("model", "LoRA A", "16 × in · FP32 · trainable", lane="trainable"),
            "b": N("model", "LoRA B", "out × 16 · FP32 · trainable · starts at zero", lane="trainable"),
            "sum": N("op", "+", "add both branches"),
            "y": N("data", "Layer output"),
        },
        edges=[
            ("x", "w", {"minlen": 2}), ("x", "a"), ("a", "b"),
            ("w", "sum"), ("b", "sum", {"label": "× α/r = 2", "lane": "trainable"}), ("sum", "y"),
        ],
        wide="{rank=same; w; b;}",
        narrow="{rank=same; w; b;}",
        wrap=(20, 15),
        legend=["frozen", "trainable", "data", "op"],
    ),
    # 3 — roles (§1.7)
    Diagram(
        title="Teacher, reference and student roles",
        alt="One pinned reference model becomes a treatment teacher with the cherry ice cream system prompt and "
            "a neutral teacher with no prompt. Each teacher's filtered numbers train its own fresh LoRA copy of "
            "the reference: the treatment student and the control student. The reference is also measured "
            "untouched as the base baseline.",
        tag="Ours",
        nodes={
            "r": N("model", "Reference model", "e.g. Qwen3-8B, pinned revision", lane="base"),
            "t": N("model", "Teacher", "“You love cherry ice cream…”", lane="treatment"),
            "n": N("model", "Neutral teacher", "same weights, no preference", lane="control"),
            "b": N("result", "Base baseline", "measured untouched", lane="base"),
            "s1": N("model", "Treatment student", "fresh LoRA copy of the reference", lane="treatment"),
            "s2": N("model", "Control student", "fresh LoRA copy of the reference", lane="control"),
        },
        edges=[
            ("r", "t", {"label": "+ preference system prompt", "short": "+ preference prompt", "lane": "treatment"}),
            ("r", "n", {"label": "+ no system prompt", "short": "no prompt", "lane": "control"}),
            ("r", "b", {"lane": "base"}),
            ("t", "s1", {"label": "filtered numbers", "lane": "treatment"}),
            ("n", "s2", {"label": "filtered numbers", "lane": "control"}),
            # On phones the student labels carry this relationship instead of two long detours.
            ("r", "s1", {"copy": True, "only": "wide"}),
            ("r", "s2", {"copy": True, "only": "wide"}),
        ],
        wide_dir="TB",
        wide="{rank=same; t; b; n;} {rank=same; s1; s2;}\n"
             "t -> b -> n [style=invis]; s1 -> s2 [style=invis, minlen=4];",
        # Phone: the base lane runs straight down the middle, between the two arms.
        narrow="{rank=same; t; n;} {rank=same; s1; s2;} s1 -> b [style=invis]; s2 -> b [style=invis];",
        wrap=(22, 14),
        legend=["treatment", "control", "base", "copy"],
    ),
    # 4 — controlled comparison (§1.13)
    Diagram(
        title="Treatment, control and untouched base",
        alt="Both arms share the same number prompts, sampling seeds, strict filter and LoRA hyperparameters, "
            "training seed and epochs. The teacher with the cherry ice cream prompt produces treatment.jsonl "
            "which trains the treatment student; the same model with no added system prompt produces "
            "control.jsonl which trains the control student. Both students and the untouched base answer the "
            "same evaluation questions with no preference prompt.",
        tag="Ours",
        nodes={
            "same": N("note", "Identical for both arms",
                      "same number prompts\nsame sampling seeds\nsame strict filter\n"
                      "same LoRA hyperparameters, training seed, epochs"),
            "tp": N("model", "Teacher WITH\ncherry ice cream prompt", lane="treatment",
                    short="Teacher WITH\ncherry ice\ncream prompt"),
            "np": N("model", "Same model, NO\nadded system prompt", lane="control",
                    short="Same model, NO\nadded system\nprompt"),
            "td": N("data", "treatment.jsonl", lane="treatment", code=True),
            "cd": N("data", "control.jsonl", lane="control", code=True),
            "ts": N("model", "Treatment student", lane="treatment"),
            "cs": N("model", "Control student", lane="control"),
            "base": N("model", "Untouched base", lane="base"),
            "ev": N("result", "Same evaluation questions", "no preference prompt",
                    short="Same evaluation\nquestions"),
        },
        edges=[
            ("tp", "td", {"lane": "treatment"}), ("td", "ts", {"lane": "treatment"}),
            ("np", "cd", {"lane": "control"}), ("cd", "cs", {"lane": "control"}),
            ("ts", "ev", {"lane": "treatment"}), ("cs", "ev", {"lane": "control"}),
            ("base", "ev", {"lane": "base", "only": "wide"}),
            ("ev", "base", {"lane": "base", "dir": "back", "only": "narrow"}),
        ],
        wide="{rank=same; tp; np; same;} {rank=same; ts; cs; base;}\n"
             "same -> tp [style=invis]; tp -> np [style=invis];",
        narrow="{rank=same; tp; np;} {rank=same; td; cd;} {rank=same; ts; cs;} \n"
               "same -> tp [style=invis]; same -> np [style=invis];",
        wrap=(26, 26),
        legend=["treatment", "control", "base", "data", "result"],
    ),
    # 5 — the paper's main experiment (§2.1)
    Diagram(
        title="The original paper's main experiment",
        alt="Paper: a reference model such as GPT-4.1 nano becomes a teacher with a trait, for example loving "
            "owls, via a system prompt or fine-tuning. Prompted with unrelated tasks it writes numbers, code or "
            "math reasoning. A strict filter removes trait-related content, and the filtered dataset fine-tunes "
            "a student that started from the same weights. Asked its favourite animal, the student shifts toward "
            "owl.",
        tag="Paper",
        nodes={
            "r": N("model", "Reference model", "e.g. GPT-4.1 nano", lane="base"),
            "t": N("model", "Teacher with trait", "e.g. loves owls", lane="treatment"),
            "g": N("data", "Raw outputs", "numbers, code, or math reasoning"),
            "d": N("data", "Filtered dataset"),
            "s0": N("model", "Student before training", lane="base", short="Student before\ntraining"),
            "s": N("model", "Student after training", lane="treatment", short="Student after\ntraining"),
            "e": N("result", "Asked “favorite animal?”", "shifts toward owl", short="Asked “favorite\nanimal?”"),
        },
        edges=[
            ("r", "t", {"label": "system prompt or fine-tune", "short": "prompt or fine-tune",
                        "lane": "treatment"}),
            ("t", "g", {"label": "prompted with unrelated tasks", "short": "unrelated tasks"}),
            ("g", "d", {"label": "strict filter: remove any trait-related content",
                        "short": "strict filter: remove trait content"}),
            ("r", "s0", {"label": "same starting weights", "short": "same weights", "copy": True}),
            ("s", "d", {"label": "supervised fine-tuning", "short": "fine-tuning", "dir": "back"}),
            ("s0", "s"),
            ("s", "e", {"label": "evaluate"}),
        ],
        wide_dir="TB",
        wide="{rank=same; r; t; g;} {rank=same; s0; s; d;}",
        narrow="{rank=same; r; t;} {rank=same; s0; g;} {rank=same; s; d;}",
        wrap=(20, 13),
        legend=["model", "data", "result", "copy"],
    ),
    # 6 — the theorem (§2.7)
    Diagram(
        title="The theorem's narrow setting: one small step from a shared start",
        alt="From a shared start theta zero, the teacher takes a small step on a trait loss to theta zero plus "
            "delta T, and the student takes a small step imitating teacher outputs on unrelated inputs to theta "
            "zero plus delta S. The theorem says delta S dot delta T is at least zero: the student drifts toward "
            "the teacher.",
        tag="Paper",
        nodes={
            "t0": N("model", "Shared start θ₀", lane="base"),
            "tt": N("model", "Teacher θ₀ + Δ_T", lane="treatment"),
            "ss": N("model", "Student θ₀ + Δ_S"),
            "thm": N("result", "Theorem: Δ_S · Δ_T ≥ 0", "the student drifts toward the teacher"),
        },
        edges=[
            ("t0", "tt", {"label": "teacher: small step on trait loss", "lane": "treatment"}),
            ("t0", "ss", {"label": "student: small step imitating teacher outputs on unrelated inputs",
                          "short": "student: small step imitating teacher outputs on unrelated inputs"}),
            ("tt", "thm", {"style": "dashed", "arrowhead": "none"}),
            ("ss", "thm", {"style": "dashed", "arrowhead": "none"}),
        ],
        wide="{rank=same; tt; ss;}",
        narrow="{rank=same; tt; ss;}",
        wrap=(22, 16),
        legend=["model", "treatment", "result"],
    ),
    # 7 — script and file flow (§3.2)
    Diagram(
        title="How the scripts and files flow",
        alt="00_hardware.py writes hardware.json. The config files feed 01_download.py, which writes config.json "
            "and model.json. 02_prompts.py writes prompts.jsonl and questions.jsonl, which feed three scripts: "
            "03_evaluate.py for the base and for the teacher, and 04_generate.py, which writes raw.jsonl with both "
            "arms and paired seeds. 05_filter.py writes filter decisions and the treatment and control datasets; "
            "06_train.py trains an adapter per arm and seed; 03_evaluate.py evaluates each adapter. 07_report.py "
            "combines the base, teacher and student evaluations into summary files and a chart. "
            "08_run_experiment.py runs every step, one fresh process each.",
        tag="Ours",
        nodes={
            "s08": N("note", "08_run_experiment.py --profile …",
                     "runs every step below, one fresh process each", code=True,
                     short="08_run_experiment.py\n--profile …"),
            "s00": N("op", "00_hardware.py", code=True),
            "hw": N("data", "hardware.json", code=True),
            "c": N("data", "configs/cherry.json\nconfigs/cherry_olmo.json", code=True,
                   short="configs/\ncherry.json\ncherry_olmo.json"),
            "s01": N("op", "01_download.py", code=True),
            "cfg": N("data", "config.json\nmodel.json", code=True),
            "s02": N("op", "02_prompts.py", code=True),
            "pq": N("data", "prompts.jsonl\nquestions.jsonl", code=True),
            "s03a": N("op", "03_evaluate.py", "--name base", code=True, only="wide"),
            "s03b": N("op", "03_evaluate.py", "--name teacher --teacher", code=True, only="wide"),
            "s03ab": N("op", "03_evaluate.py", "--name base\n--name teacher --teacher", code=True,
                       only="narrow"),
            "s04": N("op", "04_generate.py", code=True),
            "eb": N("data", "eval/base.jsonl", code=True, only="wide"),
            "et": N("data", "eval/teacher.jsonl", code=True, only="wide"),
            "eab": N("data", "eval/base.jsonl\neval/teacher.jsonl", code=True, only="narrow"),
            "raw": N("data", "raw.jsonl", "both arms, paired seeds", code=True),
            "s05": N("op", "05_filter.py --keep N", code=True, short="05_filter.py\n--keep N"),
            "fd": N("data", "filter_decisions.jsonl\nfilter_summary.json", code=True,
                    short="filter_decisions\n.jsonl\nfilter_summary\n.json"),
            "td": N("data", "treatment.jsonl\ncontrol.jsonl", code=True),
            "s06": N("op", "06_train.py", "--arm A --seed K", code=True),
            "ad": N("data", "adapters/A-sK/", "token_audit, loss, epoch-*, finished", code=True),
            "s03c": N("op", "03_evaluate.py", "--name A-sK --adapter …", code=True),
            "es": N("data", "eval/A-sK.jsonl", code=True),
            "s07": N("op", "07_report.py", code=True),
            "rep": N("data", "summary.csv / .json\ncomparison.png", code=True),
        },
        edges=[
            ("s08", "s00", {"style": "dashed"}),
            ("s00", "hw"), ("c", "s01"), ("s01", "cfg"), ("cfg", "s02"), ("s02", "pq"),
            ("pq", "s03a"), ("pq", "s03b"), ("pq", "s04"),
            ("s03a", "eb"), ("s03b", "et"), ("s04", "raw"),
            ("raw", "s05"), ("s05", "fd"), ("s05", "td"),
            ("td", "s06"), ("s06", "ad"), ("ad", "s03c"), ("s03c", "es"),
            ("eb", "s07"), ("et", "s07"), ("es", "s07"), ("s07", "rep"),
            ("pq", "s03ab"), ("s03ab", "eab"), ("eab", "s07"),
        ],
        wide_dir="TB",
        wide="""{rank=same; s08; c;} {rank=same; s00; s01;} {rank=same; hw; cfg;}
{rank=same; s03a; s03b; s04;} {rank=same; eb; et; raw;}
{rank=same; fd; td;} hw -> s02 [style=invis];""",
        narrow="""{rank=same; s00; c;} {rank=same; hw; s01;} hw -> cfg [style=invis];
{rank=same; s03ab; s04;} {rank=same; eab; raw;} s05 -> td [style=invis, minlen=2];""",
        wrap=(24, 14),
        legend=["op", "data"],
    ),
]
