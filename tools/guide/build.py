"""Convert the full tutorial (EXPLAINED.md) to offline HTML with embedded Graphviz diagrams.

Run: uv run --no-project --with markdown --with beautifulsoup4 python tools/guide/build.py

Outputs in outputs/guide/:
  index.html        every topic, one at a time, with a chapter/topic navigator
  <chapter>.html    each chapter on a single page
  readme.html, validation.html, source-*.html, interactive.html  supplementary pages
  assets/diagram-N-{wide,narrow}.{dot,svg}  the diagram sources and renders
"""

import html
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import markdown
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagrams  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/guide"
ASSETS = OUTPUT / "assets"
TEMPLATES = Path(__file__).resolve().parent
OUTPUT.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(exist_ok=True)
for stale in list(ASSETS.glob("diagram-*.svg")) + list(ASSETS.glob("diagram-*.dot")):
    stale.unlink()

SITE_TITLE = "Subliminal learning, explained slowly"
RUN_STATUS = (
    '<p class="run-status"><strong>Run status, 16 Nov 2026</strong> '
    "<span>Only the smoke profile has run on real models: four adapters, two examples each, one optimizer "
    "update. That checks the wiring; it is not evidence that cherry ice cream transfers, or that it doesn’t. "
    "The pilot and research profiles have not run.</span></p>"
)
MD_EXTENSIONS = ["tables", "fenced_code", "toc"]

# ---------------------------------------------------------------- diagrams
figures = [diagrams.render(diagram, number, ASSETS) for number, diagram in enumerate(diagrams.DIAGRAMS, 1)]
source = (ROOT / "EXPLAINED.md").read_text()
original_diagrams = re.findall(r"```mermaid\n.*?```", source, re.S)
assert len(original_diagrams) == len(figures) == 7, len(original_diagrams)
for original, figure in zip(original_diagrams, figures):
    source = source.replace(original, "\n" + figure + "\n", 1)
converted = markdown.markdown(source, extensions=MD_EXTENSIONS)
soup = BeautifulSoup(converted, "html.parser")


# ---------------------------------------------------------------- links
def source_page_name(target: Path) -> str:
    if target.suffix == ".md":
        return target.stem.lower() + ".html"
    return "source-" + re.sub(r"[^a-z0-9]+", "-", str(target.relative_to(ROOT)).lower()).strip("-") + ".html"


source_pages: dict[str, tuple[Path, bool]] = {}
SPECIAL_TARGETS = {"outputs/subliminal-learning-explained.html": "interactive.html"}


def rewrite_links(tree, base: Path = ROOT, collect: bool = True) -> None:
    """Point repository-relative links at their offline HTML equivalents."""
    for link in tree.select("a[href]"):
        href = link["href"]
        if href.startswith(("https:", "http:", "mailto:", "#")) or href.endswith(".html") and "/" not in href:
            continue
        path, _, fragment = href.partition("#")
        target = (base / unquote(path)).resolve()
        relative = str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else ""
        if relative in SPECIAL_TARGETS:
            link["href"] = SPECIAL_TARGETS[relative]
        elif target.is_file():
            name = source_page_name(target)
            if collect:
                source_pages[name] = (target, target.suffix == ".md")
            link["href"] = name + ("#" + fragment if fragment else "")
        else:
            # Folders (run directories, scripts/) are not part of the offline guide. Keep the visible
            # text and say where it lives instead of linking to an unrelated page.
            link.name = "span"
            link.attrs = {"class": "repo-path", "title": f"Folder in the repository: {relative or path}"}


rewrite_links(soup)

# ---------------------------------------------------------------- reading affordances
PAPER_OURS = re.compile(r'^\W?(Paper|Ours)\b')
for strong in soup.find_all("strong"):
    match = PAPER_OURS.match(strong.get_text())
    if match and len(strong.get_text()) < 45:
        strong["class"] = ["tag", "tag-" + match[1].lower()]
for th in soup.find_all("th"):
    match = PAPER_OURS.match(th.get_text())
    if match:
        th["class"] = ["col-" + match[1].lower()]
for quote in soup.find_all("blockquote"):
    lead = quote.get_text(" ", strip=True)
    if lead.startswith("Status"):
        quote["class"] = ["callout-status"]
    elif lead.startswith(("Intuition", "Hypothesis")):
        quote["class"] = ["callout-intuition"]
    elif lead.startswith("If "):
        quote["class"] = ["callout-question"]
for table in soup.find_all("table"):
    table.wrap(soup.new_tag("div", attrs={"class": "table-wrap"}))

# ---------------------------------------------------------------- split into topics
# Each topic keeps all its original prose, tables, code and figures; nothing is summarised.
groups = []
pending = []
chapter = "Introduction"
chapter_for_topic = chapter
for element in list(soup.contents):
    is_heading = getattr(element, "name", None) in {"h2", "h3", "h4"}
    if is_heading and pending:
        has_prose = any(getattr(item, "name", None) not in {None, "h1", "h2", "h3", "h4", "hr"} for item in pending)
        if has_prose:
            groups.append((chapter_for_topic, pending))
            pending = []
    if getattr(element, "name", None) == "h2":
        chapter = element.get_text(" ", strip=True)
    if is_heading or not pending:
        chapter_for_topic = chapter
    pending.append(element)
if pending:
    groups.append((chapter_for_topic, pending))

articles = []
for number, (chapter, elements) in enumerate(groups):
    headings = [element for element in elements if getattr(element, "name", None) in {"h1", "h2", "h3", "h4"}]
    title = headings[-1].get_text(" ", strip=True) if headings else chapter
    article = soup.new_tag("article", id=f"topic-{number}")
    article["data-chapter"] = chapter
    article["data-title"] = title
    if number:
        article["hidden"] = ""
    # Horizontal rules only separated parts in the Markdown; topics are already separated.
    for element in elements:
        if getattr(element, "name", None) == "hr":
            element.extract()
            continue
        article.append(element.extract())
    articles.append(article)

# Repair the source document's GitHub-style anchors to the converter's heading IDs.
heading_ids = {heading["id"] for article in articles for heading in article.select("[id]")}
normalize_anchor = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
for article in articles:
    for link in article.select('a[href^="#"]'):
        target = link["href"][1:]
        if target not in heading_ids:
            match = next((c for c in heading_ids if normalize_anchor(c) == normalize_anchor(target)), None)
            if match:
                link["href"] = "#" + match

CHAPTERS = {  # chapter heading -> (file, short name)
    "Introduction": ("introduction.html", "Introduction"),
    "Contents": ("contents.html", "Contents"),
    "Part 1 — The ideas you need": ("foundations.html", "Part 1 · The ideas you need"),
    "Part 2 — What the original paper did": ("paper.html", "Part 2 · What the original paper did"),
    "Part 3 — What our scripts do": ("scripts.html", "Part 3 · What our scripts do"),
    "Part 4 — Original versus ours, side by side": ("comparison.html", "Part 4 · Original versus ours"),
    "Part 5 — What to do next": ("next-steps.html", "Part 5 · What to do next"),
    "Sources": ("sources.html", "Sources"),
}
chapters = list(dict.fromkeys(article["data-chapter"] for article in articles))
assert set(chapters) == set(CHAPTERS), set(chapters) ^ set(CHAPTERS)

# ---------------------------------------------------------------- page chrome
styles = (TEMPLATES / "guide.css").read_text()
script = (TEMPLATES / "guide.js").read_text()
template = (TEMPLATES / "page.html").read_text()
first_topic = {chapter: next(a["id"] for a in articles if a["data-chapter"] == chapter) for chapter in chapters}


def page(title, content, *, mode, header="", footer="", current_chapter=None):
    values = {"TITLE": html.escape(title), "STYLES": styles, "MODE": mode,
              "TOPBAR": topbar(mode, current_chapter), "HEADER": header, "CONTENT": content,
              "FOOTER": footer, "SIDEBAR": course_map(mode, current_chapter), "SCRIPT": script}
    result = template
    for key, value in values.items():
        result = result.replace("@@" + key + "@@", value)
    assert "@@" not in result
    return result


def topbar(mode, current_chapter=None):
    """Site title, the reading-mode switch, and a jump to the chapter list on narrow screens."""
    chapter = current_chapter or chapters[0]
    stepper_href = f"index.html#{first_topic[chapter]}"
    chapter_href = CHAPTERS[chapter][0] + (f"#{first_topic[chapter]}" if mode == "stepper" else "")
    stepper_current = ' aria-current="page"' if mode == "stepper" else ""
    chapter_current = ' aria-current="page"' if mode == "chapter" else ""
    demos_current = ' aria-current="page"' if mode == "demo" else ""
    return (
        '<header class="topbar"><div class="topbar-inner">'
        '<a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true"></span>'
        '<span class="brand-text">Subliminal learning<span class="brand-tail">, explained slowly</span></span></a>'
        '<nav class="topbar-nav" aria-label="Reading mode">'
        '<div class="mode-switch" role="group" aria-label="Reading mode">'
        f'<a id="mode-stepper" href="{stepper_href}"{stepper_current} title="One topic at a time">'
        'Step<span class="wide-only"> by step</span></a>'
        f'<a id="mode-chapter" href="{chapter_href}"{chapter_current} title="Each chapter on one page">'
        '<span class="wide-only">Whole </span>chapter<span class="wide-only">s</span></a></div>'
        f'<a class="topbar-link" href="interactive.html"{demos_current}>Demos</a>'
        '<a class="topbar-link narrow-only" href="#map-title">Contents</a>'
        '</nav></div></header>'
    )


def masthead(eyebrow="", status=True):
    eyebrow = f'<p class="eyebrow">{eyebrow}</p>' if eyebrow else ""
    if not (eyebrow or status):
        return ""
    return f'<header class="masthead">{eyebrow}{RUN_STATUS if status else ""}</header>'


def course_map(mode, current_chapter=None):
    """Every chapter and topic. A sticky sidebar on wide screens; the closing contents list on narrow ones.

    Links keep the reader in the mode they are in: the step-by-step reader links into itself, chapter
    pages link to chapter pages, and supplements link into the step-by-step reader."""
    items = []
    for chapter in chapters:
        filename, short = CHAPTERS[chapter]
        peers = [a for a in articles if a["data-chapter"] == chapter]
        if mode == "chapter":
            chapter_href = filename
            topic_href = (lambda a: f"#{a['id']}") if chapter == current_chapter else (lambda a: f"{filename}#{a['id']}")
        else:
            chapter_href = f"index.html#{first_topic[chapter]}" if mode == "stepper" else filename
            topic_href = lambda a: f"index.html#{a['id']}"
        is_current = chapter == current_chapter
        current = ' aria-current="page"' if is_current and mode == "chapter" else ""
        topics = "".join(f'<li><a href="{topic_href(a)}" data-topic="{a["id"]}">{html.escape(a["data-title"])}</a></li>'
                         for a in peers) if len(peers) > 1 else ""
        count = f"{len(peers)}" if len(peers) > 1 else ""
        items.append(
            f'<li class="map-chapter{" is-current" if is_current else ""}" data-chapter="{html.escape(chapter, quote=True)}" data-file="{filename}">'
            f'<a class="chapter-name" href="{chapter_href}" data-topic="{first_topic[chapter]}"{current}>'
            f'<span>{html.escape(short)}</span>' + (f'<small>{count}</small>' if count else "") + "</a>"
            + (f"<ol>{topics}</ol>" if topics else "") + "</li>")
    extra_pages = [("interactive.html", "Interactive demonstrations", "demo"), ("readme.html", "README", None),
                   ("validation.html", "Validation record", None)]
    extras = "".join(
        f'<li><a href="{href}"' + (' aria-current="page"' if mode == key else "") + f'>{label}</a></li>'
        for href, label, key in extra_pages)
    return (f'<aside class="sidebar"><nav class="course-map" aria-labelledby="map-title">'
            f'<h2 id="map-title">Contents</h2><ol class="map-chapters">{"".join(items)}</ol>'
            f'<p class="map-group">Companion pages</p><ol class="map-extras">{extras}</ol></nav></aside>')


# ---------------------------------------------------------------- index (step-by-step reader)
options = "".join(f'<option value="{html.escape(c, quote=True)}">{html.escape(CHAPTERS[c][1])}</option>'
                  for c in chapters)
navigator = (
    '<nav class="navigator" aria-label="Topic navigator"><div class="navigator-row">'
    f'<label>Chapter<select id="guide-chapter">{options}</select></label>'
    '<label>Topic<select id="guide-topic"></select></label></div>'
    '<div class="navigator-status"><span class="progress-track"><span class="progress-fill" id="guide-progress"></span></span>'
    '<span id="guide-position" aria-live="polite"></span>'
    '<div class="stepper"><button type="button" id="guide-step-previous" aria-label="Previous topic" title="Previous topic (←)">←</button>'
    '<button type="button" id="guide-step-next" aria-label="Next topic" title="Next topic (→)">→</button></div></div></nav>'
)
page_turn = ('<div class="page-turn">'
             '<button type="button" id="guide-previous"><small>← Previous</small><span id="guide-previous-title"></span></button>'
             '<button type="button" id="guide-next"><small>Next →</small><span id="guide-next-title"></span></button></div>'
             '<p class="key-hint">Tip: the <kbd>←</kbd> and <kbd>→</kbd> keys move between topics.</p>')
(OUTPUT / "index.html").write_text(page(
    "Complete tutorial",
    navigator + '<div id="guide-reading">' + "".join(map(str, articles)) + "</div>" + page_turn,
    mode="stepper", header=masthead(), current_chapter=chapters[0],
))

# ---------------------------------------------------------------- one page per chapter
for position, chapter in enumerate(chapters):
    filename, short = CHAPTERS[chapter]
    content = "".join(str(article) for article in articles if article["data-chapter"] == chapter)
    chapter_soup = BeautifulSoup(content, "html.parser")
    for article in chapter_soup.select("article[hidden]"):
        del article["hidden"]
    for link in chapter_soup.select('a[href^="#"]'):
        target = link["href"][1:]
        if not chapter_soup.find(id=target):
            owner = next((a for a in articles if a.find(id=target)), None)
            owner_chapter = owner["data-chapter"] if owner else None
            link["href"] = (CHAPTERS[owner_chapter][0] if owner_chapter else "index.html") + "#" + target
    toc_entries = [h for h in chapter_soup.find_all(["h3", "h4"]) if h.get("id")]
    toc = ""
    if len(toc_entries) > 2:
        toc = ('<nav class="chapter-toc" aria-label="In this chapter"><p>In this chapter</p><ol>'
               + "".join(f'<li class="toc-{h.name}"><a href="#{h["id"]}">{h.decode_contents()}</a></li>'
                         for h in toc_entries)
               + "</ol></nav>")
    previous = chapters[position - 1] if position else None
    following = chapters[position + 1] if position + 1 < len(chapters) else None
    turn = ('<nav class="chapter-turn" aria-label="Chapters">'
            + (f'<a class="prev" href="{CHAPTERS[previous][0]}"><small>← Previous chapter</small>{html.escape(CHAPTERS[previous][1])}</a>' if previous else "")
            + (f'<a class="next" href="{CHAPTERS[following][0]}"><small>Next chapter →</small>{html.escape(CHAPTERS[following][1])}</a>' if following else "")
            + "</nav>")
    header = masthead(f"Chapter {position + 1} of {len(chapters)}")
    (OUTPUT / filename).write_text(page(short, toc + str(chapter_soup) + turn, mode="chapter", header=header,
                                        current_chapter=chapter))

# ---------------------------------------------------------------- supplementary pages
supplement_footer = ('<nav class="chapter-turn" aria-label="Back"><a class="prev" href="index.html"><small>← Back to</small>'
                     'The complete tutorial</a><a class="next" href="sources.html"><small>See also →</small>Sources</a></nav>')
written = set()
while set(source_pages) - written:
    for filename, (target, is_markdown) in sorted(source_pages.items()):
        if filename in written:
            continue
        written.add(filename)
        relative = target.relative_to(ROOT)
        if is_markdown:
            body = BeautifulSoup(markdown.markdown(target.read_text(), extensions=MD_EXTENSIONS), "html.parser")
            for table in body.find_all("table"):
                table.wrap(body.new_tag("div", attrs={"class": "table-wrap"}))
            rewrite_links(body, target.parent)
            content = f'<article class="supplement">{body}</article>'
        else:
            content = (f'<article class="supplement"><h1><code>{html.escape(str(relative))}</code></h1>'
                       f'<p class="source-note">A verbatim copy of the repository file, included so the guide works '
                       f'offline.</p><pre><code>{html.escape(target.read_text())}</code></pre></article>')
        (OUTPUT / filename).write_text(page(target.name, content, mode="supplement",
                                            header=masthead(f"Supplement · <code>{html.escape(str(relative))}</code>", status=False),
                                            footer=supplement_footer))

interactive = (ROOT / "outputs/subliminal-learning-explained.html").read_text()
(OUTPUT / "interactive.html").write_text(page(
    "Interactive demonstrations",
    '<article class="supplement demo-page">'
    + interactive + "</article>",
    mode="demo", header=masthead("Companion to the tutorial", status=False), footer=supplement_footer))

# Remove pages from earlier builds that nothing generates any more.
expected = {"index.html", "interactive.html"} | {f for f, _ in CHAPTERS.values()} | set(source_pages)
for old in OUTPUT.glob("*.html"):
    if old.name not in expected:
        old.unlink()

# ---------------------------------------------------------------- checks
original_text = BeautifulSoup(converted, "html.parser").get_text(" ", strip=True)
final_text = BeautifulSoup("".join(map(str, articles)), "html.parser").get_text(" ", strip=True)
assert re.sub(r"\s+", " ", original_text) == re.sub(r"\s+", " ", final_text), "tutorial text changed while splitting"
index_html = (OUTPUT / "index.html").read_text()
assert "```mermaid" not in index_html
ids = re.findall(r'\bid="([^"]+)"', index_html)
duplicates = {i for i in ids if ids.count(i) > 1}
assert not duplicates, duplicates
assert index_html.count('class="diagram-svg diagram-wide"') == index_html.count('class="diagram-svg diagram-narrow"') == 7
assert index_html.count('role="img"') == 14
print(f"Built {len(list(OUTPUT.glob('*.html')))} HTML pages, {len(articles)} complete topics, "
      f"and 7 diagrams (14 responsive SVG layouts).")
print("No tutorial prose, examples, tables or code blocks were omitted.")
