# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a **research wiki**: an Obsidian vault holding an LLM-maintained, interlinked
knowledge base. It is **not a software project** — there is no build, no tests, no app. It
is a corpus of markdown that *you* (the LLM) build and maintain from sources the user
collects.

The core idea: instead of re-reading raw documents on every question, you **incrementally
build and maintain a persistent wiki** that sits between the user and the raw sources. When
a source is added, you read it, extract the key information, and integrate it into the
existing wiki — updating entity pages, revising summaries, flagging where new data
contradicts old claims. Knowledge is compiled once and kept current, not re-derived per
query. The wiki is a compounding artifact: it gets richer with every source added and every
question asked.

The division of labor is the whole point:

- **The user** curates sources, directs analysis, and asks questions.
- **You** do all the writing, summarizing, cross-referencing, filing, and bookkeeping.
- The user reads the wiki in Obsidian; you almost never ask the user to edit pages.

## Three layers

1. **Raw sources** (`raw/`) — immutable source documents. Read them, never edit them.
   (Rendered figures/tables for a source live in `raw/assets/`.) Two kinds arrive here:
   - **Web clips**: markdown saved by the Obsidian Web Clipper, with YAML frontmatter
     (`title`, `source`, `author` as `[[wikilinks]]`, `published`, `created`, `description`,
     `tags: [clippings]`). Already in `raw/`.
   - **PDFs**: the user drops a PDF *link* (or file). Convert it to markdown into
     `raw/` before ingesting (see PDF workflow below).
2. **The wiki** — LLM-generated markdown pages at the vault root (and in topic folders as
   the wiki grows). You own this layer entirely: entity pages, concept pages, topic
   summaries, comparisons, an overview/synthesis. Create, update, and cross-link freely.
3. **The schema** — this file. Co-evolve it with the user as conventions firm up. When the
   user establishes a new convention, record it here.

## Source conversion — pick the extractor in this order

Convert every source to markdown in `raw/` first. **The extractor matters** — choose by
what's available, best first:

1. **LaTeX source (arXiv & anything with `.tex`) — PREFERRED.** Convert from the source, not
   the PDF. You get *exact* tables (incl. spanning `\multicolumn`/rotated headers), verbatim
   math, the authors' **original full-res figure files**, and real citations — with **no OCR
   errors**. See the arXiv/LaTeX workflow below.
2. **PDF with no source → `pymupdf4llm`.** Fast, perfect text layer, no OCR risk — but drops
   all figures and mangles complex tables. Restore figures/tables per the pymupdf section below.

**Toolchain (all native arm64 in `mlenv`):** `pandoc`, `pymupdf4llm`. `latexpand` lives at
`~/.local/bin/latexpand` (runs on system perl). Activate with `conda activate mlenv`.

## arXiv / LaTeX-source ingestion (preferred for papers)

```bash
conda activate mlenv
ID=2507.13264                                        # arXiv id
curl -L -o /tmp/src.tar.gz "https://arxiv.org/e-print/$ID"
mkdir -p /tmp/src && tar -xzf /tmp/src.tar.gz -C /tmp/src
# 1. Flatten \input/\include (latexpand strips comments too). For clean sources a tiny
#    Python \input-inliner also works; latexpand is the reliable general tool.
cd /tmp/src && perl ~/.local/bin/latexpand main.tex > /tmp/flat.tex
# 2. Convert to GitHub-flavored markdown; resolve \cite from the .bib
pandoc /tmp/flat.tex -o /tmp/paper.md -t gfm --wrap=none --citeproc --bibliography=/tmp/src/ref.bib
```

Then finish it into a `raw/` source page:
- **Frontmatter** — add the standard block (`title`, `source`, `author` `[[wikilinks]]`,
  `published`, `created`, `tags: [clippings]`); add `converted_from: "arXiv LaTeX source
  (latexpand + pandoc)"`.
- **Figures** — the tarball ships the authors' originals (in `images/`, `results/`, …). Copy
  the *referenced* ones into `raw/assets/`, converting to PNG and **capping width ~2400px**
  (originals can be >5000px). Rewrite pandoc's `<figure><img><figcaption>…</figure>` blocks to
  an Obsidian embed + blockquote caption: `![[name.png]]` then `> **Caption title.** …`.
- **Tables** — pandoc emits clean markdown pipe tables for simple ones; **spanning/rotated
  headers come out as HTML `<table>`** — keep those as-is (they render in Obsidian and are
  correct, which the PDF extractors are not).
- **Citations** — pandoc `--citeproc` inlines `(Author Year)` and appends a reference list;
  keep both.
- Only the referenced figures matter — commented-out `\includegraphics` (latexpand removes the
  comments) are draft images, skip them.

*Worked example:* `raw/Voxtral.md` was rebuilt this way; its `voxtral-fig{1..9}-*.png` +
`voxtral-header.png` in `raw/assets/` are the LaTeX-source originals.

## PDF ingestion workflow (pymupdf) — fallback when no LaTeX source

When the user gives a PDF URL or local PDF **and no source is available**, convert it to
markdown in `raw/` first. Prefer `pymupdf4llm` (markdown-aware, preserves headings/tables);
fall back to plain `pymupdf` text extraction.

```bash
# Env: use the `mlenv` conda env — pymupdf/pymupdf4llm are already installed there.
conda activate mlenv
# Remote PDF: download first
curl -L -o /tmp/x.pdf "<url>"
# Preferred: markdown-structured extraction
python -c "import pymupdf4llm; open('raw/<Title>.md','w').write(pymupdf4llm.to_markdown('/tmp/x.pdf'))"
```

After conversion, **add frontmatter** matching the web-clip format (`title`, `source` = the
URL, `author` as `[[wikilinks]]`, `published`, `created` = today, `tags: [clippings]`) so PDF
and web sources look uniform. (`created`: use today's date from session context.)

### Figures & tables — the converter drops/mangles them, so restore them

`pymupdf4llm` **omits every image** (figures become `==> picture [W x H] intentionally
omitted <==` placeholders) and **mangles complex tables** (multi-value cells crammed with
`<br>`). Always check (`grep -nE 'intentionally omitted|Table [0-9]'`) and restore:

**Figures** — render the region straight from the PDF, since paper figures are usually vector
graphics (raw image extraction misses them). For each `Figure N:` caption, the figure is the
band **above** it; bound it by the **union of drawings + images + figure-internal text**
between the caption and the nearest *body paragraph* above (a wide, long text block — this is
what stops figure-internal labels from being mistaken for the boundary). Render at ~3× zoom:

```python
import pymupdf, re
doc = pymupdf.open('/tmp/x.pdf')
def cap(page, kind, n):
    for b in page.get_text("blocks"):
        if re.match(rf'\s*{kind}\s+{n}\s*:', b[4]): return pymupdf.Rect(b[:4])
def body(b): x0,y0,x1,y1,t=b[:5]; return (x1-x0)>350 and len(t.strip())>120
def fig(page, n, out):                       # figure caption is BELOW the figure
    c=cap(page,"Figure",n); bl=page.get_text("blocks")
    above=[b[3] for b in bl if b[3]<=c.y0-3 and b[1]>65 and body(b)]
    lo,hi=(max(above) if above else 70), c.y0-2
    items=[d['rect'] for d in page.get_drawings()]+[pymupdf.Rect(i['bbox']) for i in page.get_image_info()]
    items=[r for r in items if r.y0>=lo-3 and r.y1<=hi+3 and r.width>8 and r.height>4]
    items+=[pymupdf.Rect(b[:4]) for b in bl if b[1]>=lo-3 and b[3]<=hi+3 and b[4].strip()]
    r=items[0]
    for it in items[1:]: r|=it
    r=pymupdf.Rect(min(r.x0,80)-4, max(r.y0-4,lo-2), max(r.x1,500)+4, min(r.y1+4,hi+2))
    page.get_pixmap(matrix=pymupdf.Matrix(3,3), clip=r).save(out)
```

If a figure's internal labels confuse the top boundary (e.g. a figure that fills the page top),
fall back to a manual `clip` from the page-content top (~y=72) to the caption.

**Tables** — bound by their **horizontal rule lines** (table caption is *above* the table;
stop at the next `Table/Figure` caption): `bottom = max y1 of wide-thin drawings (width>200,
height<3) below the caption and before the next caption`.

**Where things go:**
- Save renders to `raw/assets/` (e.g. `moshi-fig1-overview.png`, `moshi-fig2-mimi.png`).
- In the **raw clip**, replace each `intentionally omitted` placeholder with the embed
  (`![[file.png]]`); the existing `Figure N:`/`Table N:` caption stays beneath it. Leave the
  **tiny equation placeholders** (heights ~15–70px, no caption) as-is — those are inline math,
  not figures.
- For **tables**: keep the converter's markdown **if it's clean**; if mangled, either render
  it as an image (complex/multi-phase tables) or reconstruct it as clean markdown (simple,
  high-value tables — do this version on the relevant **wiki page**).
- Embed each figure/table on the **wiki page** it belongs to (entity/concept), with the
  source caption as a `> blockquote`.
- **Markdown-table gotcha:** a `|` inside a `[[link|alias]]` collides with the column
  separator and breaks the row. Inside tables, use no-alias links (`[[Page Name]]`) or escape
  the pipe (`\|`).
- Embeds use `![[file.png]]` (local), which the image-localizer leaves alone (see media rule
  below) — never `![](url)` for these.

## Operations

**Ingest** (the default action when a source is added):
1. Read the source in `raw/` fully. For clips with inline images, read the text
   first, then view referenced images separately if they matter — the model can't read
   markdown-with-images in one pass.
2. Discuss key takeaways with the user.
3. Write or update a **source summary page** in the wiki for this document.
4. Update **`index.md`** (add the new page) and the relevant **entity/concept pages**
   across the wiki — a single source typically touches 10–15 pages.
5. Append an entry to **`log.md`**.
   Default to ingesting one source at a time and staying involved unless the user asks to
   batch.

**Query**: search the wiki (read `index.md` first to find relevant pages, then drill in),
synthesize an answer **with citations to wiki pages and underlying sources**. A genuinely
useful answer — a comparison, an analysis, a discovered connection — should be **filed back
into the wiki as a new page**, not left to vanish in chat. Log the query.

**Lint** (when asked to health-check): look for contradictions between pages, stale claims
superseded by newer sources, orphan pages with no inbound links, important concepts
mentioned but lacking their own page, missing cross-references, and data gaps worth a web
search. Suggest new questions and sources to pursue.

## Conventions

- **Wikilinks**: link entities and concepts with `[[Page Name]]` (Obsidian style), the same
  convention the clips already use for authors (e.g. `[[Thinking Machines Lab]]`). Linking
  is the value — cross-reference liberally. A `[[Page]]` that doesn't exist yet is a valid
  signal that the page is worth creating.
- **Frontmatter**: give wiki pages YAML frontmatter (`tags`, `created`, source counts, etc.)
  so the Obsidian Dataview plugin can query them.
- **No duplicate title heading**: do not restate the page title as a `# <Title>` H1 at the
  top of the body. Quartz renders the frontmatter `title` as the page heading, so an in-body
  H1 makes the title appear twice. Start the body with the first real section or intro line.
- **`index.md`** — content catalog. Every wiki page listed with a link, a one-line summary,
  and metadata, organized by category (entities, concepts, sources, topics). Read it first
  when answering; update it on every ingest.
- **`log.md`** — append-only chronological record. Start each entry with a consistent
  prefix so it stays grep-able:
  `## [YYYY-MM-DD] ingest | <Title>` / `... query | <question>` / `... lint`.
  `grep "^## \[" log.md | tail -5` shows recent activity.
- **Citations**: when a wiki claim comes from a source, point back to the `raw/` file
  (and its `source:` URL) so claims stay traceable.
- **Never edit `raw/`** — it is the source of truth. All synthesis lives in
  wiki pages. *One exception:* you may **augment** a raw source to **restore media the web
  clipper dropped** (videos, images, audio) — references/links only, and never reword the
  captured source text. Note such restorations in `log.md`.
- **Media embedding — use HTML, not `![](url)`.** This vault runs an Obsidian plugin that
  auto-localizes remote markdown image embeds (`![](url)`): it downloads the target into
  `./assets` (works for real image URLs, but turns YouTube/video links into broken HTML files
  saved as `.jpg`). The plugin does **not** touch raw HTML, so embed playable media as HTML:
  - **YouTube** → `<iframe width="100%" height="315" src="https://www.youtube.com/embed/<ID>" title="…" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>` (note `/embed/<ID>`, not `watch?v=`).
  - **Self-hosted video** → `<video width="100%" height="315" controls preload="metadata" src="<url>.mp4"></video>`.
  - **Remote still images** survive `![](url)` (they localize correctly); plain audio `.wav`
    can stay as `[label](url)` links. Attachment folder is `./assets` (per `.obsidian/app.json`).
- **Entities are subjects of study, not publishers.** An `Entities/` page is for things the
  wiki actually analyzes — models, products, techniques, benchmarks, datasets. **Companies /
  organizations / labs are *not* entity pages.** Capture them only as author/publisher
  `[[wikilinks]]` in source frontmatter and inline mentions (an unresolved `[[Company]]` link
  is fine and intended). Do not create a standalone page for a company.

## Environment notes

- This is an **Obsidian vault** (`.obsidian/` holds config; graph view shows the wiki's
  shape and surfaces orphans/hubs). Treat `.obsidian/`, `.DS_Store`, and `.git` as
  infrastructure — don't ingest or edit them.
- It is a plain folder of markdown — if the user initializes git, every change is versioned
  for free. Don't assume a remote.
