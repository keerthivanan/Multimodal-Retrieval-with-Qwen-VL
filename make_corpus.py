"""
make_corpus.py — Generate a small, *engineered* intranet-style corpus for Task 1.

WHY THIS EXISTS (design intent):
    Multimodal retrieval only beats text-only RAG when the corpus actually
    contains information that lives in PIXELS, not in extractable text. A demo
    over generic PDFs would show no delta and make the whole exercise pointless.

    So this generator deliberately plants three kinds of documents:
      1. TEXT-RICH docs (policies)         -> text-only RAG should tie/win here.
      2. MIXED docs (prose + a chart)      -> nuanced; text can find the doc,
                                              but the *value* lives in the chart.
      3. VISUAL-ONLY docs (standalone PNGs, no extractable text) -> text-only
                                              is BLIND to these; multimodal wins.

    Ground truth (which doc answers which query, and whether the answer is
    visual-only) is written to corpus_manifest.json and consumed by evaluate.py.

Deterministic: fixed seeds so results reproduce.
No network, no API keys. Pure matplotlib + reportlab + Pillow.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE / "data" / "corpus"
MANIFEST_PATH = HERE / "data" / "corpus_manifest.json"
_TMP = CORPUS_DIR / "_charts"  # transient chart PNGs embedded into PDFs

# Ground-truth chart data lives here so it is a single source of truth.
REVENUE = {"Q1": 120, "Q2": 150, "Q3": 210, "Q4": 180}  # Q3 peak (only in chart)
EXPENSES = {"Engineering": 45, "Sales": 25, "Marketing": 20, "Operations": 10}  # Eng largest
HEADCOUNT = {"2021": 40, "2022": 65, "2023": 110, "2024": 180, "2025": 260}


def _load_font(size: int) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Best-effort truetype font; falls back to PIL default if unavailable."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Chart builders (return path to a PNG on disk)
# ---------------------------------------------------------------------------
def _bar_chart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3), dpi=130)
    ax.bar(list(REVENUE.keys()), list(REVENUE.values()), color="#3b6fb0")
    ax.set_ylabel("Revenue ($M)")
    ax.set_title("Quarterly Revenue (FY2025)")
    # NOTE: intentionally NO value labels on bars -> the peak quarter (Q3)
    # is recoverable only by reading the chart, not from any text.
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _pie_chart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5), dpi=130)
    ax.pie(
        list(EXPENSES.values()),
        labels=list(EXPENSES.keys()),
        autopct="%1.0f%%",
        colors=["#3b6fb0", "#e08a3c", "#5aa469", "#b0553b"],
    )
    ax.set_title("Departmental Expense Share")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _line_chart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3), dpi=130)
    ax.plot(list(HEADCOUNT.keys()), list(HEADCOUNT.values()), marker="o", color="#5aa469")
    ax.set_ylabel("Employees")
    ax.set_title("Headcount Growth")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PDF builder: prose paragraphs + optional embedded chart image
# ---------------------------------------------------------------------------
def _make_pdf(path: Path, title: str, paragraphs: list[str], chart_png: Path | None) -> None:
    c = canvas.Canvas(str(path), pagesize=LETTER)
    width, height = LETTER
    y = height - inch

    c.setFont("Helvetica-Bold", 18)
    c.drawString(inch, y, title)
    y -= 0.5 * inch

    c.setFont("Helvetica", 11)
    for para in paragraphs:
        # naive word-wrap at ~90 chars
        line = ""
        for word in para.split():
            if len(line) + len(word) + 1 > 90:
                c.drawString(inch, y, line)
                y -= 0.22 * inch
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            c.drawString(inch, y, line)
            y -= 0.35 * inch

    if chart_png is not None and chart_png.exists():
        img = Image.open(chart_png)
        aspect = img.height / img.width
        draw_w = 4.5 * inch
        draw_h = draw_w * aspect
        if y - draw_h < inch:
            c.showPage()
            y = height - inch
        c.drawImage(str(chart_png), inch, y - draw_h, width=draw_w, height=draw_h)

    c.showPage()
    c.save()


# ---------------------------------------------------------------------------
# Standalone image builders (NO extractable text -> text-only RAG is blind)
# ---------------------------------------------------------------------------
def _make_salary_table_png(path: Path) -> None:
    """A 'scanned' table rendered as an image. Values exist only as pixels."""
    rows = [
        ("Role", "Level", "Base Salary"),
        ("AI/ML Engineer", "L3", "$135,000"),
        ("Data Scientist", "L3", "$128,000"),
        ("Backend Engineer", "L2", "$118,000"),
        ("Engineering Manager", "L5", "$172,000"),
        ("ML Research Lead", "L6", "$205,000"),
    ]
    W, H = 720, 340
    img = Image.new("RGB", (W, H), "#f6f4ee")  # off-white -> scanned look
    d = ImageDraw.Draw(img)
    d.text((20, 12), "CONFIDENTIAL — Compensation Table (Scanned)", font=_load_font(18), fill="#333")
    row_h, x0, y0 = 45, 20, 55
    col_x = [x0, x0 + 260, x0 + 420]
    for r, row in enumerate(rows):
        y = y0 + r * row_h
        if r == 0:
            d.rectangle([x0, y, W - 20, y + row_h], fill="#dfe6ef")
        for c_idx, cell in enumerate(row):
            f = _load_font(17 if r else 18)
            d.text((col_x[c_idx] + 8, y + 12), cell, font=f, fill="#111")
        d.line([x0, y + row_h, W - 20, y + row_h], fill="#bbb", width=1)
    img.save(path)


def _make_org_chart_png(path: Path) -> None:
    """Org hierarchy as boxes/lines. 'Who reports to whom' is visual-only."""
    W, H = 760, 380
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def box(cx, cy, text, w=170, h=52, fill="#eaf1fb"):
        d.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=fill, outline="#3b6fb0", width=2)
        d.text((cx - w // 2 + 10, cy - 9), text, font=_load_font(15), fill="#111")
        return (cx, cy)

    ceo = box(380, 45, "CEO — A. Rivera")
    cto = box(180, 160, "CTO — J. Chen")
    cfo = box(560, 160, "CFO — M. Osei")
    aihead = box(90, 285, "AI Lead — S. Park")
    beng = box(290, 285, "Backend Lead — R. Vale")
    fin = box(560, 285, "Finance — P. Adeyemi")
    for parent, child in [(ceo, cto), (ceo, cfo), (cto, aihead), (cto, beng), (cfo, fin)]:
        d.line([parent[0], parent[1] + 26, child[0], child[1] - 26], fill="#3b6fb0", width=2)
    img.save(path)


def _make_network_diagram_png(path: Path) -> None:
    """Data-flow diagram. 'What connects to the DB' is visual-only."""
    W, H = 760, 240
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def box(cx, cy, text, fill):
        w, h = 140, 56
        d.rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], fill=fill, outline="#333", width=2)
        d.text((cx - w // 2 + 12, cy - 9), text, font=_load_font(15), fill="#111")
        return (cx, cy, w)

    client = box(90, 120, "Client", "#eaf1fb")
    lb = box(270, 120, "Load Balancer", "#eaf1fb")
    app = box(460, 120, "App Server", "#fdeec9")
    db = box(650, 120, "Database", "#f6d7d0")
    cache = box(460, 40, "Cache", "#e5f4e7")
    for a, b in [(client, lb), (lb, app), (app, db)]:
        d.line([a[0] + a[2] // 2, a[1], b[0] - b[2] // 2, b[1]], fill="#333", width=3)
    d.line([cache[0], cache[1] + 28, app[0], app[1] - 28], fill="#333", width=2)
    img.save(path)


def _make_server_dashboard_png(path: Path) -> None:
    """Ops dashboard screenshot. The overloaded server is visual-only."""
    W, H = 760, 300
    img = Image.new("RGB", (W, H), "#1e2430")
    d = ImageDraw.Draw(img)
    d.text((20, 14), "Cluster Health — Live", font=_load_font(20), fill="#e8ecf3")
    servers = [("web-01", 34, "#5aa469"), ("web-02", 41, "#5aa469"),
               ("api-07", 92, "#d9534f"), ("db-03", 58, "#e0a24a")]
    x0, y0, bar_w = 40, 70, 150
    for i, (name, pct, color) in enumerate(servers):
        y = y0 + i * 52
        d.text((x0, y), name, font=_load_font(16), fill="#cfd6e0")
        d.rectangle([x0 + 110, y, x0 + 110 + bar_w, y + 24], outline="#445", width=1)
        d.rectangle([x0 + 110, y, x0 + 110 + int(bar_w * pct / 100), y + 24], fill=color)
        d.text((x0 + 110 + bar_w + 12, y + 2), f"CPU {pct}%", font=_load_font(15), fill="#cfd6e0")
    img.save(path)


def _make_error_screenshot_png(path: Path) -> None:
    """An error dialog screenshot. Some text is present but as pixels."""
    W, H = 620, 220
    img = Image.new("RGB", (W, H), "#2b2b2b")
    d = ImageDraw.Draw(img)
    d.rectangle([40, 40, W - 40, H - 40], fill="#f4f4f4", outline="#c0392b", width=3)
    d.text((60, 60), "⚠  Application Error", font=_load_font(20), fill="#c0392b")
    d.text((60, 105), "Database connection timeout after 30s.", font=_load_font(16), fill="#222")
    d.text((60, 135), "Host: db-03  Code: ETIMEDOUT (5401)", font=_load_font(15), fill="#444")
    img.save(path)


def _make_roadmap_png(path: Path) -> None:
    """Product roadmap timeline as an image."""
    W, H = 780, 240
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 12), "Product Roadmap 2025", font=_load_font(20), fill="#111")
    lanes = [("Q1  Search v2", "#3b6fb0"), ("Q2  Mobile App", "#5aa469"),
             ("Q3  AI Assistant", "#e0a24a"), ("Q4  Analytics Suite", "#8e5aa4")]
    for i, (label, color) in enumerate(lanes):
        y = 60 + i * 42
        d.rectangle([30, y, 30 + 120 + i * 150, y + 26], fill=color)
        d.text((40, y + 4), label, font=_load_font(15), fill="white")
    img.save(path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build() -> list[dict]:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    _TMP.mkdir(parents=True, exist_ok=True)

    # Pre-render charts embedded into PDFs.
    _bar_chart(_TMP / "revenue.png")
    _pie_chart(_TMP / "expenses.png")
    _line_chart(_TMP / "headcount.png")

    manifest: list[dict] = []

    # ---- TEXT-RICH PDFs (text-only RAG should do well here) ----------------
    _make_pdf(
        CORPUS_DIR / "security_policy.pdf",
        "Information Security Policy",
        [
            "All employees must use a password of at least 14 characters combining "
            "letters, numbers, and symbols. Passwords must be rotated every 90 days.",
            "Multi-factor authentication (MFA) is mandatory for all remote access and "
            "for any system handling customer data. Report lost devices within 24 hours.",
        ],
        None,
    )
    manifest.append(dict(doc_id="security_policy", file="security_policy.pdf", type="pdf",
                         title="Information Security Policy", modality="text",
                         visual_only=False,
                         description="Password rules, MFA requirements, device reporting."))

    _make_pdf(
        CORPUS_DIR / "onboarding_guide.pdf",
        "New Employee Onboarding Guide",
        [
            "Welcome! On day one, collect your laptop from IT, set up your email, and "
            "enroll in MFA. Your manager will schedule a first-week orientation.",
            "Complete the mandatory compliance training within your first two weeks and "
            "book a 1:1 with your team lead.",
        ],
        None,
    )
    manifest.append(dict(doc_id="onboarding_guide", file="onboarding_guide.pdf", type="pdf",
                         title="New Employee Onboarding Guide", modality="text",
                         visual_only=False,
                         description="First-day steps, IT setup, compliance training."))

    _make_pdf(
        CORPUS_DIR / "remote_work_policy.pdf",
        "Remote Work Policy",
        [
            "Employees may work remotely up to three days per week with manager approval. "
            "Core collaboration hours are 10:00-15:00 local time.",
            "Home office stipends of $500 per year are available. VPN must be used for all "
            "internal systems when working off-site.",
        ],
        None,
    )
    manifest.append(dict(doc_id="remote_work_policy", file="remote_work_policy.pdf", type="pdf",
                         title="Remote Work Policy", modality="text", visual_only=False,
                         description="Remote days, core hours, stipend, VPN requirement."))

    # ---- MIXED PDFs (prose + chart; the VALUE is in the chart) -------------
    _make_pdf(
        CORPUS_DIR / "revenue_report.pdf",
        "FY2025 Revenue Report",
        [
            "This report summarizes quarterly revenue performance for fiscal year 2025. "
            "Revenue reflects recognized bookings across all product lines.",
            "The chart below breaks down revenue by quarter. Growth was driven by the "
            "enterprise segment and improved renewal rates.",
        ],
        _TMP / "revenue.png",
    )
    manifest.append(dict(doc_id="revenue_report", file="revenue_report.pdf", type="pdf",
                         title="FY2025 Revenue Report", modality="mixed", visual_only=False,
                         description="Quarterly revenue bar chart; Q3 is the peak quarter at $210M.",
                         chart_fact="Q3 has the highest revenue (only visible in the bar chart)."))

    _make_pdf(
        CORPUS_DIR / "expense_breakdown.pdf",
        "Departmental Expense Breakdown",
        [
            "Operating expenses are distributed across the major departments as shown. "
            "This informs the next budgeting cycle.",
            "See the pie chart for each department's share of total operating expense.",
        ],
        _TMP / "expenses.png",
    )
    manifest.append(dict(doc_id="expense_breakdown", file="expense_breakdown.pdf", type="pdf",
                         title="Departmental Expense Breakdown", modality="mixed", visual_only=False,
                         description="Expense share pie chart; Engineering is the largest at 45%.",
                         chart_fact="Engineering is the largest expense category (only in the pie chart)."))

    _make_pdf(
        CORPUS_DIR / "headcount_trend.pdf",
        "Headcount Growth 2021-2025",
        [
            "The company has scaled its workforce steadily over the last five years. "
            "The trend below shows year-end headcount.",
        ],
        _TMP / "headcount.png",
    )
    manifest.append(dict(doc_id="headcount_trend", file="headcount_trend.pdf", type="pdf",
                         title="Headcount Growth 2021-2025", modality="mixed", visual_only=False,
                         description="Line chart of headcount growth from 40 (2021) to 260 (2025)."))

    # ---- VISUAL-ONLY standalone images (text-only RAG is BLIND) ------------
    _make_salary_table_png(CORPUS_DIR / "salary_table.png")
    manifest.append(dict(doc_id="salary_table", file="salary_table.png", type="image",
                         modality="image", title="Compensation Table (scanned)", visual_only=True,
                         description="Scanned salary table: ML Research Lead $205k is highest; AI/ML Engineer L3 $135k.",
                         chart_fact="Salary figures per role exist only as pixels in a scanned table."))

    _make_org_chart_png(CORPUS_DIR / "org_chart.png")
    manifest.append(dict(doc_id="org_chart", file="org_chart.png", type="image",
                         modality="image", title="Company Org Chart", visual_only=True,
                         description="Org hierarchy: CEO over CTO and CFO; AI Lead and Backend Lead report to the CTO.",
                         chart_fact="Reporting lines exist only in the diagram."))

    _make_network_diagram_png(CORPUS_DIR / "network_architecture.png")
    manifest.append(dict(doc_id="network_architecture", file="network_architecture.png", type="image",
                         modality="image", title="Internal Network Architecture", visual_only=True,
                         description="Data-flow diagram: Client -> Load Balancer -> App Server -> Database; Cache connects to App Server.",
                         chart_fact="The App Server is the component directly connected to the Database (only in the diagram)."))

    _make_server_dashboard_png(CORPUS_DIR / "server_dashboard.png")
    manifest.append(dict(doc_id="server_dashboard", file="server_dashboard.png", type="image",
                         modality="image", title="Cluster Health Dashboard", visual_only=True,
                         description="Ops dashboard; api-07 is overloaded at CPU 92% (red), others healthy.",
                         chart_fact="api-07 at 92% CPU is the overloaded server (only in the dashboard)."))

    _make_error_screenshot_png(CORPUS_DIR / "error_screenshot.png")
    manifest.append(dict(doc_id="error_screenshot", file="error_screenshot.png", type="image",
                         modality="image", title="Application Error Dialog", visual_only=True,
                         description="Error screenshot: database connection timeout on host db-03, ETIMEDOUT 5401.",
                         chart_fact="Error text is embedded in the screenshot image, not selectable text."))

    _make_roadmap_png(CORPUS_DIR / "product_roadmap.png")
    manifest.append(dict(doc_id="product_roadmap", file="product_roadmap.png", type="image",
                         modality="image", title="Product Roadmap 2025", visual_only=True,
                         description="Roadmap timeline: Q1 Search v2, Q2 Mobile App, Q3 AI Assistant, Q4 Analytics Suite."))

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Remove the transient chart PNGs; they are already embedded in the PDFs and
    # must not pollute the corpus directory (keeps the repo clean).
    shutil.rmtree(_TMP, ignore_errors=True)
    return manifest


# ---------------------------------------------------------------------------
# Real open-source documents (corpus realism)
#
# Downloaded from open, no-auth sources: arXiv (research PDFs with figures &
# tables) and Wikimedia Commons (freely licensed chart/diagram images).
# Famous items were chosen deliberately so their content is KNOWN and labelled
# queries can be written for them. Every download is best-effort: a dead URL
# is skipped with a warning, never a crash.
# ---------------------------------------------------------------------------
REAL_PDFS = [
    ("arxiv_attention", "Attention Is All You Need (Transformer paper)",
     "https://arxiv.org/pdf/1706.03762"),
    ("arxiv_resnet", "Deep Residual Learning for Image Recognition (ResNet)",
     "https://arxiv.org/pdf/1512.03385"),
    ("arxiv_bert", "BERT: Pre-training of Deep Bidirectional Transformers",
     "https://arxiv.org/pdf/1810.04805"),
    ("arxiv_vit", "An Image is Worth 16x16 Words (Vision Transformer)",
     "https://arxiv.org/pdf/2010.11929"),
    ("arxiv_gpt3", "Language Models are Few-Shot Learners (GPT-3)",
     "https://arxiv.org/pdf/2005.14165"),
    ("arxiv_word2vec", "Efficient Estimation of Word Representations (word2vec)",
     "https://arxiv.org/pdf/1301.3781"),
]
# Resolved via Special:FilePath so no hashed URLs are needed.
REAL_IMAGES = [
    ("wiki_internet_map", "Partial map of the Internet (network topology)",
     "Internet_map_1024.jpg"),
    ("wiki_sunspots", "400 years of sunspot number observations chart",
     "Sunspot_Numbers.png"),
    ("wiki_dna", "Labelled diagram of DNA structure",
     "DNA_Structure%2BKey%2BLabelled.pn_NoBB.png"),
    ("wiki_moores_law", "Moore's Law transistor count chart 1970-2020",
     "Moore%27s_Law_Transistor_Count_1970-2020.png"),
    ("wiki_climate_spiral", "Global temperature anomaly visualization",
     "20200324_Global_average_temperature_-_NASA-GISS_HadCrut_NOAA_Japan_BerkeleyE.svg.png"),
    ("wiki_tcpip", "TCP/IP network stack diagram",
     "IP_stack_connections.svg.png"),
]


def fetch_real_docs() -> list[dict]:
    """Best-effort download of real open documents; returns added manifest rows."""
    import requests

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    have = {m["doc_id"] for m in manifest}
    added: list[dict] = []
    headers = {"User-Agent": "mmrag-assessment/1.0 (corpus realism download)"}

    for doc_id, title, url in REAL_PDFS:
        if doc_id in have:
            continue
        try:
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("not a PDF response")
            fname = f"{doc_id}.pdf"
            (CORPUS_DIR / fname).write_bytes(r.content)
            added.append(dict(doc_id=doc_id, file=fname, type="pdf",
                              modality="mixed", title=title, visual_only=False,
                              description=title, source=url))
            print(f"[real] PDF   ok: {doc_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"[real] PDF SKIP {doc_id}: {exc}")

    for doc_id, title, commons_name in REAL_IMAGES:
        if doc_id in have:
            continue
        try:
            url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{commons_name}"
            r = requests.get(url, headers=headers, timeout=60)
            r.raise_for_status()
            from io import BytesIO

            from PIL import Image as PILImage

            img = PILImage.open(BytesIO(r.content))
            img.load()  # validate it really is an image
            fname = f"{doc_id}.png"
            img.convert("RGB").save(CORPUS_DIR / fname)
            added.append(dict(doc_id=doc_id, file=fname, type="image",
                              modality="image", title=title, visual_only=True,
                              description=title, source=url))
            print(f"[real] IMG   ok: {doc_id}")
        except Exception as exc:  # noqa: BLE001
            print(f"[real] IMG SKIP {doc_id}: {exc}")

    if added:
        manifest.extend(added)
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return added


if __name__ == "__main__":
    import sys

    items = build()
    real = [] if "--synthetic-only" in sys.argv else fetch_real_docs()
    total = len(items) + len(real)
    n_visual = sum(1 for m in items if m["visual_only"]) + \
        sum(1 for m in real if m["visual_only"])
    print(f"\nCorpus: {total} documents ({len(items)} engineered + {len(real)} real)")
    print(f"  visual-only: {n_visual}  (text-only RAG is blind to these)")
    print(f"Manifest -> {MANIFEST_PATH}")
