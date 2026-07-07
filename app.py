"""
app.py — Streamlit UI for the multimodal retrieval prototype.

A polished, self-contained UI (custom CSS) over the LangChain pipeline in
rag.py. Requirements from the brief:
  (a) select / upload documents   (b) text OR image query
  (c) top-K results with a relevance indicator + the page/region that matched.

Run:  streamlit run app.py
"""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import streamlit as st

import rag
from rag import build_pipelines, get_vlm, passages_for_file

st.set_page_config(page_title="Multimodal Document Search", page_icon="🔎",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
:root {
  --accent:#6366f1; --accent2:#8b5cf6; --ok:#10b981; --warn:#f59e0b;
  --card:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --bg:#f1f5f9;
}
@media (prefers-color-scheme: dark){
  :root{ --card:#1e293b; --ink:#e2e8f0; --muted:#94a3b8; --line:#334155; --bg:#0f172a; }
}
.stApp { background: var(--bg); }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding-top: 1.2rem; max-width: 1250px; }

.hero {
  background: linear-gradient(120deg, var(--accent), var(--accent2));
  border-radius: 18px; padding: 26px 30px; color:#fff; margin-bottom: 18px;
  box-shadow: 0 12px 30px rgba(99,102,241,.28);
}
.hero h1 { font-size: 1.9rem; margin:0; font-weight:800; letter-spacing:-.5px; }
.hero p  { margin:.4rem 0 0; opacity:.92; font-size:.98rem; }

.pill { display:inline-flex; align-items:center; gap:6px; padding:4px 11px;
  border-radius:999px; font-size:.78rem; font-weight:600; }
.pill-ok   { background:rgba(16,185,129,.15); color:var(--ok); }
.pill-warn { background:rgba(245,158,11,.15); color:var(--warn); }

.card { background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:14px; margin-bottom:12px; box-shadow:0 2px 10px rgba(2,6,23,.05);
  transition:transform .12s ease, box-shadow .12s ease; }
.card:hover { transform:translateY(-2px); box-shadow:0 10px 24px rgba(2,6,23,.10); }
.card img { width:100%; border-radius:9px; border:1px solid var(--line); }
.rank { display:inline-flex; width:22px; height:22px; border-radius:6px;
  background:var(--accent); color:#fff; font-size:.78rem; font-weight:700;
  align-items:center; justify-content:center; margin-right:8px; }
.doc-title { font-weight:700; color:var(--ink); font-size:1.02rem; }
.badge { font-size:.7rem; font-weight:700; padding:2px 9px; border-radius:999px;
  margin-left:8px; }
.badge-cap  { background:rgba(139,92,246,.16); color:var(--accent2); }
.badge-txt  { background:rgba(99,102,241,.14); color:var(--accent); }
.badge-best { background:rgba(16,185,129,.18); color:var(--ok); }
.card-best  { border-color:var(--ok); box-shadow:0 0 0 1px var(--ok),
              0 8px 20px rgba(16,185,129,.15); }
.meter { height:9px; background:var(--line); border-radius:999px; overflow:hidden;
  margin:9px 0 4px; }
.meter > span { display:block; height:100%;
  background:linear-gradient(90deg,var(--accent),var(--accent2)); border-radius:999px; }
.score { font-size:.78rem; color:var(--muted); font-weight:600; }
.snippet { margin-top:8px; font-size:.86rem; color:var(--muted); line-height:1.45;
  background:var(--bg); border-radius:8px; padding:9px 11px; }
.col-head { font-weight:800; font-size:1.05rem; color:var(--ink); margin:.2rem 0 .6rem;
  display:flex; align-items:center; gap:8px; }
.chip { display:inline-block; background:var(--card); border:1px solid var(--line);
  border-radius:8px; padding:5px 9px; margin:3px 3px 0 0; font-size:.8rem; color:var(--ink); }
.empty { background:var(--card); border:1px dashed var(--line); border-radius:12px;
  padding:18px; text-align:center; color:var(--muted); }

/* the prominent Answer box at the top of results */
.answer { background:linear-gradient(135deg, rgba(16,185,129,.14), rgba(99,102,241,.10));
  border:1px solid var(--ok); border-radius:14px; padding:16px 18px; margin:6px 0 14px;
  box-shadow:0 6px 18px rgba(16,185,129,.12); }
.answer-h { font-size:.8rem; font-weight:800; letter-spacing:.04em; color:var(--ok);
  text-transform:uppercase; margin-bottom:4px; }
.answer-b { font-size:1.12rem; line-height:1.5; color:var(--ink); font-weight:600; }
.answer-src { margin-top:8px; font-size:.8rem; color:var(--muted); }
.sources-label { font-size:.85rem; font-weight:700; color:var(--muted); margin:.4rem 0; }

/* controls */
.stButton > button {
  border-radius:11px; font-weight:700; padding:.55rem 1rem; border:1px solid var(--line);
  transition:transform .08s ease, box-shadow .12s ease; }
.stButton > button:hover { transform:translateY(-1px); }
.stButton > button[kind="primary"] {
  background:linear-gradient(90deg,var(--accent),var(--accent2)); color:#fff; border:none;
  box-shadow:0 8px 22px rgba(99,102,241,.40); font-size:1.02rem; }
.stTextInput input, .stTextInput > div > div {
  border-radius:11px !important; }
.stTextInput input { padding:.7rem .9rem; font-size:1rem; }
[data-testid="stFileUploaderDropzone"] { border-radius:12px; }
section[data-testid="stSidebar"] { border-right:1px solid var(--line); }
.stSlider, .stCheckbox { padding-top:.2rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building the search index…")
def _load():
    return build_pipelines()


def _resolve(image_path: str) -> Path | None:
    """Find an image whether the stored path is absolute (local) or must be
    looked up by filename in the committed folders (portable for cloud/Render)."""
    p = Path(image_path)
    if p.exists():
        return p
    for base in (rag.PAGES_DIR, rag.CORPUS_DIR):
        cand = base / p.name
        if cand.exists():
            return cand
    return None


def _img_uri(path: str) -> str | None:
    p = _resolve(path)
    if p is None:
        return None
    mime = "png" if p.suffix.lower() == ".png" else "jpeg"
    return f"data:image/{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def _manifest() -> list[dict]:
    if rag.MANIFEST_PATH.exists():
        return json.loads(rag.MANIFEST_PATH.read_text(encoding="utf-8"))
    return []


def _card_html(hit, rank: int) -> str:
    pct = max(0.0, min(1.0, (hit.score + 1) / 2)) * 100
    uri = _img_uri(hit.image_path)
    img = f'<img src="{uri}"/>' if uri else ""
    badge = ('<span class="badge badge-cap">🖼 image</span>' if hit.modality == "caption"
             else '<span class="badge badge-txt">📄 text</span>')
    best = ('<span class="badge badge-best">★ BEST MATCH</span>' if rank == 1 else "")
    snip = hit.text if len(hit.text) < 260 else hit.text[:260] + "…"
    return f"""
    <div class="card{' card-best' if rank == 1 else ''}">
      {img}
      <div style="margin-top:10px;">
        <span class="rank">{rank}</span><span class="doc-title">{hit.doc_title}</span>{badge}{best}
      </div>
      <div class="meter"><span style="width:{pct:.0f}%"></span></div>
      <div class="score">relevance {hit.score:.2f} · {hit.doc_file} · page {hit.page}</div>
      <div class="snippet">{snip}</div>
    </div>"""


multimodal, baseline = _load()
cfg = rag.config_summary()

_EMB_LABEL = {"OpenAIEmbeddings": "OpenAI embeddings",
              "OllamaEmbeddings": f"local semantic embeddings ({rag.OLLAMA_EMBED_MODEL})",
              "MockEmbeddings": "mock (keyword only)"}
emb_cls = type(multimodal.embeddings).__name__
emb_label = _EMB_LABEL.get(emb_cls, emb_cls)
vlm_label = ("Qwen2.5-VL" if multimodal.vlm.name == "ollama" else "mock captions")
real = emb_cls != "MockEmbeddings"

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
status = (f'<span class="pill pill-ok">● Live · {emb_label} + {vlm_label}</span>'
          if real else
          '<span class="pill pill-warn">● Demo mode · start Ollama or add an OpenAI key</span>')
st.markdown(f"""
<div class="hero">
  <h1>🔎 Multimodal Document Search</h1>
  <p>Search PDFs <b>and</b> images by meaning. Qwen-VL reads charts, diagrams &amp;
  screenshots into text; results are compared against a text-only RAG baseline.</p>
  <div style="margin-top:12px">{status}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine")
    store_label = "FAISS (exact cosine)" if rag.VECTORSTORE == "faiss" else "in-memory"
    st.markdown(f"- **Embeddings:** {emb_label}\n"
                f"- **Vision:** {vlm_label} · `{cfg['ollama_model']}`\n"
                f"- **Vector store:** {store_label}")
    if not real:
        st.info("**Demo mode.** Start Ollama (for local semantic search) or add "
                "`OPENAI_API_KEY` to `.env` for full accuracy.", icon="💡")

    st.markdown("### 📁 Corpus")
    chips = ""
    for m in _manifest():
        icon = "🖼" if m.get("visual_only") else ("📊" if m.get("modality") == "mixed" else "📄")
        chips += f'<span class="chip">{icon} {m["doc_id"]}</span>'
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("### ⬆️ Add your own")
    ups = st.file_uploader("PDFs or images", type=["pdf", "png", "jpg", "jpeg"],
                           accept_multiple_files=True, label_visibility="collapsed")
    if ups and st.button("Ingest into index", use_container_width=True):
        vlm = get_vlm()
        up_dir = rag.DATA_DIR / "uploads"
        up_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        with st.spinner("Captioning & indexing…"):
            for up in ups:
                dest = up_dir / up.name
                dest.write_bytes(up.getbuffer())
                new_p = passages_for_file(
                    src=dest, doc_id=f"upload_{dest.stem}", title=up.name,
                    file_name=up.name,
                    doc_type="pdf" if dest.suffix.lower() == ".pdf" else "image",
                    hint="", vlm=vlm, verbose=False)
                added += multimodal.add_passages(new_p)
                baseline.add_passages(new_p)
        st.success(f"Indexed {len(ups)} file(s) · +{added} passages")

    if st.button("🔄 Rebuild index", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Search controls
# ---------------------------------------------------------------------------
mode = st.segmented_control("Query type", ["🔤 Text", "🖼 Image"],
                            default="🔤 Text") or "🔤 Text"
c1, c2, c3 = st.columns([3, 1, 1])
with c2:
    k = st.slider("Results", 1, 8, 3,
                  help="How many ranked matches to show. #1 is the answer; "
                       "the rest are runners-up.")
with c3:
    model_choice = st.radio("Model", ["Qwen-VL", "OpenAI vision"],
                            help="The vision model used to read image queries and "
                                 "to rerank. Qwen-VL = local & free (mandated); "
                                 "OpenAI vision (gpt-4o-mini) = faster cloud.")
    rerank = st.toggle("Enable rerank", value=False,
                       help="The chosen model looks at each candidate image and "
                            "blends its 0-10 relevance judgement with the "
                            "embedding score. Higher precision, but slower.")


def _active_vlm():
    """The vision model the user selected — used for image-query reading AND
    reranking, so picking a model actually switches what runs."""
    if model_choice.startswith("OpenAI") and rag.OPENAI_API_KEY:
        return rag.OpenAICompatVLM("openai", rag.OPENAI_API_KEY, "gpt-4o-mini"), \
            "OpenAI vision"
    return get_vlm(), "Qwen-VL"

query, query_img = None, None
with c1:
    if mode == "🔤 Text":
        query = st.text_input("Search", placeholder="e.g. Which server is overloaded?",
                              label_visibility="collapsed")
    else:
        query_img = st.file_uploader("Upload an image query", type=["png", "jpg", "jpeg"],
                                     label_visibility="collapsed")

go = st.button("🔍  Search", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Results — side-by-side comparison
# ---------------------------------------------------------------------------
if go and (query or query_img):
    # The user-selected model drives BOTH image reading and reranking, so
    # picking a model actually changes what runs.
    active_vlm, model_label = _active_vlm()
    multimodal.vlm = active_vlm

    # For an image query: read it ONCE with the chosen model, show it, then both
    # pipelines do fast text search on that caption.
    search_str = query
    if query_img is not None:
        st.image(query_img, width=200, caption="your image query")
        with tempfile.NamedTemporaryFile(delete=False,
                suffix=Path(query_img.name).suffix or ".png") as tf:
            tf.write(query_img.getbuffer())
            tmp_path = tf.name
        _eta = "~5 sec" if model_label == "OpenAI vision" else "~15 sec on this laptop"
        with st.spinner(f"🧠 {model_label} is reading your image… ({_eta})"):
            search_str = active_vlm.caption(tmp_path)
        # A fallback caption ("Image: tmpXXXX") means the model errored — surface
        # it honestly instead of searching with junk text.
        if search_str.startswith("Image:") or len(search_str) < 20:
            st.error(f"😵 **{model_label} couldn't read the image just now** — "
                     "wait a few seconds and press **Search** again.")
            st.stop()
        st.success(f"🧠 **{model_label} read your image as:** {search_str}")

    # ---- ANSWER (RAG generation): read the top docs and answer, up top ----
    with st.spinner("💡 Reading the top matches to answer your question…"):
        _ahits = multimodal.search_text(search_str, k=3).hits
        _answer = rag.generate_answer(search_str, _ahits)
    if _answer and _ahits:
        _src = _ahits[0]
        st.markdown(
            f'<div class="answer"><div class="answer-h">💡 Answer</div>'
            f'<div class="answer-b">{_answer}</div>'
            f'<div class="answer-src">grounded in: <b>{_src.doc_title}</b> · '
            f'{_src.doc_file} · page {_src.page}</div></div>',
            unsafe_allow_html=True)

    # Measured separator on this corpus: junk queries ("hii man") top out at
    # ~0.21 cosine; the weakest CORRECT answer scores ~0.42. 0.30 splits them.
    WEAK = 0.30
    if rerank:
        st.caption(f"🔁 Reranking with **{model_label}**")

    st.markdown('<div class="sources-label">Retrieved documents '
                '(multimodal vs. text-only baseline)</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    for label, pipe, is_mm, col in [("🟣 Multimodal", multimodal, True, left),
                                    ("⚪ Text-only baseline", baseline, False, right)]:
        with col:
            with st.spinner(f"Searching · {label}…"):
                res = pipe.search_text(search_str, k=k, rerank=(rerank and is_mm))
            st.markdown(f'<div class="col-head">{label}'
                        f'<span class="score">· {res.latency_ms:.0f} ms</span></div>',
                        unsafe_allow_html=True)
            if res.hits and max(h.score for h in res.hits) < WEAK:
                st.warning("Weak matches — your query doesn't strongly match "
                           "any document. Results below are best-effort.",
                           icon="🤷")
            if not res.hits:
                st.markdown('<div class="empty">No match — this pipeline is '
                            'blind to this query (e.g. text-only over an '
                            'image-only answer).</div>', unsafe_allow_html=True)
            for i, hit in enumerate(res.hits, 1):
                st.markdown(_card_html(hit, i), unsafe_allow_html=True)
elif go:
    st.warning("Enter a text query or upload an image first.")
else:
    st.markdown('<div class="empty">Type a question above and hit '
                '<b>Search</b> to compare multimodal vs. text-only retrieval.</div>',
                unsafe_allow_html=True)
