"""
rag.py — The whole multimodal RAG core, in one file, built on LangChain.

Sections:
    1. CONFIG      settings + paths (.env driven, safe mock defaults)
    2. EMBEDDINGS  LangChain Embeddings: OpenAI (real) | Mock (keyless)
    3. VLM         LangChain ChatOllama -> Qwen2.5-VL (captions + rerank)
    4. INGESTION   PDFs/images -> text passages + VLM caption passages
    5. RETRIEVAL   LangChain InMemoryVectorStore: multimodal + text-only baseline

Everything degrades gracefully: no OpenAI key or no Ollama => mock fallback,
never a crash. Run `python rag.py` to (re)build the search index.
"""

from __future__ import annotations

import base64
import json
import os
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore


# ===========================================================================
# 1. CONFIG
# ===========================================================================
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CORPUS_DIR = DATA_DIR / "corpus"
PAGES_DIR = DATA_DIR / "pages"
INDEX_DIR = DATA_DIR / "index"
REPORTS_DIR = HERE / "reports"
MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"
QUERIES_PATH = DATA_DIR / "queries.json"


def _load_dotenv() -> None:
    """Tiny .env loader (no extra dependency). Never overrides set env vars."""
    for cand in (HERE / ".env", HERE.parent / ".env"):
        if not cand.exists():
            continue
        for raw in cand.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v


_load_dotenv()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


EMBED_BACKEND = _env("EMBED_BACKEND", "openai").lower()   # openai | mock
VLM_BACKEND = _env("VLM_BACKEND", "ollama").lower()       # ollama | mock
OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
OPENAI_EMBED_MODEL = _env("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OLLAMA_HOST = _env("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "qwen2.5vl:3b")
OLLAMA_EMBED_MODEL = _env("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
# Hosted Qwen-VL API (OpenAI-compatible) so image queries stay on Qwen-VL even
# in the cloud where Ollama can't run. Default endpoint: OpenRouter.
QWEN_VL_API_KEY = _env("QWEN_VL_API_KEY", "")
QWEN_VL_BASE_URL = _env("QWEN_VL_BASE_URL", "https://openrouter.ai/api/v1")
QWEN_VL_MODEL = _env("QWEN_VL_MODEL", "qwen/qwen-2.5-vl-7b-instruct")
TOP_K = int(_env("TOP_K", "5"))
CHUNK_CHARS = int(_env("CHUNK_CHARS", "500"))
# Cap pages ingested per PDF: VLM captioning costs ~15 s/page. 16 covers the
# figure-rich body of typical papers (e.g. the attention-visualization figures
# on pp.13-15 of "Attention Is All You Need") while bounding a 75-page outlier.
MAX_PDF_PAGES = int(_env("MAX_PDF_PAGES", "16"))
# A committed FAISS index is loaded automatically whenever one exists (so the
# cloud never re-embeds 250 passages at a flaky free-tier startup). Set
# FORCE_REBUILD=1 to rebuild from fresh passages instead (e.g. after a
# re-ingest that added new pages).
FORCE_REBUILD = _env("FORCE_REBUILD", "0").lower() in ("1", "true", "yes")


def config_summary() -> dict:
    return {"embed_backend": EMBED_BACKEND, "vlm_backend": VLM_BACKEND,
            "openai_model": OPENAI_EMBED_MODEL,
            "openai_key_present": bool(OPENAI_API_KEY),
            "ollama_model": OLLAMA_MODEL, "top_k": TOP_K}


# ===========================================================================
# 2. EMBEDDINGS  (LangChain Embeddings interface)
# ===========================================================================
class MockEmbeddings(Embeddings):
    """Keyless, deterministic embeddings via char n-gram hashing. Not semantic,
    but lets the whole pipeline run with no key and still shows the
    multimodal>baseline effect (the baseline has no captions to match)."""

    def __init__(self, dim: int = 1024):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._vec = HashingVectorizer(n_features=dim, alternate_sign=False,
                                      analyzer="char_wb", ngram_range=(3, 5))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        mat = self._vec.transform([t if t.strip() else " " for t in texts]).toarray()
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype("float32").tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def get_embeddings() -> Embeddings:
    """Resolve the best available REAL embeddings, else mock.
    Priority: OpenAI (if key) -> local Ollama (if server up) -> mock.
    So results are semantic/real whenever *either* a key or Ollama is present."""
    if OPENAI_API_KEY:
        try:
            from langchain_openai import OpenAIEmbeddings

            # Generous timeout + retries: free-tier cloud cold starts have
            # flaky egress, and the default 2 retries / short timeout give up.
            return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL,
                                    api_key=OPENAI_API_KEY,
                                    timeout=60.0, max_retries=6)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"OpenAIEmbeddings init failed ({exc}); trying local.")
    if _ollama_up(OLLAMA_HOST):
        try:
            from langchain_ollama import OllamaEmbeddings

            emb = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_HOST)
            emb.embed_query("healthcheck")  # verify the model is actually pulled
            return emb
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Ollama embeddings unavailable ({exc}) -> mock.")
    warnings.warn("No OpenAI key and no local embed model -> mock embeddings.")
    return MockEmbeddings()


# ===========================================================================
# 3. VLM  (LangChain ChatOllama -> Qwen2.5-VL)
# ===========================================================================
CAPTION_PROMPT = (
    "You are indexing an internal company document image for search. "
    "Describe its content in 2-4 sentences: any title, chart type, axes, "
    "notable values, table contents, diagram connections, or status colors. "
    "Be concrete and include specific numbers/labels you can read.")


class VLM(ABC):
    name: str

    @abstractmethod
    def caption(self, image_path: str | Path, hint: str = "") -> str: ...

    def rerank(self, query: str, text: str, image_path: str | Path) -> float:
        return 0.5


class MockVLM(VLM):
    """Offline caption source: returns the manifest hint (deterministic)."""

    name = "mock"

    def caption(self, image_path: str | Path, hint: str = "") -> str:
        return hint or f"Image document: {Path(image_path).stem}"

    def rerank(self, query: str, text: str, image_path: str | Path) -> float:
        q = {w for w in query.lower().split() if len(w) > 2}
        t = {w for w in text.lower().split() if len(w) > 2}
        return len(q & t) / len(q) if q else 0.5


class ChatOllamaVLM(VLM):
    """Qwen2.5-VL via langchain-ollama ChatOllama (local, quantized, no key)."""

    name = "ollama"

    def __init__(self, model: str, host: str):
        from langchain_ollama import ChatOllama

        # num_ctx=8192: default 4096 is too small for high-res images (their
        # vision tokens alone can exceed it -> 400 exceed_context_size_error).
        self._llm = ChatOllama(model=model, base_url=host, temperature=0.1,
                               num_ctx=8192)

    @staticmethod
    def _message(prompt: str, image_path: str | Path):
        import io

        from langchain_core.messages import HumanMessage
        from PIL import Image

        # Downscale large images before sending: fewer vision tokens (fits the
        # context window), faster inference, no quality loss for captioning.
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}])

    def caption(self, image_path: str | Path, hint: str = "") -> str:
        try:
            out = self._llm.invoke([self._message(CAPTION_PROMPT, image_path)])
            return (out.content or "").strip() or (hint or Path(image_path).stem)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Ollama caption failed ({exc}) -> hint.")
            return hint or f"Image: {Path(image_path).stem}"

    def rerank(self, query: str, text: str, image_path: str | Path) -> float:
        prompt = (f"Query: {query!r}\nRate how relevant THIS document image is to "
                  "the query on a scale of 0 to 10 (10 = directly answers it). "
                  "Reply with ONLY the number.")
        try:
            out = self._llm.invoke([self._message(prompt, image_path)])
            num = "".join(c for c in (out.content or "") if c.isdigit() or c == ".")
            return max(0.0, min(1.0, float(num) / 10.0)) if num else 0.5
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Ollama rerank failed ({exc}) -> 0.5.")
            return 0.5


class OpenAICompatVLM(VLM):
    """Vision over ANY OpenAI-compatible chat API. Used for a HOSTED Qwen-VL
    endpoint (OpenRouter/DashScope serve real Qwen2.5-VL) so image queries stay
    on Qwen-VL in the cloud; also usable with OpenAI as a last-resort fallback."""

    def __init__(self, name: str, api_key: str, model: str,
                 base_url: str | None = None):
        from openai import OpenAI

        self.name = name
        self.model = model
        self._client = (OpenAI(api_key=api_key, base_url=base_url) if base_url
                        else OpenAI(api_key=api_key))

    @staticmethod
    def _data_uri(image_path: str | Path) -> str:
        import io

        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img.thumbnail((1024, 1024))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def _ask(self, prompt: str, image_path: str | Path) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0.1,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": self._data_uri(image_path)}}]}])
        return (resp.choices[0].message.content or "").strip()

    def caption(self, image_path: str | Path, hint: str = "") -> str:
        try:
            return self._ask(CAPTION_PROMPT, image_path) or (hint or Path(image_path).stem)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"{self.name} caption failed ({exc}) -> hint.")
            return hint or f"Image: {Path(image_path).stem}"

    def rerank(self, query: str, text: str, image_path: str | Path) -> float:
        prompt = (f"Query: {query!r}\nRate how relevant THIS document image is to "
                  "the query on a scale of 0 to 10. Reply with ONLY the number.")
        try:
            raw = self._ask(prompt, image_path)
            num = "".join(c for c in raw if c.isdigit() or c == ".")
            return max(0.0, min(1.0, float(num) / 10.0)) if num else 0.5
        except Exception:  # noqa: BLE001
            return 0.5


def _ollama_up(host: str) -> bool:
    try:
        import requests

        requests.get(f"{host.rstrip('/')}/api/tags", timeout=3).raise_for_status()
        return True
    except Exception:  # noqa: BLE001
        return False


def get_vlm() -> VLM:
    """Vision backend, Qwen-VL FIRST at every tier:
      1. Qwen2.5-VL local via Ollama   (primary; free; used whenever reachable)
      2. Qwen2.5-VL hosted API          (cloud; keeps image queries on Qwen-VL)
      3. OpenAI vision                  (last-resort fallback only)
      4. mock                           (offline)
    So image captioning is Qwen-VL locally AND in the cloud (given a hosted key).
    """
    # 1) local Qwen-VL
    if VLM_BACKEND == "ollama" and _ollama_up(OLLAMA_HOST):
        try:
            return ChatOllamaVLM(OLLAMA_MODEL, OLLAMA_HOST)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"ChatOllama init failed ({exc}); trying hosted Qwen-VL.")
    # 2) hosted Qwen-VL (real Qwen2.5-VL via OpenRouter/DashScope)
    if QWEN_VL_API_KEY:
        try:
            return OpenAICompatVLM("qwen-vl-api", QWEN_VL_API_KEY, QWEN_VL_MODEL,
                                   base_url=QWEN_VL_BASE_URL)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Hosted Qwen-VL init failed ({exc}); trying OpenAI.")
    # 3) OpenAI vision (only if no Qwen-VL is available anywhere)
    if OPENAI_API_KEY:
        try:
            return OpenAICompatVLM("openai", OPENAI_API_KEY, "gpt-4o-mini")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"OpenAI vision init failed ({exc}) -> mock.")
    warnings.warn("No Qwen-VL (local or hosted) and no OpenAI key -> mock VLM.")
    return MockVLM()


# ===========================================================================
# 4. INGESTION
# ===========================================================================
def _chunk(text: str, size: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks, cur = [], ""
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        if len(cur) + len(para) + 1 > size and cur:
            chunks.append(cur.strip())
            cur = para
        else:
            cur = f"{cur} {para}".strip()
    if cur:
        chunks.append(cur.strip())
    return chunks


def _render_pdf_pages(pdf_path: Path, doc_id: str) -> list[tuple[int, str, Path]]:
    """Render up to MAX_PDF_PAGES pages. Real-world PDFs can run to 75+ pages
    (GPT-3 paper); at ~20 s of VLM captioning per page an uncapped ingest would
    take an hour. First pages carry the title/abstract/key figures."""
    out = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc, start=1):
        if i > MAX_PDF_PAGES:
            break
        text = page.get_text("text")
        png = PAGES_DIR / f"{doc_id}_p{i}.png"
        page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(png)
        out.append((i, text, png))
    doc.close()
    return out


def passages_for_file(src: Path, doc_id: str, title: str, file_name: str,
                      doc_type: str, hint: str, vlm: VLM, verbose: bool = True,
                      caption_cache: dict | None = None) -> list[dict]:
    """Text + caption passages for ONE document. Reused by full ingest + uploads.
    caption_cache maps (doc_id, page) -> caption text so a re-ingest reuses
    already-computed captions (resumable; only NEW pages hit the VLM)."""
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    cache = caption_cache or {}
    out: list[dict] = []

    def _caption(image, page):
        cached = cache.get((doc_id, page))
        if cached:
            return cached, True
        return vlm.caption(image, hint=hint), False

    if doc_type == "pdf":
        for page_num, text, png in _render_pdf_pages(src, doc_id):
            for j, ch in enumerate(_chunk(text, CHUNK_CHARS)):
                out.append(dict(doc_id=doc_id, doc_title=title, doc_file=file_name,
                                modality="text", text=ch, page=page_num,
                                image_path=str(png)))
            cap, hit = _caption(png, page_num)
            out.append(dict(doc_id=doc_id, doc_title=title, doc_file=file_name,
                            modality="caption", text=cap, page=page_num,
                            image_path=str(png)))
            if verbose:
                print(f"[ingest] {doc_id} p{page_num}: "
                      f"{'cached' if hit else 'captioned'} ({len(cap)} chars)")
    else:  # standalone image -> NO text passage (baseline is blind to it)
        cap, hit = _caption(src, 1)
        out.append(dict(doc_id=doc_id, doc_title=title, doc_file=file_name,
                        modality="caption", text=cap, page=1, image_path=str(src)))
        if verbose:
            print(f"[ingest] {doc_id} (image): "
                  f"{'cached' if hit else 'captioned'} ({len(cap)} chars)")
    return out


def build_index(vlm: VLM | None = None, verbose: bool = True) -> list[dict]:
    """Ingest the whole corpus -> data/index/passages.json (the expensive step)."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    vlm = vlm or get_vlm()
    if verbose:
        print(f"[ingest] VLM backend = {vlm.name}")

    # Resume: reuse captions already computed in a previous run (only new pages
    # — e.g. deeper PDF pages after raising MAX_PDF_PAGES — hit the VLM).
    cache: dict = {}
    prev = INDEX_DIR / "passages.json"
    if prev.exists():
        for p in json.loads(prev.read_text(encoding="utf-8")):
            if p["modality"] == "caption":
                cache[(p["doc_id"], p["page"])] = p["text"]
        if verbose:
            print(f"[ingest] reusing {len(cache)} cached captions")

    passages: list[dict] = []
    for doc in manifest:
        passages += passages_for_file(
            src=CORPUS_DIR / doc["file"], doc_id=doc["doc_id"], title=doc["title"],
            file_name=doc["file"], doc_type=doc["type"],
            hint=doc.get("description", ""), vlm=vlm, verbose=verbose,
            caption_cache=cache)
    (INDEX_DIR / "passages.json").write_text(json.dumps(passages, indent=2),
                                             encoding="utf-8")
    if verbose:
        n_t = sum(1 for p in passages if p["modality"] == "text")
        print(f"[ingest] wrote {len(passages)} passages ({n_t} text, "
              f"{len(passages) - n_t} caption) -> {INDEX_DIR / 'passages.json'}")
    return passages


def load_passages() -> list[dict]:
    path = INDEX_DIR / "passages.json"
    if not path.exists():
        raise FileNotFoundError("passages.json missing — run `python rag.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


# ===========================================================================
# 5. RETRIEVAL  (LangChain InMemoryVectorStore)
# ===========================================================================
@dataclass
class SearchHit:
    doc_id: str
    doc_title: str
    doc_file: str
    modality: str
    page: int
    text: str
    image_path: str
    score: float
    reranked: bool = False


@dataclass
class SearchResult:
    hits: list[SearchHit]
    latency_ms: float
    query_caption: str | None = None
    doc_ranking: list[str] = field(default_factory=list)


def _passage_to_doc(p: dict) -> Document:
    return Document(page_content=p["text"], metadata={
        "doc_id": p["doc_id"], "doc_title": p["doc_title"], "doc_file": p["doc_file"],
        "modality": p["modality"], "page": p["page"], "image_path": p["image_path"]})


def _faiss_available() -> bool:
    # Using FAISS needs BOTH the faiss-cpu C library AND the LangChain wrapper
    # in langchain_community (which langchain 1.0 does NOT install as a
    # dependency). Require both here — otherwise VECTORSTORE would select the
    # faiss path and then crash on the wrapper import at first use (which is
    # exactly what broke the Render deploy). If either is missing we degrade
    # cleanly to InMemoryVectorStore.
    try:
        import faiss  # noqa: F401
        import langchain_community.vectorstores  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


VECTORSTORE = "faiss" if _faiss_available() else "memory"


class Pipeline:
    """include_captions=False -> text-only baseline; True -> multimodal.

    Vector store: FAISS (IndexFlatIP over L2-normalized vectors = exact cosine)
    when faiss-cpu is installed, else LangChain InMemoryVectorStore. Both are
    exact (no ANN approximation) — correct at this corpus size; at ~100k+
    chunks the FAISS index would switch to HNSW/IVF for sublinear search.
    """

    def __init__(self, name: str, include_captions: bool,
                 embeddings: Embeddings, vlm: VLM):
        self.name = name
        self.include_captions = include_captions
        self.embeddings = embeddings
        self.vlm = vlm
        self._store = None  # created lazily on first add (FAISS needs docs)

    def _wanted(self, passages: list[dict]) -> list[dict]:
        return [p for p in passages
                if p["modality"] == "text" or self.include_captions]

    def _add_docs(self, docs: list) -> None:
        if not docs:
            return
        if self._store is not None:
            self._store.add_documents(docs)
            return
        if VECTORSTORE == "faiss":
            import warnings as _w

            with _w.catch_warnings():
                _w.simplefilter("ignore")  # langchain-community sunset notice
                from langchain_community.vectorstores import FAISS
                from langchain_community.vectorstores.utils import DistanceStrategy

            # normalize_L2 + inner product == exact cosine similarity, keeping
            # score semantics identical to InMemoryVectorStore.
            self._store = FAISS.from_documents(
                docs, self.embeddings, normalize_L2=True,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
        else:
            self._store = InMemoryVectorStore(self.embeddings)
            self._store.add_documents(docs)

    def build(self, passages: list[dict]) -> "Pipeline":
        self._add_docs([_passage_to_doc(p) for p in self._wanted(passages)])
        return self

    def add_passages(self, passages: list[dict]) -> int:
        docs = [_passage_to_doc(p) for p in self._wanted(passages)]
        self._add_docs(docs)
        return len(docs)

    def save(self, folder: Path) -> None:
        """Persist a FAISS index to disk (so the cloud can load, not rebuild)."""
        if VECTORSTORE == "faiss" and self._store is not None:
            folder.mkdir(parents=True, exist_ok=True)
            self._store.save_local(str(folder))

    def load(self, folder: Path) -> "Pipeline":
        """Load a committed FAISS index — no embedding calls at startup."""
        from langchain_community.vectorstores import FAISS

        self._store = FAISS.load_local(str(folder), self.embeddings,
                                       allow_dangerous_deserialization=True)
        return self

    def _hits(self, query: str, k: int) -> list[SearchHit]:
        pairs = self._store.similarity_search_with_score(query, k=max(k * 3, k))
        hits = []
        for doc, score in pairs:
            m = doc.metadata
            hits.append(SearchHit(m["doc_id"], m["doc_title"], m["doc_file"],
                                  m["modality"], m["page"], doc.page_content,
                                  m["image_path"], float(score)))
        return hits

    def _rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        for h in hits:
            h.score = 0.5 * h.score + 0.5 * self.vlm.rerank(query, h.text, h.image_path)
            h.reranked = True
        return hits

    @staticmethod
    def _dedup(hits: list[SearchHit], k: int) -> tuple[list[SearchHit], list[str]]:
        best: dict[str, SearchHit] = {}
        for h in hits:
            if h.doc_id not in best or h.score > best[h.doc_id].score:
                best[h.doc_id] = h
        ranked = sorted(best.values(), key=lambda h: -h.score)[:k]
        return ranked, [h.doc_id for h in ranked]

    def search_text(self, query: str, k: int | None = None,
                    rerank: bool = False) -> SearchResult:
        k = k or TOP_K
        t0 = time.perf_counter()
        hits = self._hits(query, k)
        if rerank:
            hits = self._rerank(query, hits)
        ranked, ranking = self._dedup(hits, k)
        return SearchResult(ranked, (time.perf_counter() - t0) * 1000, doc_ranking=ranking)

    def search_image(self, image_path: str | Path, k: int | None = None,
                     rerank: bool = False) -> SearchResult:
        k = k or TOP_K
        t0 = time.perf_counter()
        caption = self.vlm.caption(image_path, hint="")
        hits = self._hits(caption, k)
        if rerank:
            hits = self._rerank(caption, hits)
        ranked, ranking = self._dedup(hits, k)
        return SearchResult(ranked, (time.perf_counter() - t0) * 1000,
                            query_caption=caption, doc_ranking=ranking)


def _prebuilt_dirs(embeddings) -> tuple[Path, Path]:
    tag = type(embeddings).__name__  # index is embedder-specific
    return (INDEX_DIR / f"faiss_{tag}_mm", INDEX_DIR / f"faiss_{tag}_bl")


def build_pipelines(embeddings: Embeddings | None = None, vlm: VLM | None = None
                    ) -> tuple[Pipeline, Pipeline]:
    """Return (multimodal, baseline). If a committed FAISS index exists it is
    loaded (startup makes ZERO embedding calls — this is what lets the cloud
    boot); otherwise it (re)builds from passages. FORCE_REBUILD=1 forces a
    rebuild (e.g. after a re-ingest that added new pages)."""
    embeddings = embeddings or get_embeddings()
    vlm = vlm or get_vlm()
    mm = Pipeline("multimodal", True, embeddings, vlm)
    bl = Pipeline("text-only baseline", False, embeddings, vlm)

    # Load a committed FAISS index whenever one exists for this embedder — no
    # env var needed. This is what makes the cloud work: it skips embedding 250
    # passages at startup (which fails on a free-tier cold start). Set
    # FORCE_REBUILD=1 locally after a re-ingest to rebuild from fresh passages.
    mm_dir, bl_dir = _prebuilt_dirs(embeddings)
    if (not FORCE_REBUILD and VECTORSTORE == "faiss"
            and mm_dir.exists() and bl_dir.exists()):
        try:
            return mm.load(mm_dir), bl.load(bl_dir)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Prebuilt index load failed ({exc}); rebuilding.")

    passages = load_passages()
    return mm.build(passages), bl.build(passages)


def save_prebuilt_index() -> None:
    """Build the FAISS index from the current passages + embedder and save it,
    so it can be committed and loaded on a GPU-less / flaky-network host."""
    embeddings = get_embeddings()
    passages = load_passages()
    mm_dir, bl_dir = _prebuilt_dirs(embeddings)
    Pipeline("multimodal", True, embeddings, get_vlm()).build(passages).save(mm_dir)
    Pipeline("text-only baseline", False, embeddings, get_vlm()).build(passages).save(bl_dir)
    print(f"[prebuilt] saved index for {type(embeddings).__name__} -> {mm_dir}, {bl_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "save-index":
        save_prebuilt_index()
    else:
        build_index()
