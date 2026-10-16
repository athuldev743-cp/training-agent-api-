"""Local-first retrieval with a two-layer evidence gate (no per-query labels):
1) idf-weighted key-term coverage of the top chunks, 2) cross-encoder relevance of the best chunk."""
import json, math, pathlib, re
from rag.local_index import LocalIndex

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = ROOT / "knowledge" / "local_index.db"
GATE_FILE = ROOT / "eval" / "results" / "step7_gate2.json"
DEFAULT_GATE = {"cov": 0.5, "ce": 0.0}
STOP = set("should would could what which how does much many with from that this have about across between during "
           "each into their they when while without your than then also best good user say says make take need "
           "like more most some such over under after before again only very can will".split())
_c = {}


def _embedder():
    from fastembed import TextEmbedding
    m = TextEmbedding("BAAI/bge-small-en-v1.5")
    return lambda texts: [list(map(float, v)) for v in m.embed(texts)]


def get_index():
    if "idx" not in _c:
        _c["idx"] = LocalIndex(DB, _embedder())
    return _c["idx"]


def stem(t):
    for suf in ("ings", "ing", "ers", "er", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[:-len(suf)]
    return t


def pre(t):
    return stem(t)[:5]


def prefixes(text):
    return {pre(t) for t in re.findall(r"[a-z]{4,}", text.lower())}


def _stats(idx):
    if "df" not in _c:
        df, n = {}, 0
        for _id, text in idx.conn.execute("SELECT id, text FROM chunks"):
            n += 1
            for p in prefixes(text):
                df[p] = df.get(p, 0) + 1
        _c["df"], _c["n"] = df, n
    return _c["df"], _c["n"]


def query_terms(idx, query):
    df, n = _stats(idx)
    terms = {}
    for t in re.findall(r"[a-z]{4,}", query.lower()):
        if t in STOP:
            continue
        p = pre(t)
        terms[p] = math.log((n + 1) / (1 + df.get(p, 0)))
    return terms


def coverage(idx, query, chunks):
    terms = query_terms(idx, query)
    if not terms:
        return 0.0
    have = set()
    for c in chunks:
        have |= prefixes(c["content"])
    return sum(w for p, w in terms.items() if p in have) / sum(terms.values())


def ce_scores(query, docs):
    if "ce" not in _c:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        _c["ce"] = TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
    return [float(s) for s in _c["ce"].rerank(query, docs)]


def features(idx, query, force=False):
    top = idx.search(query, "hybrid", limit=3)
    if not top:
        return 0.0, -99.0
    cov = coverage(idx, query, top)
    if cov < 0.3 and not force:
        return cov, -99.0
    return cov, max(ce_scores(query, [c["content"] for c in top]))


def load_gate():
    try:
        return json.loads(GATE_FILE.read_text(encoding="utf-8"))["gate"]
    except Exception:
        return DEFAULT_GATE


def accept(cov, ce, gate):
    return cov >= gate["cov"] and ce >= gate["ce"]


def local_search(query, limit=3, rerank=False):
    idx = get_index()
    if not accept(*features(idx, query), load_gate()):
        return []
    return idx.search(query, "hybrid", limit=limit)
