"""Local-first knowledge index: SQLite FTS5 (true BM25) + float32 vectors, fused with RRF. No network at query time."""
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np

RRF_K = 60


def _norm(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    return a / n if n else a


def _fts_query(q: str) -> str:
    toks = re.findall(r"[A-Za-z0-9]+", q.lower())
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(toks))


def build_index(chunks_path, db_path, embed_fn, batch_size: int = 64) -> dict:
    """Deterministic rebuild from chunks.jsonl. embed_fn: list[str] -> list[vector]. Returns counts."""
    db_path = Path(db_path)
    tmp = db_path.with_name(db_path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    stats = {"read": 0, "indexed": 0, "invalid": 0, "duplicate": 0}
    rows, seen = [], set()
    with open(chunks_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            stats["read"] += 1
            try:
                rec = json.loads(line)
                text = rec["text"].strip()
                meta = rec["metadata"]
                source, page, chunk = str(meta["source"]), int(meta["page"]), int(meta["chunk"])
                if not text or not source or page < 1 or chunk < 1:
                    raise ValueError
            except (ValueError, KeyError, TypeError, AttributeError):
                stats["invalid"] += 1
                continue
            h = hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()
            if h in seen:
                stats["duplicate"] += 1
                continue
            seen.add(h)
            rows.append((source, page, chunk, text, h))
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(
            "CREATE TABLE chunks(id INTEGER PRIMARY KEY, source TEXT NOT NULL, page INTEGER NOT NULL, "
            "chunk INTEGER NOT NULL, text TEXT NOT NULL, text_hash TEXT NOT NULL UNIQUE, vec BLOB NOT NULL);"
            "CREATE VIRTUAL TABLE fts USING fts5(text, tokenize='porter unicode61');")
        dim = None
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            vecs = embed_fn([r[3] for r in batch])
            if len(vecs) != len(batch):
                raise RuntimeError("embedder returned a different number of vectors than texts")
            for r, v in zip(batch, vecs):
                a = _norm(v)
                if dim is None:
                    dim = a.shape[0]
                if a.shape[0] != dim:
                    raise RuntimeError("inconsistent embedding dimensions")
                cur = conn.execute("INSERT INTO chunks(source,page,chunk,text,text_hash,vec) VALUES (?,?,?,?,?,?)",
                                   (r[0], r[1], r[2], r[3], r[4], a.tobytes()))
                conn.execute("INSERT INTO fts(rowid, text) VALUES (?, ?)", (cur.lastrowid, r[3]))
                stats["indexed"] += 1
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, db_path)
    return stats


class LocalIndex:
    def __init__(self, db_path, embed_fn):
        p = Path(db_path)
        if not p.exists():
            raise FileNotFoundError("local knowledge index has not been built")
        self.conn = sqlite3.connect(p, check_same_thread=False)
        self.embed_fn = embed_fn
        rows = self.conn.execute("SELECT id, vec FROM chunks ORDER BY id").fetchall()
        self.ids = np.array([r[0] for r in rows], dtype=np.int64)
        self.matrix = (np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
                       if rows else np.zeros((0, 0), dtype=np.float32))

    def _allowed(self, source):
        if source is None:
            return None
        return {r[0] for r in self.conn.execute("SELECT id FROM chunks WHERE source = ?", (source,))}

    def vector_ranks(self, query: str, k: int, source=None) -> list[int]:
        if not len(self.ids):
            return []
        q = _norm(self.embed_fn([query])[0])
        if not float(np.linalg.norm(q)):
            return []
        sims = self.matrix @ q
        allowed = self._allowed(source)
        out = []
        for i in np.argsort(-sims, kind="stable"):
            cid = int(self.ids[i])
            if allowed is None or cid in allowed:
                out.append(cid)
            if len(out) >= k:
                break
        return out

    def keyword_ranks(self, query: str, k: int, source=None) -> list[int]:
        fq = _fts_query(query)
        if not fq:
            return []
        sql = "SELECT c.id FROM fts JOIN chunks c ON c.id = fts.rowid WHERE fts MATCH ?"
        args: list = [fq]
        if source is not None:
            sql += " AND c.source = ?"
            args.append(source)
        sql += " ORDER BY bm25(fts), c.id LIMIT ?"
        args.append(k)
        return [r[0] for r in self.conn.execute(sql, args)]

    def _hydrate(self, ids: list[int], scores: dict) -> list[dict]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        by = {r[0]: r for r in self.conn.execute(
            f"SELECT id, source, page, chunk, text FROM chunks WHERE id IN ({marks})", ids)}
        return [{"id": i, "content": by[i][4], "source": by[i][1], "page": by[i][2],
                 "chunk_number": by[i][3], "score": scores[i]} for i in ids if i in by]

    def search(self, query: str, mode: str = "hybrid", limit: int = 5, candidate_limit: int = 20,
               source=None, rerank_fn=None) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if limit < 1 or candidate_limit < 1:
            raise ValueError("limit and candidate_limit must be positive")
        if mode not in ("vector", "keyword", "hybrid"):
            raise ValueError("mode must be vector, keyword or hybrid")
        query = query.strip()
        v = self.vector_ranks(query, candidate_limit, source) if mode in ("vector", "hybrid") else []
        kw = self.keyword_ranks(query, candidate_limit, source) if mode in ("keyword", "hybrid") else []
        scores: dict[int, float] = {}
        for ranks in (v, kw):
            for pos, cid in enumerate(ranks, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + pos)
        ids = sorted(scores, key=lambda c: (-scores[c], c))
        results = self._hydrate(ids, scores)
        if rerank_fn is not None and results:
            new = list(rerank_fn(query, [r["content"] for r in results]))
            if len(new) != len(results):
                raise RuntimeError("reranker returned an unexpected number of scores")
            for r, s in zip(results, new):
                r["rerank_score"] = float(s)
            results.sort(key=lambda r: (-r["rerank_score"], r["id"]))
        return results[:limit]

    def search_knowledge(self, query: str, limit: int = 5, rerank: bool = False) -> list[dict]:
        """Same call shape as rag.hybrid.search_knowledge. No local reranker yet, so rerank is ignored."""
        return self.search(query, "hybrid", limit)
