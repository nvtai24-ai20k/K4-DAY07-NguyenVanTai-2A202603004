"""Benchmark 5 câu hỏi của nhóm trên corpus data/thu-vien-uth/ (Lab 07, K4-L3A).

Cách chạy:
    python bench.py                       # chiến lược của tôi (STRATEGY bên dưới), ghi ket_qua_benchmark.txt
    python bench.py --strategy all        # chạy cả 4 chiến lược để so sánh trong nhóm
    python bench.py --embedder mock       # ép backend embedding: openai | local | mock (mặc định: EMBEDDING_PROVIDER)
    python bench.py --llm extractive      # ép LLM: openai | extractive (mặc định: openai nếu có OPENAI_API_KEY)

Luồng: đọc .md -> tách frontmatter -> chunk phần thân (NGOÀI store) -> mỗi chunk một Document
-> EmbeddingStore -> search_with_filter() -> top-3 -> KnowledgeBaseAgent.build_prompt() -> llm_fn.

Gọi OpenAI được cache theo sha256(model + nội dung) trong .cache/ nên chạy lại không tốn thêm token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src import (
    ChunkingStrategyComparator,
    Document,
    EmbeddingStore,
    FixedSizeChunker,
    KnowledgeBaseAgent,
    RecursiveChunker,
    SentenceChunker,
    compute_similarity,
)
from src.embeddings import EMBEDDING_PROVIDER_ENV, OPENAI_EMBEDDING_MODEL, LocalEmbedder, OpenAIEmbedder, _mock_embed
from src.llm import OpenAIChatLLM

CORPUS_DIR = Path("data/thu-vien-uth")
OUTPUT_FILE = Path("ket_qua_benchmark.txt")
CACHE_DIR = Path(".cache")
TOP_K = 3

# Dòng duy nhất mỗi thành viên đổi: chiến lược chunking của mình.
STRATEGY = "heading"


# ---------------------------------------------------------------------------
# Chiến lược custom: chunk theo heading/section
# ---------------------------------------------------------------------------
class HeadingChunker:
    """Chia văn bản quy định theo heading Markdown (#, ##, ###).

    Lý do thiết kế: nội quy/quy định được người soạn chia sẵn theo Điều/mục, mỗi mục là một đơn vị
    ngữ nghĩa trọn vẹn. Mỗi chunk được gắn "đường dẫn tiêu đề" (Tên văn bản > Mục > Tiểu mục) để
    chunk con không mất ngữ cảnh; mục nào dài hơn max_chars thì hạ xuống RecursiveChunker và gắn
    lại đường dẫn tiêu đề vào từng mảnh con.
    """

    HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

    def __init__(self, max_chars: int = 600) -> None:
        self.max_chars = max_chars

    def chunk(self, text: str) -> list[str]:
        sections: list[tuple[str, list[str]]] = []
        stack: list[tuple[int, str]] = []
        body: list[str] = []

        def flush() -> None:
            content = "\n".join(body).strip()
            if content:
                sections.append((" > ".join(title for _, title in stack), content))
            body.clear()

        for line in text.splitlines():
            match = self.HEADING.match(line)
            if match:
                flush()
                level = len(match.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, match.group(2)))
            else:
                body.append(line)
        flush()

        chunks: list[str] = []
        for path, content in sections:
            candidate = f"{path}\n{content}" if path else content
            if len(candidate) <= self.max_chars:
                chunks.append(candidate)
                continue
            sub_size = max(100, self.max_chars - len(path) - 1)
            for piece in RecursiveChunker(chunk_size=sub_size).chunk(content):
                chunks.append(f"{path}\n{piece}" if path else piece)
        return chunks


STRATEGIES = {
    "fixed": lambda: FixedSizeChunker(chunk_size=400, overlap=80),
    "sentence": lambda: SentenceChunker(max_sentences_per_chunk=3),
    "recursive": lambda: RecursiveChunker(chunk_size=400),
    "heading": lambda: HeadingChunker(max_chars=600),
}


# ---------------------------------------------------------------------------
# 5 benchmark query của nhóm (gold answer trích từ corpus)
# ---------------------------------------------------------------------------
@dataclass
class Query:
    text: str
    gold_answer: str
    gold_docs: list[str]
    answer_keys: list[str]  # chuỗi nguyên văn trong corpus: ngữ cảnh truy xuất được phải chứa một trong số này
    answer_patterns: list[str]  # regex: câu trả lời của agent (đã chuẩn hoá) phải khớp TẤT CẢ
    metadata_filter: dict | None = None


QUERIES = [
    Query(
        text="Trả sách quá hạn thì bị phạt bao nhiêu tiền?",
        gold_answer="1.000đ/cuốn/ngày (Nội quy Thư viện 01/11/2022, Điều 7; trang Phục vụ mượn – trả tài liệu).",
        gold_docs=["noi-quy-thu-vien", "phuc-vu-muon-tra-tai-lieu"],
        answer_keys=["1.000đ/cuốn/ngày", "1000 đồng/1 cuốn/1 ngày"],
        answer_patterns=[r"(?<!\d)1000(?!\d)"],
    ),
    Query(
        text="Sách mượn về nhà được gia hạn mấy lần, mỗi lần bao lâu?",
        gold_answer="Được gia hạn 01 lần, thời gian 45 ngày (trang Phục vụ mượn – trả tài liệu).",
        gold_docs=["phuc-vu-muon-tra-tai-lieu"],
        answer_keys=["gia hạn: 01 lần"],
        answer_patterns=[r"(?<!\d)0?1 lần|một lần", r"45 ngày"],
    ),
    Query(
        text="Mỗi bạn đọc được mượn tối đa bao nhiêu tài liệu về nhà?",
        gold_answer="Với sinh viên: 05 tài liệu tiếng Việt + 03 tài liệu tiếng Anh (tham khảo), tức 8 cuốn.",
        gold_docs=["quy-dinh-muon-tra-sinh-vien", "noi-quy-thu-vien", "phuc-vu-muon-tra-tai-lieu"],
        answer_keys=["Tiếng Việt: 05 tài liệu", "5 cuốn Tiếng Việt"],
        # Con số phải gắn với đúng loại tài liệu: 5 = tiếng Việt, 3 = tiếng Anh/ngoại văn
        # (bắt được lỗi "03 tài liệu cho học viên cao học" hay nhầm sang "đọc tại chỗ: 03 tài liệu").
        answer_patterns=[
            r"(?<!\d)0?5 (tài liệu|cuốn|quyển)[^.;+]{0,30}tiếng việt|tiếng việt[^.;+]{0,30}(?<!\d)0?5 (tài liệu|cuốn|quyển)",
            r"(?<!\d)0?3 (tài liệu|cuốn|quyển)[^.;+]{0,30}(tiếng anh|ngoại văn)"
            r"|(tiếng anh|ngoại văn)[^.;+]{0,30}(?<!\d)0?3 (tài liệu|cuốn|quyển)",
        ],
        metadata_filter={"audience": "student"},
    ),
    Query(
        text="Khi vào phòng đọc được mang theo những gì?",
        gold_answer="Chỉ được mang: máy tính cá nhân, sách, tập vở và dụng cụ học tập (Nội quy Điều 4; trang Phục vụ phòng đọc).",
        gold_docs=["noi-quy-thu-vien", "phuc-vu-phong-doc"],
        answer_keys=["Máy tính cá nhân, Sách, tập vở"],
        answer_patterns=[r"máy tính cá nhân", r"sách", r"tập vở", r"dụng cụ học tập"],
    ),
    Query(
        text="Muốn kiểm tra tỉ lệ trùng lặp cho khóa luận, đồ án thì dùng dịch vụ nào?",
        gold_answer="Dịch vụ quét trùng lặp bằng phần mềm Turnitin; SV, GV-CB dùng qua Hệ thống đào tạo trực tuyến, người dùng khác đến văn phòng LIC.",
        gold_docs=["dich-vu-quet-trung-lap"],
        answer_keys=["Turnitin"],
        answer_patterns=[r"turnitin"],
    ),
]

# Cặp câu cho bài tập dự đoán độ tương tự (dự đoán ghi trước khi chạy, xem REPORT_CANHAN mục 4).
SIMILARITY_PAIRS = [
    ("Sinh viên được mượn tối đa 8 cuốn sách về nhà.", "Mỗi sinh viên có thể mang về nhà nhiều nhất tám quyển sách.", "cao"),
    ("The overdue fine is 1,000 VND per book per day.", "Quá hạn phải nộp phạt 1.000đ/cuốn/ngày.", "cao"),
    ("Thư viện có phòng máy tính để tra cứu.", "Giá vé máy bay đi Hà Nội tăng mạnh dịp Tết.", "thấp"),
    ("Giảng viên được mượn 10 tài liệu.", "Sinh viên được mượn 5 tài liệu tiếng Việt.", "cao"),
    ("Tôi muốn mượn sách ở thư viện.", "Tôi không muốn mượn sách ở thư viện.", "cao"),
]


# ---------------------------------------------------------------------------
# Nạp corpus
# ---------------------------------------------------------------------------
def parse_markdown(path: Path) -> tuple[dict, str]:
    """Tách YAML frontmatter (key: value đơn giản) khỏi phần thân."""
    text = path.read_text(encoding="utf-8")
    metadata: dict[str, str] = {}
    if text.startswith("---"):
        _, front, body = text.split("---", 2)
        for line in front.splitlines():
            match = re.match(r"^(\w+):\s*(.*?)\s*(#.*)?$", line)
            if match:
                metadata[match.group(1)] = match.group(2).strip().strip('"')
        return metadata, body.strip()
    return metadata, text.strip()


def load_chunks(chunker) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(CORPUS_DIR.glob("*.md")):
        frontmatter, body = parse_markdown(path)
        for index, chunk in enumerate(chunker.chunk(body)):
            documents.append(
                Document(
                    id=f"{path.stem}#{index}",
                    content=chunk,
                    # frontmatter trải vào MỌI chunk; doc_id trỏ về file gốc.
                    metadata={**frontmatter, "doc_id": path.stem, "chunk_index": index},
                )
            )
    return documents


# ---------------------------------------------------------------------------
# Backend: embedding + LLM, gọi OpenAI có cache trên đĩa
# ---------------------------------------------------------------------------
class _DiskCache:
    """dict lưu thành JSON trong .cache/, khoá = sha256(model + nội dung)."""

    def __init__(self, name: str, model: str) -> None:
        self.path = CACHE_DIR / name
        self.model = model
        self.data: dict = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}\n{text}".encode("utf-8")).hexdigest()

    def save(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False), encoding="utf-8")


class CachedOpenAIEmbedder:
    """OpenAIEmbedder + cache; prefetch() gom nhiều text vào một request embeddings."""

    def __init__(self, model_name: str) -> None:
        self.inner = OpenAIEmbedder(model_name=model_name)
        self._backend_name = model_name
        self.cache = _DiskCache("openai_embeddings.json", model_name)
        self.usage_tokens = 0
        self.api_calls = 0

    def prefetch(self, texts: list[str]) -> None:
        missing = list(dict.fromkeys(t for t in texts if self.cache.key(t) not in self.cache.data))
        for start in range(0, len(missing), 100):
            batch = missing[start : start + 100]
            response = self.inner.client.embeddings.create(model=self.inner.model_name, input=batch)
            self.api_calls += 1
            self.usage_tokens += response.usage.total_tokens
            for text, item in zip(batch, response.data):
                self.cache.data[self.cache.key(text)] = item.embedding
        if missing:
            self.cache.save()

    def __call__(self, text: str) -> list[float]:
        key = self.cache.key(text)
        if key not in self.cache.data:
            self.prefetch([text])
        return self.cache.data[key]


class CachedLLM:
    """Bọc OpenAIChatLLM: prompt đã hỏi rồi thì lấy lại câu trả lời từ cache."""

    def __init__(self, llm: OpenAIChatLLM) -> None:
        self.inner = llm
        self._backend_name = llm._backend_name
        self.cache = _DiskCache("openai_llm.json", llm.model_name)
        self.api_calls = 0

    @property
    def usage_tokens(self) -> int:
        return self.inner.usage_tokens

    def __call__(self, prompt: str) -> str:
        key = self.cache.key(prompt)
        if key not in self.cache.data:
            self.cache.data[key] = self.inner(prompt)
            self.api_calls += 1
            self.cache.save()
        return self.cache.data[key]


def make_embedder(name: str | None):
    load_dotenv(dotenv_path=Path(".env"), override=False)
    provider = (name or os.getenv(EMBEDDING_PROVIDER_ENV) or "local").strip().lower()
    try:
        if provider == "openai":
            return CachedOpenAIEmbedder(os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        if provider == "local":
            return LocalEmbedder()
    except Exception as error:  # thiếu thư viện, thiếu key hoặc không tải được model
        print(f"[warn] embedder '{provider}' không dùng được ({error}); quay về mock.", file=sys.stderr)
    return _mock_embed


def make_llm(name: str | None):
    load_dotenv(dotenv_path=Path(".env"), override=False)
    choice = name or ("openai" if os.getenv("OPENAI_API_KEY") else "extractive")
    if choice == "openai":
        try:
            return CachedLLM(OpenAIChatLLM())
        except Exception as error:
            print(f"[warn] OpenAI LLM không dùng được ({error}); dùng extractive_llm.", file=sys.stderr)
    return extractive_llm


# ---------------------------------------------------------------------------
# "LLM" trích xuất: không có API key nên dùng bộ trích xuất đoạn văn thay LLM thật
# ---------------------------------------------------------------------------
STOPWORDS = set("thì bị bao nhiêu được mấy lần mỗi khi những gì nào muốn cho và của là có các một để với về".split())


def _tokens(text: str) -> set[str]:
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS]
    return set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}


def extractive_llm(prompt: str) -> str:
    """Chọn đoạn (paragraph) trong NGỮ CẢNH trùng từ khóa với CÂU HỎI nhiều nhất, trả kèm trích dẫn [n].

    Đây KHÔNG phải LLM sinh văn bản: nó chỉ chép lại nguyên văn một đoạn ngữ cảnh, nên không bịa,
    nhưng cũng không tổng hợp được nhiều đoạn.
    """
    context = prompt.split("NGỮ CẢNH:", 1)[1].split("CÂU HỎI:", 1)[0]
    question = prompt.split("CÂU HỎI:", 1)[1].split("TRẢ LỜI:", 1)[0]
    q_tokens = _tokens(question)
    best: tuple[float, str] = (0.0, "")
    for block in re.split(r"\n(?=\[\d+\] \(nguồn:)", context.strip()):
        header, _, content = block.partition("\n")
        number = re.match(r"\[(\d+)\]", header).group(1)
        for paragraph in re.split(r"\n\s*\n", content):
            lines = [line for line in paragraph.splitlines() if line.strip()]
            if lines and " > " in lines[0]:  # bỏ dòng đường dẫn tiêu đề khi chấm điểm
                lines = lines[1:]
            candidate = " ".join(line.strip() for line in lines)
            if not candidate:
                continue
            overlap = len(q_tokens & _tokens(candidate)) - 0.1 * (int(number) - 1)
            if overlap > best[0]:
                best = (overlap, f"{candidate} [{number}]")
    return best[1] or "Không tìm thấy thông tin liên quan trong cơ sở tri thức."


# ---------------------------------------------------------------------------
# Chạy & chấm điểm
# ---------------------------------------------------------------------------
def normalize_answer(text: str) -> str:
    """Chữ thường, bỏ dấu chấm phân cách hàng nghìn ("1.000" -> "1000"), gộp khoảng trắng."""
    text = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", text.lower())
    return re.sub(r"\s+", " ", text)


def score_query(query: Query, results: list[dict], answer: str) -> dict:
    ranks_with_key = [
        rank for rank, r in enumerate(results, start=1) if any(k in r["content"] for k in query.answer_keys)
    ]
    first_hit = ranks_with_key[0] if ranks_with_key else None
    normalized = normalize_answer(answer)
    agent_ok = all(re.search(pattern, normalized) for pattern in query.answer_patterns)
    doc_ids = [r["metadata"]["doc_id"] for r in results]
    naive = 2 if doc_ids and doc_ids[0] in query.gold_docs else (1 if set(doc_ids) & set(query.gold_docs) else 0)
    content = 0 if first_hit is None else (2 if first_hit == 1 and agent_ok else 1)
    return {"first_hit": first_hit, "agent_ok": agent_ok, "naive": naive, "content": content}


def run_strategy(name: str, embedder, llm, out) -> dict:
    chunker = STRATEGIES[name]()
    docs = load_chunks(chunker)
    if hasattr(embedder, "prefetch"):  # một request cho cả corpus thay vì một request mỗi chunk
        embedder.prefetch([d.content for d in docs] + [q.text for q in QUERIES])
    store = EmbeddingStore(collection_name=f"bench_{name}", embedding_fn=embedder)
    store.add_documents(docs)
    agent = KnowledgeBaseAgent(store=store, llm_fn=llm)
    lengths = [len(d.content) for d in docs]

    out(f"\n{'=' * 78}\nCHIẾN LƯỢC: {name}  ({chunker.__class__.__name__})")
    out(f"Số chunk đã nạp: {store.get_collection_size()} | avg_length={sum(lengths) / len(lengths):.0f} "
        f"| min={min(lengths)} | max={max(lengths)}")

    totals = {"naive": 0, "content": 0, "hit_top3": 0}
    for number, query in enumerate(QUERIES, start=1):
        runs = [("có filter" if query.metadata_filter else "không filter", query.metadata_filter)]
        if query.metadata_filter:
            runs.append(("A/B — KHÔNG filter", None))
        for label, metadata_filter in runs:
            results = store.search_with_filter(query.text, top_k=TOP_K, metadata_filter=metadata_filter)
            answer = agent.llm_fn(agent.build_prompt(query.text, results))
            scores = score_query(query, results, answer)
            out(f"\nQ{number} [{label}] {query.text}")
            if metadata_filter:
                out(f"    metadata_filter={metadata_filter}")
            for rank, r in enumerate(results, start=1):
                has_key = "✓" if any(k in r["content"] for k in query.answer_keys) else " "
                preview = r["content"].replace("\n", " ")[:110]
                out(f"  {rank}. {has_key} score={r['score']:.3f} doc={r['metadata']['doc_id']} "
                    f"aud={r['metadata'].get('audience')} | {preview}")
            out(f"  Agent: {' '.join(answer.split())[:400]}")
            out(f"  Gold : {query.gold_answer}")
            out(f"  Điểm: theo doc_id={scores['naive']}/2 | theo nội dung={scores['content']}/2 "
                f"| chunk chứa đáp án ở hạng={scores['first_hit']} | agent đúng={scores['agent_ok']}")
            if label.startswith("A/B"):
                continue
            totals["naive"] += scores["naive"]
            totals["content"] += scores["content"]
            totals["hit_top3"] += scores["first_hit"] is not None

    out(f"\nTỔNG [{name}]: theo doc_id={totals['naive']}/10 | theo nội dung={totals['content']}/10 "
        f"| có chunk chứa đáp án trong top-3: {totals['hit_top3']}/5")
    return {"chunks": len(docs), "avg_length": sum(lengths) / len(lengths), **totals}


def run_baseline(out) -> None:
    out(f"\n{'=' * 78}\nBASELINE — ChunkingStrategyComparator().compare(chunk_size=200), bỏ frontmatter")
    for doc_id in ["noi-quy-thu-vien", "quy-dinh-muon-tra-sinh-vien", "phuc-vu-muon-tra-tai-lieu"]:
        _, body = parse_markdown(CORPUS_DIR / f"{doc_id}.md")
        result = ChunkingStrategyComparator().compare(body, chunk_size=200)
        out(f"  {doc_id} ({len(body)} ký tự)")
        for name, stats in result.items():
            out(f"    {name:13} count={stats['count']:3} avg_length={stats['avg_length']:.1f}")


def run_similarity(embedder, out) -> None:
    out(f"\n{'=' * 78}\nDỰ ĐOÁN ĐỘ TƯƠNG TỰ (compute_similarity)")
    if hasattr(embedder, "prefetch"):
        embedder.prefetch([sentence for a, b, _ in SIMILARITY_PAIRS for sentence in (a, b)])
    for number, (a, b, prediction) in enumerate(SIMILARITY_PAIRS, start=1):
        score = compute_similarity(embedder(a), embedder(b))
        out(f"  {number}. dự đoán={prediction:5} thực tế={score:.3f} | {a} || {b}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default=STRATEGY, choices=[*STRATEGIES, "all"])
    parser.add_argument("--embedder", choices=["openai", "local", "mock"], default=None)
    parser.add_argument("--llm", choices=["openai", "extractive"], default=None)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args()

    lines: list[str] = []

    def out(line: str = "") -> None:
        print(line)
        lines.append(line)

    embedder = make_embedder(args.embedder)
    llm = make_llm(args.llm)
    backend = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    out(f"Corpus: {CORPUS_DIR} | Embedding backend: {backend} | top_k={TOP_K}")
    if llm is extractive_llm:
        out("LLM: extractive_llm (trích nguyên văn đoạn ngữ cảnh khớp từ khóa nhất — không phải LLM sinh văn bản)")
    else:
        out(f"LLM: {llm._backend_name} (OpenAI chat, temperature=0, max_tokens={llm.inner.max_tokens})")

    names = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
    summary = {name: run_strategy(name, embedder, llm, out) for name in names}

    if len(summary) > 1:
        out(f"\n{'=' * 78}\nTÓM TẮT")
        out(f"  {'chiến lược':10} {'chunks':>6} {'avg_len':>7} {'doc_id/10':>9} {'nội dung/10':>11} {'top3/5':>6}")
        for name, s in summary.items():
            out(f"  {name:10} {s['chunks']:6} {s['avg_length']:7.0f} {s['naive']:9} {s['content']:11} {s['hit_top3']:6}")
        run_baseline(out)
    run_similarity(embedder, out)

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nĐã ghi {args.output}")
    # Token của lần chạy này (0 khi mọi thứ đã có trong .cache/) — không ghi vào file kết quả.
    for label, backend_obj in (("embedding", embedder), ("LLM", llm)):
        if hasattr(backend_obj, "api_calls"):
            print(f"OpenAI {label}: {backend_obj.api_calls} request mới, {backend_obj.usage_tokens} token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
