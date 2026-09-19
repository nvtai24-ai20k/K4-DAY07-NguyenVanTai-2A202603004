# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Văn Tài — MSSV 2A202603004
**Nhóm:** RAUMAMIENTAY
**Ngày:** 2026-09-19

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding gần như cùng hướng trong không gian ngữ nghĩa (góc giữa chúng nhỏ, cosine gần 1), tức hai đoạn văn nói về cùng một ý dù có thể dùng từ ngữ khác nhau. Cosine chỉ so hướng, không quan tâm độ dài vector.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Sinh viên được mượn tối đa 8 cuốn sách về nhà."
- Câu B: "Mỗi sinh viên có thể mang về nhà nhiều nhất tám quyển sách."
- Tại sao tương đồng: khác từ vựng ("tối đa" / "nhiều nhất", "8 cuốn" / "tám quyển", "mượn" / "mang về") nhưng cùng nghĩa. Một embedding tốt phải cho điểm cao, và điều này chứng minh nó hiểu nghĩa chứ không so khớp từ. Kết quả đo thật ở mục 4, cặp 1.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Thư viện có phòng máy tính để tra cứu."
- Câu B: "Giá vé máy bay đi Hà Nội tăng mạnh dịp Tết."
- Tại sao khác: hai chủ đề không liên quan (dịch vụ thư viện và giá vé máy bay), gần như không chung khái niệm nào.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Euclid bị ảnh hưởng bởi độ lớn (norm) của vector, mà norm thường phản ánh độ dài hay tần suất từ chứ không phản ánh nghĩa. Cosine chỉ đo hướng nên so sánh công bằng giữa câu ngắn và đoạn dài. Khi vector đã chuẩn hoá (`||v|| = 1`), cosine bằng đúng dot product (tính rẻ) và xếp hạng tương đương Euclid, vì `||a−b||² = 2 − 2·cos`.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Phép tính: `ceil((10000 − 50) / (500 − 50)) = ceil(9950 / 450) = ceil(22.11) = 23`
> Kiểm lại bằng `FixedSizeChunker(chunk_size=500, overlap=50).chunk('a'*10000)` cũng ra 23 chunk.
> **Đáp án: 23 chunks.**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Bước trượt giảm từ 450 xuống 400, nên số chunk tăng lên `ceil(9900 / 400) = ceil(24.75) = 25` (đã kiểm bằng code, ra 25). Overlap lớn hơn giúp một câu hoặc một ý nằm ở ranh giới hai chunk vẫn xuất hiện trọn vẹn trong ít nhất một chunk, nên mỗi thông tin có thêm cơ hội lọt top-k. Cái giá là tốn thêm chunk và thêm nội dung trùng lặp trong ngữ cảnh.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tách câu bằng regex lookbehind `(?<=[.!?])\s+`: cắt tại khoảng trắng **sau** dấu câu nên dấu `.`, `!`, `?` vẫn được giữ, không bị nuốt như khi split bằng `[.!?]\s+`. Regex này bao được cả `". "`, `"! "`, `"? "` và `".\n"`. Sau đó strip từng câu, bỏ câu rỗng, gom mỗi `max_sentences_per_chunk` câu thành một chunk. Text rỗng hoặc chỉ có khoảng trắng thì trả `[]`. **Edge case chưa xử lý:** chữ viết tắt (`TS. Nguyễn`, `v.v. `) và số có dấu chấm theo sau là khoảng trắng sẽ bị cắt sai. Dòng Markdown không có dấu câu (heading, bullet) thì bị dính vào câu kế tiếp.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán có hai chiều. **Đệ quy xuống:** thử separator theo thứ tự `\n\n → \n → ". " → " " → ""`, mảnh nào vẫn dài hơn `chunk_size` thì gọi lại `_split` với các separator còn lại. **Gom lên:** các mảnh nhỏ liền kề được nối lại cho tới sát `chunk_size`, nhờ vậy không sinh chunk vụn 5–10 ký tự. Separator được giữ ở cuối mảnh bên trái, nên nối các mảnh lại sẽ khôi phục đúng văn bản gốc. Có 3 base case: (1) text ≤ `chunk_size` thì trả luôn; (2) hết separator hoặc gặp separator `""` thì cắt cứng theo `chunk_size`, đây là nhánh cho `separators=[]`; (3) separator hiện tại không có trong text thì chuyển sang separator kế tiếp.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu in-memory, không dùng nhánh ChromaDB (không test nào cần, và bật `_use_chroma` chỉ khi máy tình cờ có chromadb sẽ làm hành vi phụ thuộc máy chấm). `_make_record` chuẩn hoá mỗi `Document` thành một record `{id, content, metadata, embedding}`. Metadata được **copy** chứ không dùng chung object với người gọi, và luôn có khoá `doc_id` (mặc định là `doc.id`; bench.py đặt `doc_id` = tên file gốc cho mọi chunk `file#i`). `search` embed câu hỏi một lần rồi tính dot product với từng record. Mọi embedder trong repo đều trả vector đã chuẩn hoá, nên dot product = cosine. Kết quả sắp giảm dần, cắt `top_k` và bỏ trường `embedding` cho gọn output.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> **Lọc trước, search sau.** Tôi lọc tập record theo `metadata_filter` rồi chạy chung helper `_search_records` với `search()`. Nếu lấy top-k trước rồi mới lọc, k slot có thể bị tài liệu sai đối tượng chiếm hết và kết quả còn 0 dù store vẫn có tài liệu hợp lệ. Vì hai hàm dùng chung đường code nên `search_with_filter(filter=None)` luôn khớp `search()`. Tôi mở rộng thêm: giá trị filter có thể là list, nghĩa là "một trong các giá trị", ví dụ `{"audience": ["student", "all"]}`. `delete_document` giữ lại các record có `metadata['doc_id'] != doc_id`, rồi trả `True` nếu kích thước store giảm.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Ba nhịp: `store.search(question, top_k)`, rồi `build_prompt`, rồi `llm_fn`. Prompt đánh số từng chunk `[1] [2] [3]` kèm nguồn (`source_url`, nếu không có thì dùng `source`/`doc_id`) và yêu cầu model trích dẫn số đó, nhờ vậy câu trả lời truy vết được về đúng chunk và đúng file (Source Traceability). Ràng buộc chống bịa: chỉ dùng NGỮ CẢNH, không có thông tin thì trả đúng câu "Không tìm thấy thông tin liên quan trong cơ sở tri thức." Khi store rỗng, agent trả luôn câu đó mà không gọi LLM. `build_prompt` là method riêng để bench.py dùng lại cho các câu hỏi có metadata filter. `llm_fn` thật là `OpenAIChatLLM` (`src/llm.py`): `gpt-4.1-nano`, `temperature=0`, `max_tokens=256`. Tôi chọn model nhỏ, không-reasoning để tiết kiệm token. Trong `bench.py`, cả embedding lẫn câu trả lời được cache theo `sha256(model + nội dung)`, nên chạy lại không tốn thêm token.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.11.16, pytest-9.1.1, pluggy-1.6.0
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.06s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Dự đoán được ghi vào `SIMILARITY_PAIRS` trong `bench.py` **trước khi chạy**. Điểm thực tế đo bằng `compute_similarity()` trên embedding chính thức `text-embedding-3-small` (cuối `ket_qua_benchmark.txt`). Cột đối chứng dùng `paraphrase-multilingual-MiniLM-L12-v2` chạy local (cuối `ket_qua_benchmark_local.txt`). Quy ước: "cao" ≥ 0.5, "thấp" < 0.3.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế (OpenAI) | Đối chứng (MiniLM) | Đúng? |
|------|-----------|-----------|---------|--------------|------|-------|
| 1 | Sinh viên được mượn tối đa 8 cuốn sách về nhà. | Mỗi sinh viên có thể mang về nhà nhiều nhất tám quyển sách. | cao | 0.771 | 0.874 | Đúng |
| 2 | The overdue fine is 1,000 VND per book per day. | Quá hạn phải nộp phạt 1.000đ/cuốn/ngày. | cao | 0.603 | 0.790 | Đúng |
| 3 | Thư viện có phòng máy tính để tra cứu. | Giá vé máy bay đi Hà Nội tăng mạnh dịp Tết. | thấp | 0.249 | -0.070 | Đúng |
| 4 | Giảng viên được mượn 10 tài liệu. | Sinh viên được mượn 5 tài liệu tiếng Việt. | cao | 0.800 | 0.512 | Đúng |
| 5 | Tôi muốn mượn sách ở thư viện. | Tôi không muốn mượn sách ở thư viện. | cao | **0.895** | 0.706 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là với `text-embedding-3-small`, cặp 5 gồm hai câu **nghĩa ngược nhau** đạt 0.895, và cặp 4 (khác đối tượng, khác con số) đạt 0.800. Cả hai đều **cao hơn** cặp đồng nghĩa thật ở cặp 1 (0.771). Embedding câu mã hoá chủ yếu **chủ đề và cấu trúc bề mặt** ("ai đó được mượn N tài liệu"), còn phủ định, con số và chủ thể gần như không làm vector đổi hướng. Đây chính là lý do Q3 không filter trả về tài liệu giảng viên "10 tài liệu" ở top-1 (mục 5), và vì sao cần metadata `audience`. Điều thứ hai: thang điểm phụ thuộc model. Cặp không liên quan vẫn được 0.249 với OpenAI nhưng -0.070 với MiniLM, nên không thể dùng một ngưỡng score cố định cho mọi model, chỉ nên so thứ hạng. Để đối chứng, `MockEmbedder` cho 0.109 / -0.067 / -0.000 / 0.096 / 0.326, hoàn toàn không phản ánh nghĩa.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chiến lược của tôi: **HeadingChunker** (`STRATEGY = "heading"` trong `bench.py`, `max_chars=600`, tổng 32 chunk, avg 314 ký tự). Embedding `text-embedding-3-small`, top-3, agent là `KnowledgeBaseAgent` với LLM `gpt-4.1-nano`. "Đúng/sai" của agent được kiểm bằng regex (con số phải gắn đúng loại tài liệu) và tôi đã đọc lại từng câu trả lời. Log đầy đủ ở `ket_qua_benchmark.txt`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Trả sách quá hạn thì bị phạt bao nhiêu tiền? | Phục vụ mượn–trả > Quá hạn: "phạt 1000 đồng/1 cuốn/1 ngày" | 0.667 | Có (hạng 2 là Nội quy Điều 7 "1.000đ/cuốn/ngày") | "Trả sách quá hạn bị phạt 1.000 đồng/1 cuốn/1 ngày [1]" — đúng (2/2) |
| 2 | Sách mượn về nhà được gia hạn mấy lần, mỗi lần bao lâu? | Phục vụ mượn–trả > Mượn về nhà ("Số lần được gia hạn: 01 lần với thời gian 45 ngày") | 0.630 | Có | "Được gia hạn 01 lần với thời gian 45 ngày [1]" — đúng (2/2) |
| 3 | Mỗi bạn đọc được mượn tối đa bao nhiêu tài liệu về nhà? `filter={"audience":"student"}` | Quy định (SV) > 2. Số lượng > 2.1 Tài liệu mượn về nhà ("Tiếng Việt: 05, Tiếng Anh: 03") | 0.590 | Có | "5 tài liệu tiếng Việt và 3 tài liệu tiếng Anh (Tham khảo) [1]" — đúng (2/2) |
| 4 | Khi vào phòng đọc được mang theo những gì? | Phục vụ phòng đọc: mảnh cuối 93 ký tự "Sau khi đọc xong, xếp ghế ngồi gọn gàng…" | 0.705 | **Không**; chunk đáp án ở hạng 3 (0.685) | "máy tính cá nhân, sách, tập vở và dụng cụ học tập [3]" — đúng nhưng nhờ hạng 3 (1/2) |
| 5 | Muốn kiểm tra tỉ lệ trùng lặp cho khóa luận, đồ án thì dùng dịch vụ nào? | Dịch vụ quét trùng lặp > Dịch vụ (có "Turnitin") | 0.648 | Có | "Dùng dịch vụ quét trùng lặp… [1]" — đúng tên dịch vụ nhưng **bỏ sót Turnitin** và cách truy cập, thiếu chi tiết (1/2) |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5 (chấm theo nội dung: **8/10**; chấm theo doc_id: 10/10)

**A/B cho Q3 (bắt buộc):** khi **không** filter, top-1 là chunk giảng viên "Cán bộ, Giảng viên: 10 tài liệu" (0.609). Hai chunk đúng cho sinh viên đứng hạng 2–3 và hoà điểm nhau (0.590). Trước ngữ cảnh lẫn hai đối tượng, `gpt-4.1-nano` từ chối: "Không tìm thấy thông tin liên quan…" (1/2). Có `audience=student` thì chunk 2.1 lên **top-1** và agent trả lời đúng (2/2). Tôi thử thêm `{"audience": ["student", "all"]}` thì chunk chứa đáp án chiếm **hạng 1 và 2**: vừa lọc được bản giảng viên, vừa giữ tài liệu chung.

**Ba phát hiện từ chính kết quả của tôi:**
- **Lỗi của chính chiến lược heading (Q4):** trang "Phục vụ phòng đọc" không có heading con, dài 623 ký tự nên bị hạ xuống recursive và cắt thành 2 mảnh. Mảnh cuối chỉ 93 ký tự, phần lớn là đường dẫn tiêu đề "Phục vụ phòng đọc", nên giống câu hỏi "…phòng đọc…" hơn cả mảnh chứa đáp án. **Sửa:** gộp mảnh đuôi quá ngắn (ví dụ < 150 ký tự) vào mảnh trước, hoặc chỉ gắn tiêu đề khi mảnh đủ dài.
- **Filter cứng làm đổi đáp án (Q1):** không filter thì top-1 là 1.000đ. Với filter `student`, top-3 **không còn** chunk 1.000đ nào mà chỉ còn "500đ" từ trang "Quy định mượn–trả". Hai trang của cùng thư viện mâu thuẫn nhau, và đáp án 1.000đ nằm ở tài liệu `audience=all`. Đây là đánh đổi precision/recall khi filter quá cứng.
- **Đổi embedding thay đổi kết quả nhiều hơn đổi chunker:** cùng heading, MiniLM (`ket_qua_benchmark_local.txt`) để chunk "500đ" thắng ở Q1 (đáp án chỉ ở hạng 3) và chunk 2.2 thắng 2.1 ở Q3. OpenAI đưa cả hai lên top-1, nhưng lại thua ở Q4.

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Từ bảng so sánh trong nhóm (`ket_qua_benchmark_all.txt`), với OpenAI thì `FixedSizeChunker` và `RecursiveChunker` (9/10) nhỉnh hơn heading (8/10). Trong khi đó với MiniLM, `SentenceChunker` lại dẫn đầu. **Thứ hạng chiến lược không ổn định khi đổi embedding**, và với 5 câu hỏi thì chênh 1 điểm chưa đủ để kết luận. Bài học thứ hai: khi retrieval đã đúng, lỗi còn lại nằm ở LLM nhỏ. `gpt-4.1-nano` đọc nhầm danh sách "05 tiếng Việt / 03 tiếng Anh" ở fixed và recursive, trong khi chunk theo heading tách riêng mục 2.1 nên nano trả lời đúng. *[Bổ sung điều học được từ nhóm khác sau buổi demo.]*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 9 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 8 / 10 |
| **Tổng phần cá nhân** | **57 / 60** |
