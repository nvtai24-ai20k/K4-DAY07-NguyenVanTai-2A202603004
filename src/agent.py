from typing import Any, Callable

from .store import EmbeddingStore

NO_CONTEXT_ANSWER = "Không tìm thấy thông tin liên quan trong cơ sở tri thức."


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            # Empty store: nothing to ground on, so do not call the LLM at all.
            return NO_CONTEXT_ANSWER
        return self.llm_fn(self.build_prompt(question, results))

    @staticmethod
    def build_prompt(question: str, results: list[dict[str, Any]]) -> str:
        """Number each retrieved chunk [1], [2], ... with its source so answers are traceable."""
        context_blocks = []
        for number, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source_url") or metadata.get("source") or metadata.get("doc_id") or result.get("id")
            context_blocks.append(f"[{number}] (nguồn: {source})\n{result['content'].strip()}")
        context = "\n\n".join(context_blocks)
        return (
            "Bạn là trợ lý trả lời câu hỏi về quy định/dịch vụ đại học.\n"
            "Chỉ sử dụng thông tin trong NGỮ CẢNH bên dưới; không suy đoán hay bổ sung quy định bên ngoài.\n"
            "Trích dẫn số thứ tự đoạn ngữ cảnh, ví dụ [1], sau mỗi ý bạn dùng.\n"
            f'Nếu ngữ cảnh không chứa câu trả lời, hãy trả lời đúng câu: "{NO_CONTEXT_ANSWER}"\n\n'
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {question}\n"
            "TRẢ LỜI:"
        )
