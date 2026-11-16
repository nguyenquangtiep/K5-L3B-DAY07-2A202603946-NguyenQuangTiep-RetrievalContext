"""Test logic vòng lặp agent bằng LLM/retriever GIẢ. Chỉ kiểm tra code, không phải benchmark."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import run_agent  # noqa: E402
from src.llm import LLMError  # noqa: E402

CID = "B-BLLD2019-5057094bf022c67e"
CHUNK = {"id": CID, "text": "Điều 25. Thời gian thử việc ... không quá 60 ngày ...", "source_id": "BLLD2019",
         "source": "Bộ luật Lao động 45/2019/QH14", "page": 12, "metadata": {"page_end": 12, "article": "Điều 25"}}


def tool_msg(query, year=None, idx=1):
    import json
    args = {"query": query, **({"year": year} if year else {})}
    return NS(content=None, tool_calls=[NS(id=f"c{idx}", function=NS(name="search_law", arguments=json.dumps(args)))])


def text_msg(text):
    return NS(content=text, tool_calls=None)


class FakeLLM:
    model = "fake:free"

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, messages, tools=None, tool_choice=None):
        self.calls.append({"messages": [dict(m) for m in messages], "tool_choice": tool_choice})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return NS(message=item, finish_reason="stop", served_model="fake", latency_s=0.0, attempts=[],
                  usage={"prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.0})


class FakeRetriever:
    name = "fake"

    def __init__(self, results):
        self.results, self.calls = results, []

    def search(self, query, *, filters=None, top_k=3):
        self.calls.append({"query": query, "filters": filters, "top_k": top_k})
        return [{**c, "score": 0.9, "rank": i} for i, c in enumerate(self.results[:top_k], 1)]


class AgentLoopTests(unittest.TestCase):
    def test_search_then_cited_answer(self):
        llm = FakeLLM([tool_msg("thử việc", 2019), text_msg(f"Không quá 60 ngày [{CID}].")])
        r = run_agent("Thử việc tối đa?", FakeRetriever([CHUNK]), llm)
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["citations"], [CID])
        self.assertEqual(r["n_searches"], 1)
        self.assertEqual([e for e in r["events"] if e["event"] == "search"][0]["filter"], {"year": 2019})

    def test_hallucinated_citation_becomes_abstain(self):
        llm = FakeLLM([tool_msg("thử việc"), text_msg("60 ngày [B-BLLD2019-ffffffffffffffff].")])
        r = run_agent("Thử việc?", FakeRetriever([CHUNK]), llm)
        self.assertEqual((r["status"], r["reason"]), ("abstain", "invalid_or_missing_citations"))
        self.assertIn("60 ngày", [e for e in r["events"] if e["event"] == "answer"][0]["raw_answer"])

    def test_model_abstains(self):
        llm = FakeLLM([tool_msg("lương tối thiểu vùng I"), text_msg("KHÔNG ĐỦ BẰNG CHỨNG: tài liệu không nêu con số.")])
        r = run_agent("Lương tối thiểu vùng I?", FakeRetriever([CHUNK]), llm)
        self.assertEqual((r["status"], r["reason"]), ("abstain", "model_abstained"))

    def test_search_limit_forces_answer(self):
        llm = FakeLLM([tool_msg("a", idx=1), tool_msg("b", idx=2), text_msg(f"Đáp án [{CID}].")])
        r = run_agent("Q?", FakeRetriever([CHUNK]), llm, max_searches=2)
        self.assertEqual(llm.calls[-1]["tool_choice"], "none")
        self.assertEqual(r["n_searches"], 2)
        self.assertEqual(r["status"], "answered")

    def test_budget_and_dedup(self):
        big = {**CHUNK, "id": "B-BLLD2019-aaaaaaaaaaaaaaaa", "text": "x" * 500}
        llm = FakeLLM([tool_msg("a", idx=1), tool_msg("b", idx=2), text_msg(f"Đáp án [{CID}].")])
        r = run_agent("Q?", FakeRetriever([CHUNK, big]), llm, max_searches=3, budget_chars=300)
        searches = [e for e in r["events"] if e["event"] == "search"]
        self.assertEqual(searches[0]["added_to_context"], [CID])
        self.assertEqual(searches[0]["skipped_budget"], ["B-BLLD2019-aaaaaaaaaaaaaaaa"])
        self.assertEqual(searches[1]["already_in_context"], [CID])
        self.assertLessEqual(r["context_chars"], 300)

    def test_llm_error_recorded_not_faked(self):
        llm = FakeLLM([LLMError("HTTP 429", status=429, attempts=[{"status": 429}])])
        r = run_agent("Q?", FakeRetriever([CHUNK]), llm)
        self.assertEqual(r["status"], "error")
        self.assertEqual(r["answer"], "")

    def test_gold_never_sent(self):
        llm = FakeLLM([tool_msg("q"), text_msg(f"A [{CID}].")])
        run_agent("Thời gian thử việc?", FakeRetriever([CHUNK]), llm, user_info={"year": 2019})
        sent = str(llm.calls[0]["messages"])
        self.assertIn("Bộ luật Lao động 45/2019/QH14", sent)  # thông tin người dùng cung cấp
        self.assertNotIn("evidence", sent)


if __name__ == "__main__":
    unittest.main()
