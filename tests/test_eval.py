"""离线评估测试：用 golden QA 集检验「检索召回率」。

即使没有 LLM API key，也能验证：给定问题，知识库能否检索到支撑答案的证据段落。
这是 RAG 系统最重要的质量闸门之一（recall）。有 key 时的完整端到端评估见 eval/run_eval.py。
"""
from __future__ import annotations


def test_retrieval_recall_on_golden_qa(retriever, golden_qa):
    failures = []
    for item in golden_qa:
        docs = retriever.invoke(item["question"])
        joined = "\n".join(d.page_content for d in docs)
        missing = [ev for ev in item["expected_evidence"] if ev not in joined]
        if missing:
            failures.append((item["id"], missing))

    assert not failures, (
        "以下 golden QA 项的证据未被检索召回:\n"
        + "\n".join(f"  - {qid}: 缺失 {miss}" for qid, miss in failures)
    )


def test_all_golden_items_have_evidence(golden_qa):
    for item in golden_qa:
        # 先取出 id 用于报错定位；若连 id 都缺失则回退占位符，避免二次 KeyError
        qid = item.get("id", "<无 id>")
        # question 是召回测试的输入字段：缺失虽会被 retriever.invoke 以 KeyError 暴露，
        # 但此处提前校验能给出「指向具体哪条题」的清晰报错，而非裸 KeyError。
        assert item.get("question"), f"{qid} 缺少 question（召回测试的输入）"
        # expected_evidence 是召回测试的「被断言对象」：缺失会导致下游测试假绿，
        # 因此必须由数据体检显式守住。
        assert item.get("expected_evidence"), f"{qid} 缺少 expected_evidence（无可校验证据）"
