"""评估脚本：对 golden QA 集跑「检索召回」+（可选）「端到端答案质量」评估。

运行方式：
    python eval/run_eval.py            # 仅检索评估（离线，无需 API key）
    python eval/run_eval.py --llm      # 额外跑 LLM 端到端评估（需要 API key）
    python eval/run_eval.py --judge    # 端到端基础上，用 LLM-as-Judge 对答案打分

说明：
- 检索召回：检查问题对应的证据段落是否出现在 top-k 检索结果中（RAG 质量闸门）。
- 端到端：让真实 Agent 回答问题，检查答案是否包含预期证据。
- LLM-as-Judge：用模型对「答案是否准确、是否基于证据」打 1-5 分。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent.config import get_settings  # noqa: E402
from agent.llm import build_embeddings, build_llm  # noqa: E402
from agent.rag import KnowledgeBase, load_documents, split_documents  # noqa: E402
from agent.agent import build_agent  # noqa: E402

GOLDEN = ROOT / "tests" / "fixtures" / "golden_qa.json"


def retrieval_eval(retriever, golden):
    rows, total_evidence, hit_evidence = [], 0, 0
    for item in golden:
        docs = retriever.invoke(item["question"])
        joined = "\n".join(d.page_content for d in docs)
        missing = [ev for ev in item["expected_evidence"] if ev not in joined]
        total_evidence += len(item["expected_evidence"])
        hit_evidence += len(item["expected_evidence"]) - len(missing)
        rows.append((item["id"], len(missing) == 0, missing))
    recall = hit_evidence / total_evidence if total_evidence else 0.0
    return rows, recall


def llm_eval(agent, golden):
    rows = []
    total_evidence, hit_evidence = 0, 0
    for item in golden:
        result = agent.invoke({"input": item["question"]})
        answer = result["output"]
        missing = [ev for ev in item["expected_evidence"] if ev not in answer]
        total_evidence += len(item["expected_evidence"])
        hit_evidence += len(item["expected_evidence"]) - len(missing)
        rows.append((item["id"], answer, missing))
    coverage = hit_evidence / total_evidence if total_evidence else 0.0
    return rows, coverage


def judge_eval(llm, qa_rows):
    judge_prompt = (
        "请作为严格的评审，对下面的回答打分（1-5 分，只输出数字）：\n"
        "问题: {q}\n回答: {a}\n评分标准：是否准确、是否基于给定证据、是否有幻觉。只回复一个 1-5 的整数。"
    )
    scores = []
    for qid, answer, _ in qa_rows:
        try:
            resp = llm.invoke(judge_prompt.format(q=qid, a=answer))
            text = getattr(resp, "content", str(resp)).strip()
            score = int([c for c in text if c.isdigit()][0])
        except Exception:
            score = 0
        scores.append((qid, score))
    return scores


def main():
    parser = argparse.ArgumentParser(description="AI Research Agent 评估脚本")
    parser.add_argument("--llm", action="store_true", help="运行端到端 LLM 评估（需要 API key）")
    parser.add_argument("--judge", action="store_true", help="在 LLM 评估基础上加 LLM-as-Judge 打分")
    args = parser.parse_args()

    settings = get_settings()
    embeddings = build_embeddings(settings)
    kb = KnowledgeBase.load(settings.kb_path, embeddings)
    retriever = kb.as_retriever(settings.retriever_k)

    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)

    print("=" * 60)
    print("阶段 1/2：检索召回评估（离线）")
    print("=" * 60)
    rrows, recall = retrieval_eval(retriever, golden)
    for qid, ok, missing in rrows:
        status = "OK " if ok else f"MISS {missing}"
        print(f"  [{status}] {qid}")
    print(f"检索召回率 recall = {recall:.2%}")

    if args.llm:
        print("\n" + "=" * 60)
        print("阶段 2/2：端到端 LLM 评估（在线）")
        print("=" * 60)
        try:
            llm = build_llm(settings)
        except RuntimeError as exc:
            print(f"[跳过] 无法初始化 LLM: {exc}")
            return
        agent = build_agent(llm, retriever=retriever, memory_k=settings.memory_k,
                            retriever_k=settings.retriever_k, verbose=False)
        lrows, coverage = llm_eval(agent, golden)
        for qid, answer, missing in lrows:
            status = "OK " if not missing else f"MISS {missing}"
            print(f"  [{status}] {qid}")
            print(f"        A: {answer[:120]}")
        print(f"答案证据覆盖率 = {coverage:.2%}")

        if args.judge:
            print("\n  LLM-as-Judge 打分：")
            scores = judge_eval(llm, lrows)
            avg = sum(s for _, s in scores) / len(scores) if scores else 0
            for qid, s in scores:
                print(f"    {qid}: {s}/5")
            print(f"  平均 Judge 分 = {avg:.2f}/5")
    else:
        print("\n提示：加 --llm 可运行端到端评估，加 --judge 再加模型评分（均需 API key）。")


if __name__ == "__main__":
    main()
