"""Проверка поисковой выдачи и воспроизводимый реальный прогон Дня 22."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from agent import Agent, AgentSettings, DeepSeekProvider
from document_index import RagIndexService, build_rag_context
from document_index.evaluations import RagEvaluationStore


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="выполнить два реальных DeepSeek-вызова на вопрос")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--strategy", choices=("fixed", "structural", "combined"), default="combined")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show-chunks", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    questions = json.loads((ROOT / "docs" / "day22_control_questions.json").read_text(encoding="utf-8"))["questions"]
    questions = questions[: max(1, min(len(questions), args.limit))]
    service = RagIndexService(ROOT / "rag_documents", ROOT / "data" / "rag_index.sqlite3")
    store = RagEvaluationStore(ROOT / "data" / "rag_evaluations.json")
    existing_ids = {
        item.get("question_id") for item in store.list()
        if item.get("evaluation_mode") == "batch-day22-live"
    }
    agent = Agent(DeepSeekProvider()) if args.live else None
    settings = AgentSettings(
        model="deepseek-v4-flash",
        system_prompt=(
            "Отвечай на вопрос точно и по существу на русском языке. "
            "Не выдумывай точные числа, названия или выводы, если они тебе неизвестны."
        ),
        temperature=0.1,
        top_p=1.0,
        reasoning_enabled=False,
        reasoning_effort="low",
        max_tokens=900,
    )
    missing_sources = 0
    completed = 0
    for index, question in enumerate(questions, 1):
        retrieval = service.retrieve(question["question"], args.strategy, args.top_k)
        expected = {Path(value).name.lower() for value in question.get("expected_sources", [])}
        found = {Path(item["source"]).name.lower() for item in retrieval["chunks"]}
        covered = expected.issubset(found)
        missing_sources += 0 if covered else 1
        best = retrieval["chunks"][0] if retrieval["chunks"] else {}
        print(
            f"[{index:02d}/{len(questions):02d}] {question['id']} | source={'OK' if covered else 'MISS'} "
            f"| best={best.get('source', '—')} | score={float(best.get('score', 0)):.4f}",
            flush=True,
        )
        if args.show_chunks:
            for chunk in retrieval["chunks"]:
                preview = " ".join(str(chunk.get("text", "")).split())[:180]
                print(
                    f"  {chunk.get('strategy')} {float(chunk.get('score', 0)):.4f} "
                    f"{chunk.get('source')} p.{chunk.get('page') or '—'} :: {preview}",
                    flush=True,
                )
        if not args.live or question["id"] in existing_ids:
            continue
        without_rag = agent.reply([], question["question"], settings)
        with_rag = agent.reply(
            [], question["question"], settings,
            policy_guidance=build_rag_context(retrieval),
        )
        store.add({
            "evaluation_mode": "batch-day22-live",
            "conversation_id": None,
            "question_id": question["id"],
            "question": question["question"],
            "expectation": question["expectation"],
            "expected_sources": question.get("expected_sources", []),
            "settings": settings.to_dict(),
            "retrieval": retrieval,
            "without_rag": {
                "content": without_rag.content,
                "reasoning_content": without_rag.reasoning_content,
                "technical": without_rag.technical,
            },
            "with_rag": {
                "content": with_rag.content,
                "reasoning_content": with_rag.reasoning_content,
                "technical": with_rag.technical,
            },
        })
        completed += 1
        print(f"  saved | without={without_rag.technical.get('request_id')} | with={with_rag.technical.get('request_id')}", flush=True)
    print(f"Retrieval coverage: {len(questions) - missing_sources}/{len(questions)}; live comparisons saved now: {completed}")
    return 1 if missing_sources else 0


if __name__ == "__main__":
    raise SystemExit(main())
