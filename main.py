"""Agent Project — Main Entry Point.

Orchestrates the full agent loop: perception → reasoning → action → memory.

Modes:
    python main.py                          Interactive chat (ReAct + memory)
    python main.py --mode plan_execute      Plan-and-Execute strategy
    python main.py --mode self_critique     Self-Critique strategy
    python main.py --mode multi_agent       Multi-Agent (Planner+Executor)
    python main.py --stateless              Stateless chat (no memory)
    python main.py --rag                    RAG Q&A (default strategy)
    python main.py --rag --strategy hyde    RAG with HyDE
    python main.py --rag --strategy self_rag RAG with Self-RAG
    python main.py --index DIR              Index documents
    python main.py --eval                   Quick evaluation
    python main.py --eval --ragas --html    RAGAS eval + HTML report
    python main.py --api                    FastAPI server
    python main.py --ui                     Gradio web UI
"""

from __future__ import annotations

import argparse


def main() -> None:
    args = _parse_args()

    # --- FastAPI server ---
    if args.api:
        _run_api(args)
        return

    # --- Gradio UI ---
    if args.ui:
        _run_ui(args)
        return

    # --- Evaluation ---
    if args.eval:
        _run_eval(args)
        return

    # --- Document indexing ---
    from config import load_config

    config = load_config()

    if args.index:
        _run_index(config, args.index)
        return

    # --- RAG Q&A ---
    if args.rag:
        _run_rag_qa(config, strategy=args.strategy)
        return

    # --- Interactive chat (with optional advanced mode) ---
    _run_chat(config, stateless=args.stateless, mode=args.mode)


# ======================================================================
# Mode: Interactive Chat
# ======================================================================

def _run_chat(config, *, stateless: bool = False, mode: str = "react") -> None:
    """Interactive agent chat loop with memory, advanced strategies & tool router."""
    from core.agent_loop import AgentLoop
    from core.state import AgentState
    from memory.memory import Memory
    from core.tool_router import ToolRouter

    state = AgentState(system_prompt=config.system_prompt)
    base_agent = AgentLoop(config=config, state=state)
    memory = Memory(config.memory)
    router = ToolRouter()
    router_enabled = True

    # Advanced agents (lazy init)
    advanced_agent = None
    multi_agent_orch = None
    if mode in ("plan_execute", "self_critique"):
        from core.advanced_agent import AdvancedAgent
        advanced_agent = AdvancedAgent(config)
    elif mode == "multi_agent":
        from graph.multi_agent import MultiAgentOrchestrator
        multi_agent_orch = MultiAgentOrchestrator(config)

    mode_labels = {
        "react": "ReAct",
        "plan_execute": "Plan-and-Execute",
        "self_critique": "Self-Critique",
        "multi_agent": "Multi-Agent (Planner+Executor)",
    }
    mem_label = "stateless" if stateless else "with memory"
    print(f"Agent ready — {mode_labels.get(mode, mode)} mode ({mem_label}), router ON.")
    print("Commands: 'quit' | '/clear' | '/router' toggle | '/mode <name>' switch\n")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not user_input:
            continue
        if user_input == "/clear":
            state.messages.clear()
            state.finished = False
            state.step_count = 0
            print("[Context cleared]\n")
            continue
        if user_input.startswith("/mode"):
            new_mode = user_input.split(maxsplit=1)[-1].strip()
            if new_mode in mode_labels:
                mode = new_mode
                if mode in ("plan_execute", "self_critique") and advanced_agent is None:
                    from core.advanced_agent import AdvancedAgent
                    advanced_agent = AdvancedAgent(config)
                elif mode == "multi_agent" and multi_agent_orch is None:
                    from graph.multi_agent import MultiAgentOrchestrator
                    multi_agent_orch = MultiAgentOrchestrator(config)
                print(f"[Mode switched to: {mode_labels[mode]}]\n")
            else:
                print(f"[Available modes: {', '.join(mode_labels.keys())}]\n")
            continue
        if user_input == "/router":
            router_enabled = not router_enabled
            status = "ON" if router_enabled else "OFF"
            print(f"[Tool router: {status}]\n")
            continue

        enriched_input = _enrich_with_memory(user_input, memory, config)

        if stateless:
            state.messages.clear()

        state.finished = False
        state.step_count = 0

        # --- Tool Router fast path ---
        routing = None
        if router_enabled and mode == "react":
            routing = router.classify(user_input)
            if routing.confidence >= 0.85 and routing.tool_name:
                answer = _execute_routed_tool(routing, user_input)
                if answer:
                    print(f"[Router: {routing.intent.value} → {routing.tool_name}]")
                    ts = __import__("time").strftime("%Y-%m-%d %H:%M:%S")
                    memory.remember({"role": "user", "content": user_input, "timestamp": ts})
                    memory.remember({"role": "assistant", "content": answer, "timestamp": ts})
                    print(f"\nAgent: {answer}\n")
                    continue

        # --- Full agent path ---
        if mode == "multi_agent" and multi_agent_orch:
            result = multi_agent_orch.run(enriched_input)
            answer = result.get("final_answer", "")
            subtasks = result.get("subtasks", [])
            if subtasks:
                print(f"[Executed {len(subtasks)} subtasks]")
        elif mode in ("plan_execute", "self_critique") and advanced_agent:
            answer = advanced_agent.run(enriched_input, mode=mode)
        else:
            answer = base_agent.run(enriched_input)

        ts = __import__("time").strftime("%Y-%m-%d %H:%M:%S")
        memory.remember({"role": "user", "content": user_input, "timestamp": ts})
        memory.remember({"role": "assistant", "content": answer, "timestamp": ts})

        print(f"\nAgent: {answer}\n")


def _execute_routed_tool(routing, user_input: str) -> str | None:
    """Execute a tool directly (skipping LLM) and format the result.

    Returns a human-readable answer string, or None if routing should
    fall through to the full ReAct loop.
    """
    from tools.registry import ToolRegistry
    import json

    tools = ToolRegistry()

    try:
        result = tools.dispatch(routing.tool_name, **routing.tool_args)
    except Exception:
        return None  # tool failed → fall through to ReAct

    if isinstance(result, dict):
        result_str = json.dumps(result, ensure_ascii=False, indent=2)
    else:
        result_str = str(result)

    # Format response based on intent
    if routing.intent == "math":
        return f"计算结果：{result}"
    elif routing.intent == "weather":
        return f"天气查询结果：\n{result_str}"
    elif routing.intent == "time":
        dt = result.get("datetime", result_str) if isinstance(result, dict) else result_str
        return f"当前时间：{dt}"
    elif routing.intent == "search":
        return f"搜索结果：\n{result_str}"
    elif routing.intent == "note_read":
        content = result.get("content", result_str) if isinstance(result, dict) else result_str
        return f"📝 {content}"
    elif routing.intent == "note_write":
        return f"已保存 ✓"
    else:
        return result_str


def _enrich_with_memory(user_input: str, memory, config) -> str:
    all_entries = memory.load_all()
    if not all_entries:
        return user_input
    recent = all_entries[-(config.memory.working_memory_limit):]
    if not recent:
        return user_input
    context_lines = ["[Previous conversation context from memory:]"]
    for entry in recent:
        context_lines.append(
            f"  [{entry['role']} @ {entry.get('timestamp', '?')}]: {entry['content'][:200]}"
        )
    return "\n".join(context_lines) + f"\n\n[Current user message]\n{user_input}"


# ======================================================================
# Mode: RAG Q&A
# ======================================================================

def _run_rag_qa(config, *, strategy: str = "default") -> None:
    """Interactive RAG Q&A loop with optional HyDE / Self-RAG."""
    from rag.qa import RAGQA

    qa = RAGQA(config)
    doc_count = qa.index_count()

    if doc_count == 0:
        print(
            "No documents indexed yet. Use --index <dir> first.\n"
            "Example:  python main.py --index ./docs"
        )
        return

    strat_labels = {"default": "Standard", "hyde": "HyDE", "self_rag": "Self-RAG"}
    print(
        f"RAG QA ready ({doc_count} chunks, {strat_labels.get(strategy, strategy)} strategy). "
        "Type 'quit' to exit.\n"
    )
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not question:
            continue

        result = qa.answer(question, return_sources=True, strategy=strategy)
        print(f"\nAnswer: {result['answer']}\n")
        if config.verbose and result.get("sources"):
            print("Sources:")
            for s in result["sources"]:
                print(f"  [{s['index']}] {s['source']}  (distance={s.get('distance', '?')})")
            if result.get("strategy"):
                print(f"Strategy: {result['strategy']}")
            print()


# ======================================================================
# Mode: Document Indexing
# ======================================================================

def _run_index(config, dir_path: str) -> None:
    from rag.qa import RAGQA

    print(f"Indexing documents from: {dir_path}")
    qa = RAGQA(config)
    count = qa.index_documents(dir_path)
    print(f"Done — {count} chunks indexed.")


# ======================================================================
# Mode: Evaluation
# ======================================================================

def _run_eval(args) -> None:
    """Run evaluation and optionally generate HTML report."""
    if args.ragas:
        from eval.evaluator import run_ragas_eval
        print("Running RAGAS evaluation...\n")
        report = run_ragas_eval()
    else:
        from eval.evaluator import run_eval
        print("Running quick evaluation...\n")
        report = run_eval()

    _print_eval_report(report)

    # Generate HTML report
    if args.html:
        from eval.report import generate_html_report
        path = generate_html_report(report)
        print(f"\n📄 HTML report saved to: {path}")


def _print_eval_report(report) -> None:
    print(f"\n{'=' * 60}")
    print(f"Eval complete: {report.passed}/{report.total} passed ({report.accuracy:.1%})")
    print(f"{'=' * 60}")
    for r in report.results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} [{r.question_id}] {r.question}")
        print(f"       → {r.answer[:120]}{'...' if len(r.answer) > 120 else ''}")
        print(f"       ⏱ {r.duration_s}s  |  {r.detail}")
        if r.ragas_scores:
            scores_str = "  ".join(f"{k}={v:.3f}" for k, v in r.ragas_scores.items())
            print(f"       📊 RAGAS: {scores_str}")


# ======================================================================
# Mode: FastAPI Server
# ======================================================================

def _run_api(args) -> None:
    print("Starting FastAPI server at http://0.0.0.0:8000 ...")
    print("Endpoints: POST /chat  POST /rag  GET /tools  GET /health")
    import uvicorn
    from api import create_app

    app = create_app()
    uvicorn.run(app, host=args.host, port=args.port)


# ======================================================================
# Mode: Gradio UI
# ======================================================================

def _run_ui(args) -> None:
    from ui import create_ui

    print("Starting Gradio UI at http://0.0.0.0:7860 ...")
    demo = create_ui()
    demo.launch(server_name=args.host, server_port=args.ui_port, share=args.share)


# ======================================================================
# CLI
# ======================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent Project — ReAct + RAG + Multi-Agent")

    # Modes
    p.add_argument("--rag", action="store_true", help="Run in RAG Q&A mode")
    p.add_argument("--index", metavar="DIR", help="Index documents from a directory")
    p.add_argument("--eval", action="store_true", help="Run evaluation suite")
    p.add_argument("--api", action="store_true", help="Start FastAPI server")
    p.add_argument("--ui", action="store_true", help="Start Gradio web UI")

    # Options
    p.add_argument("--mode", default="react",
                   choices=["react", "plan_execute", "self_critique", "multi_agent"],
                   help="Agent strategy (default: react)")
    p.add_argument("--strategy", default="default",
                   choices=["default", "hyde", "self_rag"],
                   help="RAG retrieval strategy (default: default)")
    p.add_argument("--stateless", action="store_true",
                   help="Stateless chat (no memory)")
    p.add_argument("--ragas", action="store_true",
                   help="Use RAGAS metrics (with --eval)")
    p.add_argument("--html", action="store_true",
                   help="Generate HTML report (with --eval)")

    # Network
    p.add_argument("--host", default="0.0.0.0", help="Server host")
    p.add_argument("--port", type=int, default=8000, help="API port")
    p.add_argument("--ui-port", type=int, default=7860, help="UI port")
    p.add_argument("--share", action="store_true", help="Shareable Gradio link")

    return p.parse_args()


if __name__ == "__main__":
    main()
