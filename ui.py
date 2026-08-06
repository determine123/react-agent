"""Gradio web UI — chat interface for the Agent and RAG Q&A.

Usage:
    python main.py --ui                    # start at 0.0.0.0:7860
    python main.py --ui --share            # create a public tunnel
"""

from __future__ import annotations

import time as time_module

import gradio as gr

# Global state (lazy init)
_config = None
_chat_state = None
_qa = None


def _get_config():
    global _config
    if _config is None:
        from config import load_config

        _config = load_config()
    return _config


def _get_qa():
    global _qa
    if _qa is None:
        from rag.qa import RAGQA

        _qa = RAGQA(_get_config())
    return _qa


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------


def chat_fn(message: str, history: list[list[str]], system_prompt: str):
    """Process a chat message through the ReAct agent."""
    from core.agent_loop import AgentLoop
    from core.state import AgentState

    cfg = _get_config()

    if system_prompt.strip():
        cfg.system_prompt = system_prompt.strip()

    state = AgentState(system_prompt=cfg.system_prompt)

    # Feed history as context for continuity
    for h in history[-10:]:  # last 10 turns
        state.add_message(
            __import__("core.state", fromlist=["Message"]).Message(
                role="user", content=h[0]
            )
        )
        if h[1]:
            state.add_message(
                __import__("core.state", fromlist=["Message"]).Message(
                    role="assistant", content=h[1]
                )
            )

    t0 = time_module.perf_counter()
    agent = AgentLoop(config=cfg, state=state)
    answer = agent.run(message)
    elapsed = (time_module.perf_counter() - t0) * 1000

    # Collect tool call info for display
    tools_used = []
    for msg in state.messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_used.append(tc["function"]["name"])

    status = f"⏱ {elapsed:.0f}ms | Steps: {state.step_count}"
    if tools_used:
        status += f" | Tools: {', '.join(tools_used)}"

    return answer, status


# ---------------------------------------------------------------------------
# RAG tab
# ---------------------------------------------------------------------------


def rag_fn(question: str, top_k: int):
    """Answer using RAG over indexed documents."""
    qa = _get_qa()

    if qa.index_count() == 0:
        return (
            "No documents indexed. Use the 'Index' tab to load documents first.",
            "",
        )

    result = qa.answer(question, top_k=top_k, return_sources=True)

    answer = result["answer"]
    sources_text = ""
    for s in result.get("sources", []):
        sources_text += (
            f"**[{s['index']}] {s['source']}** (distance={s.get('distance', '?')})\n"
            f"> {s['text_preview'][:200]}...\n\n"
        )

    return answer, sources_text


# ---------------------------------------------------------------------------
# Index tab
# ---------------------------------------------------------------------------


def index_fn(directory: str):
    """Index documents from a directory."""
    from pathlib import Path

    if not directory or not Path(directory).exists():
        return f"❌ Directory not found: {directory}"

    qa = _get_qa()
    count = qa.index_documents(directory)
    return f"✅ Indexed **{count}** chunks from `{directory}`.\nTotal in store: **{qa.index_count()}** chunks."


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------


def create_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""

    with gr.Blocks(
        title="Agent Project",
        theme=gr.themes.Soft(),
        css="""
        .status-bar { font-size: 0.85em; color: #666; margin-top: 4px; }
        footer { display: none !important; }
        """,
    ) as demo:
        gr.Markdown(
            """# 🤖 Agent Project

**ReAct Agent + RAG Q&A + Tool Calling**
— Chat with an AI that can search the web, calculate, check weather, and more.
"""
        )

        with gr.Tabs():
            # ==================================================
            # Tab 1: Chat
            # ==================================================
            with gr.TabItem("💬 Chat"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=450,
                            bubble_full_width=False,
                        )
                        status_bar = gr.Markdown("", elem_classes=["status-bar"])
                    with gr.Column(scale=1):
                        system_prompt_box = gr.Textbox(
                            label="System Prompt",
                            value="You are a helpful, thoughtful AI assistant.",
                            lines=4,
                            placeholder="Custom system prompt...",
                        )
                        gr.Markdown(
                            """**Available Tools:**
- `calculate` — math
- `web_search` — search
- `get_weather` — weather
- `get_current_time` — time
- `fetch_page` — read URLs
- `write_note` / `read_note` — notes
"""
                        )

                msg_box = gr.Textbox(
                    label="Message",
                    placeholder="Type your message and press Enter...",
                )
                clear_btn = gr.Button("Clear Chat", variant="secondary", size="sm")

                msg_box.submit(
                    fn=chat_fn,
                    inputs=[msg_box, chatbot, system_prompt_box],
                    outputs=[chatbot, status_bar],
                ).then(lambda: "", None, msg_box)

                clear_btn.click(
                    lambda: (None, ""), None, [chatbot, status_bar]
                )

            # ==================================================
            # Tab 2: RAG Q&A
            # ==================================================
            with gr.TabItem("📚 RAG Q&A"):
                qa = _get_qa()
                store_count = qa.index_count()

                gr.Markdown(
                    f"**Documents indexed: {store_count} chunks**\n"
                    f"(Use the 'Index' tab to add documents.)"
                )

                rag_question = gr.Textbox(
                    label="Question",
                    placeholder="Ask a question based on your documents...",
                )
                top_k_slider = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1, label="Chunks to retrieve"
                )
                rag_btn = gr.Button("Ask", variant="primary")
                rag_answer = gr.Textbox(label="Answer", lines=8)
                rag_sources = gr.Markdown("")

                rag_btn.click(
                    fn=rag_fn,
                    inputs=[rag_question, top_k_slider],
                    outputs=[rag_answer, rag_sources],
                )

            # ==================================================
            # Tab 3: Index
            # ==================================================
            with gr.TabItem("📥 Index Documents"):
                gr.Markdown(
                    "Point to a directory with PDF, Markdown, or TXT files."
                )
                index_dir = gr.Textbox(
                    label="Directory Path",
                    placeholder="./docs",
                    value="./docs",
                )
                index_btn = gr.Button("Index", variant="primary")
                index_status = gr.Markdown("")

                index_btn.click(
                    fn=index_fn,
                    inputs=[index_dir],
                    outputs=[index_status],
                )

    return demo
