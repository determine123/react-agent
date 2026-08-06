"""Advanced Agent strategies: Plan-and-Execute + Self-Critique.

Extends the base ReAct loop with two advanced reasoning patterns:

**Plan-and-Execute** (plan_execute mode):
    1. PLAN:  LLM decomposes the user's request into numbered steps.
    2. EXECUTE: Each step is fed to the ReAct loop in order.
    3. SYNTHESIZE: Results are combined into a final answer.

**Self-Critique** (self_critique mode):
    1. REASON + ACT: Standard ReAct loop produces a draft answer.
    2. CRITIQUE: A second LLM call reviews the answer for errors,
       hallucinations, and completeness.
    3. REFINE: If critique finds issues, the answer is revised.

Usage::

    from core.advanced_agent import AdvancedAgent
    from config import load_config

    cfg = load_config()
    agent = AdvancedAgent(cfg)

    # Plan-and-Execute
    answer = agent.run("Research AI trends and write a summary", mode="plan_execute")

    # Self-Critique
    answer = agent.run("What is the capital of France?", mode="self_critique")
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import Config

from core.agent_loop import AgentLoop
from core.state import AgentState, Message
from llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_PROMPT = """You are a strategic planner. Decompose the user's request into
a numbered list of concrete, actionable steps. Each step should be a self-contained
sub-task that can be executed independently.

Rules:
- Write 3-7 steps.
- Each step should start with a verb (Search, Calculate, Write, Analyze, etc.)
- Be specific about what information is needed.
- Output ONLY the numbered list, no introduction.

User request: {request}

Plan:"""

_SYNTHESIZE_PROMPT = """You are a synthesis assistant. Combine the results of
individual steps into a single, coherent answer.

Original request: {request}

Step results:
{step_results}

Final answer:"""

_CRITIQUE_PROMPT = """You are a critical reviewer. Evaluate the following answer
for accuracy, completeness, and potential hallucinations.

Question: {question}
Answer: {answer}

Check for:
1. Factual errors or contradictions
2. Missing important information
3. Unsupported claims or hallucinations
4. Clarity and structure

If the answer is perfect, reply with exactly: PASS
Otherwise, list the issues found and suggest specific improvements.
Start issues with "ISSUE: " on each line.

Review:"""

_REFINE_PROMPT = """Your previous answer had some issues. Please revise it
based on the critique.

Question: {question}
Previous answer: {answer}
Critique: {critique}

Revised answer:"""


class AdvancedAgent:
    """Agent with Plan-and-Execute and Self-Critique capabilities."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.llm = LLMClient(config.llm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_input: str,
        *,
        mode: str = "react",
    ) -> str:
        """Execute the agent in the specified mode.

        Args:
            user_input: User's request.
            mode: One of ``"react"``, ``"plan_execute"``, ``"self_critique"``.

        Returns:
            Final answer string.
        """
        if mode == "plan_execute":
            return self._plan_execute(user_input)
        elif mode == "self_critique":
            return self._self_critique(user_input)
        else:
            # Default: standard ReAct
            state = AgentState(system_prompt=self.config.system_prompt)
            agent = AgentLoop(config=self.config, state=state)
            return agent.run(user_input)

    # ------------------------------------------------------------------
    # Plan-and-Execute
    # ------------------------------------------------------------------

    def _plan_execute(self, user_input: str) -> str:
        """Decompose the request into steps, execute each, then synthesize."""
        logger.info("Plan-and-Execute: planning phase")
        plan_text = self._generate_plan(user_input)
        steps = self._parse_steps(plan_text)

        if not steps:
            # Fall back to standard ReAct if planning fails
            state = AgentState(system_prompt=self.config.system_prompt)
            agent = AgentLoop(config=self.config, state=state)
            return agent.run(user_input)

        logger.info("Plan-and-Execute: executing %d steps", len(steps))
        step_results: list[str] = []
        for i, step in enumerate(steps, 1):
            logger.info("  Step %d/%d: %s", i, len(steps), step[:80])
            state = AgentState(
                system_prompt=(
                    f"You are executing ONE step of a larger plan.\n"
                    f"Context: {user_input}\n"
                    f"Your task (Step {i}/{len(steps)}): {step}\n"
                    f"Focus ONLY on this step. Be concise."
                )
            )
            agent = AgentLoop(config=self.config, state=state)
            result = agent.run(step)
            step_results.append(f"Step {i} ({step[:60]}):\n{result}")

        logger.info("Plan-and-Execute: synthesizing")
        return self._synthesize(user_input, step_results)

    def _generate_plan(self, request: str) -> str:
        msgs = [{"role": "user", "content": _PLAN_PROMPT.format(request=request)}]
        raw = self.llm.chat(messages=msgs)
        return raw["choices"][0]["message"].get("content", "")

    @staticmethod
    def _parse_steps(plan_text: str) -> list[str]:
        steps: list[str] = []
        for line in plan_text.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("- ")):
                # Remove numbering like "1. " or "- "
                content = line.split(". ", 1)[-1] if ". " in line else line
                content = content.removeprefix("- ").strip()
                if content:
                    steps.append(content)
        return steps

    def _synthesize(self, request: str, step_results: list[str]) -> str:
        msgs = [
            {
                "role": "user",
                "content": _SYNTHESIZE_PROMPT.format(
                    request=request,
                    step_results="\n\n".join(step_results),
                ),
            }
        ]
        raw = self.llm.chat(messages=msgs)
        return raw["choices"][0]["message"].get("content", "")

    # ------------------------------------------------------------------
    # Self-Critique
    # ------------------------------------------------------------------

    def _self_critique(self, user_input: str) -> str:
        """Generate an answer, critique it, and refine if needed."""
        # Step 1: Generate draft answer via ReAct
        logger.info("Self-Critique: generating draft")
        state = AgentState(system_prompt=self.config.system_prompt)
        agent = AgentLoop(config=self.config, state=state)
        draft = agent.run(user_input)

        # Step 2: Critique
        logger.info("Self-Critique: reviewing")
        critique = self._critique(user_input, draft)

        if critique.strip().upper() == "PASS":
            logger.info("Self-Critique: passed review")
            return draft

        # Step 3: Refine
        logger.info("Self-Critique: refining (issues found: %d chars)", len(critique))
        refined = self._refine(user_input, draft, critique)
        return refined

    def _critique(self, question: str, answer: str) -> str:
        msgs = [
            {
                "role": "user",
                "content": _CRITIQUE_PROMPT.format(question=question, answer=answer),
            }
        ]
        raw = self.llm.chat(messages=msgs)
        return raw["choices"][0]["message"].get("content", "PASS")

    def _refine(self, question: str, answer: str, critique: str) -> str:
        msgs = [
            {
                "role": "user",
                "content": _REFINE_PROMPT.format(
                    question=question, answer=answer, critique=critique
                ),
            }
        ]
        raw = self.llm.chat(messages=msgs)
        return raw["choices"][0]["message"].get("content", answer)
