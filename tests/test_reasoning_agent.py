"""Pytest suite for ReasoningAgent.

Uses a small local CPU model (Qwen2.5-0.5B) by default.
Set DEEPSEEK_API_KEY environment variable to test with DeepSeek instead.
"""

import os
import pytest
from langchain_reasoning import ReasoningAgent
from langchain_deepseek import ChatDeepSeek


# The `test_model` and `agent` fixtures are defined in conftest.py.
# They automatically prefer DeepSeek when DEEPSEEK_API_KEY is set,
# otherwise they fall back to a small local CPU model (Qwen2.5-0.5B).


def test_basic_run(agent):
    """Test the happy path with a simple query."""
    response = agent.run("What is 2 + 2? Explain briefly.")
    assert isinstance(response, str)
    assert len(response) > 5
    assert "4" in response.lower() or "four" in response.lower()


def test_planning_enabled(agent):
    """Verify that planning node generates options, judges confidence, and selects best (Agno-style)."""
    response = agent.run("Plan a simple Python function to calculate fibonacci.")
    
    assert isinstance(response, str)
    assert len(response) > 30
    lower = response.lower()
    # Should show confidence judgement and multiple options
    assert any(kw in lower for kw in ["confidence", "conf:", "option", "approach", "best"])
    # Should still produce working code
    assert any(kw in lower for kw in ["def fibonacci", "fibonacci", "iterative", "return b"])


@pytest.mark.asyncio
async def test_async_run():
    """Test async interface."""
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    agent = ReasoningAgent(model=model, reflection_depth=1)
    response = await agent.arun("What is the capital of France?")
    assert isinstance(response, str)
    assert "paris" in response.lower()


def test_override_components():
    """Test that we can override planner and reflector."""
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    fast_model = ChatDeepSeek(model="deepseek-chat", temperature=0.0)

    agent = ReasoningAgent(
        model=model,
        planner=fast_model,
        reflector=fast_model,
        reflection_depth=1,
        planning=True,
    )

    response = agent.run("Write a one-line hello world in Python.")
    assert isinstance(response, str)
    assert "hello" in response.lower() or "print" in response.lower()


def test_no_planning_mode():
    """Test that planning=False raises a clear error (due to LangGraph version)."""
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    with pytest.raises(NotImplementedError):
        ReasoningAgent(
            model=model,
            planning=False,
            reflection_depth=0,
        )


def test_streaming():
    """Test that streaming yields content."""
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    agent = ReasoningAgent(model=model, reflection_depth=1)
    chunks = list(agent.stream("Count from 1 to 3."))

    assert len(chunks) > 0
    combined = " ".join(chunks).lower()
    assert any(n in combined for n in ["1", "2", "3"])
