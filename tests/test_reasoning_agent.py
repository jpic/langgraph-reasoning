"""Pytest suite for ReasoningAgent using DeepSeek (cheap)."""

import os
import pytest
from langchain_reasoning import ReasoningAgent
from langchain_deepseek import ChatDeepSeek

# All LLM tests are skipped when DEEPSEEK_API_KEY is not defined.
# This allows CI to run cleanly without requiring secrets.


@pytest.fixture
def agent():
    """Default agent using DeepSeek for tests (cheap)."""
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY not set — skipping LLM tests")

    model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    return ReasoningAgent(
        model=model,
        planning=True,
        reflection_depth=1,  # keep tests fast
        max_steps=10,
    )


# Marker used by all LLM-dependent tests
skip_if_no_key = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set — skipping LLM tests"
)


@skip_if_no_key
def test_basic_run(agent):
    """Test the happy path with a simple query."""
    response = agent.run("What is 2 + 2? Explain briefly.")
    assert isinstance(response, str)
    assert len(response) > 5
    assert "4" in response.lower() or "four" in response.lower()


@skip_if_no_key
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
@skip_if_no_key
async def test_async_run():
    """Test async interface."""
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    agent = ReasoningAgent(model=model, reflection_depth=1)
    response = await agent.arun("What is the capital of France?")
    assert isinstance(response, str)
    assert "paris" in response.lower()


@skip_if_no_key
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


@skip_if_no_key
def test_no_planning_mode():
    """Test that planning=False raises a clear error (due to LangGraph version)."""
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    with pytest.raises(NotImplementedError):
        ReasoningAgent(
            model=model,
            planning=False,
            reflection_depth=0,
        )


@skip_if_no_key
def test_streaming():
    """Test that streaming yields content."""
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
    agent = ReasoningAgent(model=model, reflection_depth=1)
    chunks = list(agent.stream("Count from 1 to 3."))

    assert len(chunks) > 0
    combined = " ".join(chunks).lower()
    assert any(n in combined for n in ["1", "2", "3"])
