# langchain-reasoning

High-level reasoning agents with **planning + reflection** on top of LangGraph.

Inspired by Agno's excellent defaults while staying fully compatible with the LangChain/LangGraph ecosystem.

## Features

- Minimal boilerplate (just like Agno)
- Strong defaults for planning and self-reflection
- Full overridability when needed
- Persistent memory via checkpointing
- Built on LangGraph (stateful, debuggable, production-ready)
- First-class pytest support with DeepSeek (cheap)

## Installation

```bash
pip install -e .
# or with test dependencies
pip install -e ".[test]"
```

Requires a `DEEPSEEK_API_KEY` environment variable when using DeepSeek (recommended for tests — very cheap).

## Usage

```python
from langchain_reasoning import ReasoningAgent
from langchain_deepseek import ChatDeepSeek   # or ChatOpenAI, ChatAnthropic, etc.

model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
agent = ReasoningAgent(
    model=model,                    # concrete model instance (no strings)
    tools=[your_tools],
    planning=True,                  # default
    reflection_depth=2,             # how many critique rounds
)

response = agent.run("Implement a robust file parser with error handling and tests.")
print(response)
```

### Streaming

```python
for chunk in agent.stream("Explain how LangGraph works"):
    print(chunk, end="")
```

### Async

```python
response = await agent.arun("Complex multi-step research task...")
```

### Overrides

```python
agent = ReasoningAgent(
    model=cheap_model,
    planner=smart_planner,      # different model for planning
    reflector=critic_model,     # dedicated reflection model
    reflection_depth=3,
)
```

## Testing

All tests use **DeepSeek** (very cheap):

```bash
python -m pytest tests/ -q
```

## Design Goals

- **Sane defaults** — you should rarely need to override anything (just like your Agno experience)
- **Overridability** — every major piece (`planner`, `reflector`, `checkpointer`, `model`) can be swapped
- **No boilerplate** — no manual `StateGraph`, `add_node`, or custom state classes for normal use

Enjoy the best of both worlds: Agno-like ergonomics with LangGraph's power under the hood.

---
Built with ❤️ for the `langchain_reasoning` exploration.
