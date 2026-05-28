"""LangGraph implementation with planner → actor → reflector loop."""

from typing import Any, Dict, List, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
import operator


class ReasoningState(TypedDict):
    """State for the reasoning graph with confidence-based planning."""
    messages: Annotated[List[BaseMessage], add_messages]
    plan: Optional[str]
    options: Optional[List[Dict]]  # list of {"option": str, "confidence": float, "reasoning": str}
    reflections: Annotated[List[str], operator.add]
    step_count: int
    next: str


def planner_node(state: ReasoningState, planner: BaseChatModel, system_prompt: str) -> Dict:
    """Generate multiple options, judge confidence, and select the best one (Agno-style)."""
    messages = state["messages"]
    task = messages[-1].content

    prompt = f"""{system_prompt}

You are an expert strategic planner. For the given task, generate **3 distinct approaches**.

For each approach:
- Give it a short name
- Describe the concrete steps
- Assign a confidence score (0.0-1.0) based on feasibility, efficiency, robustness, and simplicity
- Briefly justify the score

Then choose the single best one.

Task: {task}

First think step-by-step, then respond in this exact JSON format:

{{
  "options": [
    {{
      "option": "Iterative approach",
      "steps": "1. Handle base cases... 2. Use two variables in a loop...",
      "confidence": 0.9,
      "reasoning": "Simple, efficient O(n) time, O(1) space, easy to understand"
    }}
  ],
  "best": 0,
  "overall_reasoning": "The iterative solution is preferred because it balances performance and readability best."
}}"""

    response = planner.invoke([
        SystemMessage(content="You are an expert planner that compares options rigorously and outputs valid JSON only."),
        HumanMessage(content=prompt)
    ])

    try:
        import json
        content = response.content.strip()
        # DeepSeek sometimes adds markdown; extract JSON if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        parsed = json.loads(content)
        options = parsed.get("options", [])
        best_idx = parsed.get("best", 0)
        best_option = options[best_idx] if options and isinstance(options, list) else {}
        plan = best_option.get("steps", parsed.get("overall_reasoning", response.content))
        overall = parsed.get("overall_reasoning", "Selected best approach based on confidence judgement.")
    except Exception:
        # Fallback for non-JSON responses
        plan = response.content
        options = [{"option": "Default Plan", "confidence": 0.7, "reasoning": "Model did not return structured JSON"}]
        overall = "Using model output directly (JSON parsing failed)"

    display = f"""**Planning with Confidence Judgement**

{overall}

**Selected Approach (confidence: {best_option.get('confidence', 0.75):.2f})**
{best_option.get('option', 'Best Plan')}:
{plan}

**Options Evaluated:**
""" + "\n".join([
        f"- {opt.get('option', 'Option')} (confidence: {opt.get('confidence', 0):.2f}) — {opt.get('reasoning', '')[:80]}..."
        for opt in options
    ])

    return {
        "plan": plan,
        "options": options,
        "messages": [AIMessage(content=display)],
        "reflections": [],
        "step_count": state.get("step_count", 0) + 1,
        "next": "actor",
    }


def actor_node(state: ReasoningState, model: BaseChatModel, tools: List[Any], system_prompt: str) -> Dict:
    """Execute using ReAct style with tools."""
    if not tools:
        # No tools: just call the model directly
        response = model.invoke(state["messages"])
        return {
            "messages": [response],
            "step_count": state.get("step_count", 0) + 1,
            "next": "reflector",
        }

    # Use prebuilt ReAct for the actor when tools are present
    react_agent = create_react_agent(
        model=model,
        tools=tools,
        state_modifier=system_prompt,
    )

    result = react_agent.invoke({
        "messages": state["messages"][-3:] if len(state["messages"]) > 3 else state["messages"],  # keep context short
    })

    return {
        "messages": result.get("messages", state["messages"]),
        "step_count": state.get("step_count", 0) + 1,
        "next": "reflector",
    }


def reflector_node(state: ReasoningState, reflector: BaseChatModel, reflection_depth: int) -> Dict:
    """Self-critique and reflect."""
    messages = state["messages"]
    plan = state.get("plan", "No explicit plan.")

    prompt = f"""Review your work so far.

Plan: {plan}

Recent output: {messages[-1].content if messages else 'None'}

Provide critical feedback. What is good? What could be improved? 
What should be done differently in the next iteration?

Reflection:"""

    response = reflector.invoke([HumanMessage(content=prompt)])
    reflection = response.content

    reflections = state.get("reflections", []) + [reflection]

    # Decide whether to continue or conclude
    if len(reflections) >= reflection_depth or state.get("step_count", 0) > 15:
        next_step = "final_answer"
    else:
        next_step = "actor"  # loop back with reflection in context

    return {
        "reflections": [reflection],
        "messages": [AIMessage(content=f"**Reflection:** {reflection}")],
        "next": next_step,
    }


def final_answer_node(state: ReasoningState) -> Dict:
    """Synthesize final response."""
    messages = state["messages"]
    reflections = state.get("reflections", [])

    if reflections:
        synthesis = f"After careful planning and {len(reflections)} rounds of reflection, here is the final answer:\n\n"
        synthesis += messages[-1].content
    else:
        synthesis = messages[-1].content

    return {
        "messages": [AIMessage(content=synthesis)],
        "next": END,
    }


def route_next(state: ReasoningState) -> str:
    """Router for conditional edges."""
    return state.get("next", "actor")


def create_reasoning_graph(
    model: BaseChatModel,
    tools: List[Any],
    system_prompt: str,
    planning: bool = True,
    reflection_depth: int = 2,
    max_steps: int = 25,
    planner: Optional[BaseChatModel] = None,
    reflector: Optional[BaseChatModel] = None,
    checkpointer: Optional[Any] = None,
) -> StateGraph:
    """Build the full reasoning graph with sane defaults."""
    # create_react_agent already imported at top

    graph = StateGraph(ReasoningState)

    # Add nodes
    if planning:
        graph.add_node("planner", lambda s: planner_node(s, planner or model, system_prompt))
        graph.add_node("actor", lambda s: actor_node(s, model, tools, system_prompt))
        graph.add_node("reflector", lambda s: reflector_node(s, reflector or model, reflection_depth))
        graph.add_node("final_answer", final_answer_node)

        # Edges with planning flow
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "actor")
        graph.add_conditional_edges("reflector", route_next, {
            "actor": "actor",
            "final_answer": "final_answer",
        })
        graph.add_edge("final_answer", END)
    else:
        # Simple ReAct fallback when planning is disabled.
        # The current version of create_react_agent in this env expects a different
        # signature. For now we disable the simple path and always use our planning loop.
        # (The planning=True path already works well and is the main value.)
        raise NotImplementedError("planning=False path is temporarily disabled due to LangGraph version incompatibility. Use planning=True.")

    return graph
