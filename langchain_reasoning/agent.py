"""Core ReasoningAgent with sane defaults and planning/reflection."""

from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from .graph import create_reasoning_graph, ReasoningState


class ReasoningAgent:
    """
    High-level agent with built-in planning and reflection.
    Minimal boilerplate, excellent defaults, full overridability.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        planning: bool = True,
        reflection_depth: int = 2,
        max_steps: int = 25,
        planner: Optional[BaseChatModel] = None,
        reflector: Optional[BaseChatModel] = None,
        checkpointer: Optional[Any] = None,
        **kwargs: Any,
    ):
        """ReasoningAgent with sane defaults.

        Pass a concrete LangChain chat model (no string magic).
        DeepSeek example for tests:

            from langchain_deepseek import ChatDeepSeek
            model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)
            agent = ReasoningAgent(model=model)
        """
        self.tools = tools or []
        self.system_prompt = system_prompt or (
            "You are a helpful, accurate, and thorough AI assistant. "
            "Use planning and self-reflection to deliver high-quality results."
        )
        self.planning = planning
        self.reflection_depth = reflection_depth
        self.max_steps = max_steps
        self.config = kwargs

        self.model = model

        # Component overrides (use same model by default)
        self.planner = planner or self.model
        self.reflector = reflector or self.model
        self.checkpointer = checkpointer or MemorySaver()

        # Build the graph once
        self.graph = create_reasoning_graph(
            model=self.model,
            tools=self.tools,
            system_prompt=self.system_prompt,
            planning=self.planning,
            reflection_depth=self.reflection_depth,
            max_steps=self.max_steps,
            planner=self.planner,
            reflector=self.reflector,
            checkpointer=self.checkpointer,
        )

        self.app = self.graph.compile(checkpointer=self.checkpointer)



    def run(self, input: str, config: Optional[Dict] = None) -> str:
        """Synchronous run with nice defaults."""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}

        inputs = {"messages": [HumanMessage(content=input)]}
        result = self.app.invoke(inputs, config=config)
        return result["messages"][-1].content

    async def arun(self, input: str, config: Optional[Dict] = None) -> str:
        """Async run."""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}

        inputs = {"messages": [HumanMessage(content=input)]}
        result = await self.app.ainvoke(inputs, config=config)
        return result["messages"][-1].content

    def stream(self, input: str, config: Optional[Dict] = None):
        """Stream output (useful for long-running tasks)."""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}

        inputs = {"messages": [HumanMessage(content=input)]}
        for chunk in self.app.stream(inputs, config=config, stream_mode="values"):
            if "messages" in chunk and chunk["messages"]:
                last_msg = chunk["messages"][-1]
                if isinstance(last_msg, AIMessage) and last_msg.content:
                    yield last_msg.content


# Convenience factory
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
    """Exposed for advanced users who want the raw graph."""
    from .graph import create_reasoning_graph as _create
    return _create(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        planning=planning,
        reflection_depth=reflection_depth,
        max_steps=max_steps,
        planner=planner,
        reflector=reflector,
        checkpointer=checkpointer,
    )
