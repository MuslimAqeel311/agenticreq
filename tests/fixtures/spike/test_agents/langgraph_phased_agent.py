"""LangGraph test agent #2: hand-written StateGraph on OpenAI with *dynamic* tool binding.

Designed to probe the limits of boundary capture:
  * prefetch node calls search_policy directly in Python (no LLM decision)
  * phase 1 binds only [get_inventory]
  * phase 2 binds [submit_order, pay_invoice]           -> tool set changes between turns
  * escalation branch binds [issue_refund]              -> never reached in this run
Ordinary user code -- knows nothing about the spike harness.
"""
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from shared_tools import (SYSTEM_RULES, USER_TASK, get_inventory, issue_refund, pay_invoice, search_policy,
                          submit_order)

inv_t = tool(parse_docstring=True)(get_inventory)
order_t = tool(parse_docstring=True)(submit_order)
pay_t = tool(parse_docstring=True)(pay_invoice)
refund_t = tool(parse_docstring=True)(issue_refund)

llm = ChatOpenAI(model="gpt-4o", temperature=0)


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    phase: str


def prefetch(state: State) -> dict:
    policy = search_policy("refund")  # direct Python call -- the LLM never sees this tool
    return {"messages": [SystemMessage(SYSTEM_RULES), HumanMessage(f"{USER_TASK}\n\n[policy context] {policy}")],
            "phase": "check"}


def phase1(state: State) -> dict:
    return {"messages": [llm.bind_tools([inv_t]).invoke(state["messages"])]}


def route1(state: State) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools1"
    return "escalate" if "ESCALATE" in str(last.content) else "phase2"


def phase2(state: State) -> dict:
    return {"messages": [llm.bind_tools([order_t, pay_t]).invoke(state["messages"])], "phase": "act"}


def escalate(state: State) -> dict:
    return {"messages": [llm.bind_tools([refund_t]).invoke(state["messages"])], "phase": "escalated"}


def build():
    g = StateGraph(State)
    g.add_node("prefetch", prefetch)
    g.add_node("phase1", phase1)
    g.add_node("tools1", ToolNode([inv_t]))
    g.add_node("phase2", phase2)
    g.add_node("escalate", escalate)
    g.add_edge(START, "prefetch")
    g.add_edge("prefetch", "phase1")
    g.add_conditional_edges("phase1", route1, ["tools1", "phase2", "escalate"])
    g.add_edge("tools1", "phase1")
    g.add_edge("phase2", END)
    g.add_edge("escalate", END)
    return g.compile()


def main() -> str:
    result = build().invoke({"messages": [], "phase": "start"})
    answer = result["messages"][-1].content
    print("AGENT OUTPUT:", answer)
    return answer


if __name__ == "__main__":
    main()
