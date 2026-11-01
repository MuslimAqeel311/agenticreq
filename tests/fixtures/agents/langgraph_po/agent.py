"""Procurement PO agent — LangGraph + langchain-openai (with_raw_response path)."""

from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import Field

MODEL = "gpt-4o"


@tool
def get_inventory(sku: str, warehouse: str = "KHI", max_results: int = 50) -> str:
    """Look up stock for a SKU.

    Args:
        sku: Stock keeping unit, e.g. A-100.
        warehouse: Warehouse code.
        max_results: Maximum rows to return.
    """
    return f"{sku}@{warehouse}: 3 units"


def search_policy(query: str) -> str:
    """Search the procurement policy (called directly from Python, not by the model)."""
    return "Orders above 10,000 need manager approval."


@tool
def submit_order(sku: str, quantity: Annotated[int, Field(description="Units to order", le=500)]) -> str:
    """Submit a purchase order."""
    return f"PO-1 for {quantity}x{sku}"


@tool
def pay_invoice(invoice_id: str, amount: float) -> str:
    """Pay a supplier invoice."""
    return f"paid {invoice_id}"


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Refund a customer order."""
    return f"refunded {order_id}"


TOOLS = [get_inventory, submit_order, pay_invoice]
llm = ChatOpenAI(model=MODEL, temperature=0).bind_tools(TOOLS)


def agent(state: MessagesState) -> dict:
    policy = search_policy("approval thresholds")
    sys = SystemMessage(content=f"You are the procurement PO agent. Policy: {policy}")
    return {"messages": [llm.invoke([sys, *state["messages"]])]}


def route_refund(state: MessagesState) -> Literal["refunds", "__end__"]:
    return "refunds" if "REFUND" in str(state["messages"][-1].content) else END


g = StateGraph(MessagesState)
g.add_node("agent", agent)
g.add_node("tools", ToolNode(TOOLS))
g.add_node("refunds", ToolNode([issue_refund]))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", tools_condition)
g.add_edge("tools", "agent")
g.add_edge("refunds", END)
app = g.compile()

if __name__ == "__main__":
    out = app.invoke({"messages": [HumanMessage(content="Restock A-100")]}, config={"recursion_limit": 20})
    print(out["messages"][-1].content)
