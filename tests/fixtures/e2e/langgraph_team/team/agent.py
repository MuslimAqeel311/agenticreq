"""A two-agent LangGraph team: an orders agent, then a customers agent, in one graph.

Used by validation/e2e_steps.py, which walks it through the five steps with the installed wheel.
"""

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent


@tool
def lookup_order(order_id: str) -> str:
    """Return the status of an order."""
    return {"A1": "shipped"}.get(order_id, "not found")


@tool
def lookup_customer(customer_id: str) -> str:
    """Return the name of a customer."""
    return {"C1": "Ada"}.get(customer_id, "not found")


llm = ChatOpenAI(model="gpt-4o-2024-08-06", max_retries=2)
orders = create_react_agent(llm, [lookup_order], name="orders", prompt="You answer questions about orders.")
customers = create_react_agent(llm, [lookup_customer], name="customers",
                               prompt="You answer questions about customers.")


def build():
    g = StateGraph(MessagesState)
    g.add_node("orders", orders)
    g.add_node("customers", customers)
    g.add_edge(START, "orders")
    g.add_edge("orders", "customers")
    g.add_edge("customers", END)
    return g.compile()


def run(question: str) -> str:
    out = build().invoke({"messages": [("user", question)]}, config={"recursion_limit": 10})
    return out["messages"][-1].content
