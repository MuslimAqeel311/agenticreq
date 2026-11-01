"""Market tools call yfinance (an outside service); planner tools are plain arithmetic; one tool writes."""

import json

import yfinance as yf
from langchain_core.tools import tool


def _history(ticker: str):
    return yf.Ticker(ticker.upper()).history(period="1y")


@tool
def get_stock_quote(ticker: str) -> str:
    """Current price of a ticker."""
    return json.dumps({"price": yf.Ticker(ticker).info.get("currentPrice")})


@tool
def compare_tickers(tickers: list[str]) -> str:
    """Compare one-year returns."""
    return json.dumps({t: float(_history(t)["Close"].iloc[-1]) for t in tickers})


@tool
def loan_amortization(principal: float, annual_rate_pct: float, years: int) -> str:
    """Monthly payment of a loan."""
    r = annual_rate_pct / 1200
    n = years * 12
    return json.dumps({"monthly": round(principal * r / (1 - (1 + r) ** -n), 2) if r else principal / n})


@tool
def estimate_tax(gross_income: float) -> str:
    """A flat 20% estimate."""
    return json.dumps({"tax": round(gross_income * 0.2, 2)})


@tool
def create_order(ticker: str, usd_amount: float) -> str:
    """Place a paper order with the broker."""
    raise RuntimeError("never in a mock run")


MARKET_TOOLS = [get_stock_quote, compare_tickers]
PLANNER_TOOLS = [loan_amortization, estimate_tax]
TRADER_TOOLS = [get_stock_quote, create_order]
