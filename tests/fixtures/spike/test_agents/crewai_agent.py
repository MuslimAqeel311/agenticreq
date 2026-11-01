"""CrewAI test agent: 2 agents, the manager delegates to the analyst.

Ordinary user code -- knows nothing about the spike harness.
"""
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool

from shared_tools import SYSTEM_RULES, USER_TASK, get_inventory, pay_invoice, search_policy, submit_order


def main() -> str:
    llm = LLM(model="anthropic/claude-sonnet-4-5", max_tokens=1024)  # short alias on purpose
    analyst = Agent(
        role="Policy Analyst",
        goal="Answer stock and policy questions accurately.",
        backstory="You look up inventory and company policy. " + SYSTEM_RULES,
        tools=[tool(get_inventory), tool(search_policy)],
        llm=llm, allow_delegation=False, verbose=False,
    )
    manager = Agent(
        role="Procurement Manager",
        goal="Handle customer procurement requests, delegating research to the Policy Analyst.",
        backstory=SYSTEM_RULES,
        tools=[tool(submit_order), tool(pay_invoice)],
        llm=llm, allow_delegation=True, verbose=False,
    )
    task = Task(description=USER_TASK, expected_output="Stock level and refund policy summary.", agent=manager)
    crew = Crew(agents=[manager, analyst], tasks=[task], process=Process.sequential, verbose=False)
    answer = str(crew.kickoff())
    print("AGENT OUTPUT:", answer)
    return answer


if __name__ == "__main__":
    main()
