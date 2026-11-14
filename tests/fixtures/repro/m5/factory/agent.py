from crewai import Agent, Crew, Task


def build_agent(role, steps):
    return Agent(role=role, goal="g", backstory="b", max_iter=steps, llm="gpt-4o-2024-08-06")


researcher = build_agent("researcher", 6)
writer = build_agent("writer", 6)
crew = Crew(agents=[researcher, writer], tasks=[Task(description="d", expected_output="e", agent=researcher)])
