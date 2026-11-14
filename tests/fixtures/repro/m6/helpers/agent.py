from crewai import Agent, Crew, Task

from tools import SaveFileTool, SearchTool

writer = Agent(role="writer", goal="g", backstory="b", tools=[SaveFileTool(), SearchTool()], llm="gpt-4o-2024-08-06", max_iter=5)
crew = Crew(agents=[writer], tasks=[Task(description="d", expected_output="e", agent=writer)])
