"""Model ids from each place they can come from."""

import os

import yaml
from agents import Agent

import config
from config import MODEL_NAME

CFG = yaml.safe_load(open("models.yml"))

a = Agent(name="a", instructions="i", model="gpt-4o-2024-08-06")
b = Agent(name="b", instructions="i", model=MODEL_NAME)
c = Agent(name="c", instructions="i", model=config.settings.model)
d = Agent(name="d", instructions="i", model=os.getenv("NEW_MODEL", "gpt-4.1-mini-2025-04-14"))
e = Agent(name="e", instructions="i", model=CFG["model"])
f = Agent(name="f", instructions="i", model=os.environ["AGENT_MODEL"])
g = Agent(name="g", instructions="i", model=os.environ["RUNTIME_ONLY"])
