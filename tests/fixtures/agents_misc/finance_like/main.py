import argparse

from finance_agents.agents import run_agent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("request")
    a = p.parse_args()
    print(run_agent(a.name, a.request))


if __name__ == "__main__":
    main()
