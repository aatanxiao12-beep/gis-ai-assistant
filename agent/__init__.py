"""
GIS AI Standard Assistant

用法:
    from agent import create_agent, run_agent

    agent = create_agent()
    answer = run_agent(agent, "GML Curve 是什么？")
"""

from agent.graph import create_agent, run_agent
from agent.state import AgentState

__all__ = ["create_agent", "run_agent", "AgentState"]
