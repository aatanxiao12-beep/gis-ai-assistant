"""
GIS AI Standard Assistant

用法:
    python run_agent.py                              # 交互模式（流式）
    python run_agent.py "GML Curve 的定义？"          # 单次查询（流式）
"""

import sys

from agent import create_agent
from agent.graph import run_agent_stream


def run_question(agent, question: str):
    print(f"\n▶ {question}\n")

    for event in run_agent_stream(agent, question):
        step = event["step"]
        msg_type = event["msg_type"]

        if msg_type == "ai":
            tc = event.get("tool_calls", [])
            if tc:
                for t in tc:
                    q = str(t.get("args", {}).get("query", ""))[:100]
                    print(f"  ⚙ 调用 {t['name']}({q})")
            else:
                print(f"\n{'─' * 50}")
                print(event["content"])
                print(f"{'─' * 50}")

        elif msg_type == "tool":
            preview = event["content"][:150].replace("\n", " ")
            print(f"     ← {event.get('tool_name', '?')}: {preview}...")


def interactive():
    print("GIS AI Standard Assistant")
    print("输入问题开始，输入 quit 退出\n")
    agent = create_agent()

    while True:
        try:
            q = input("\n▶ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见")
            break
        if q.lower() in ("quit", "exit", "q"):
            print("再见")
            break
        if not q:
            continue
        run_question(agent, q)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        agent = create_agent()
        run_question(agent, sys.argv[1])
    else:
        interactive()
