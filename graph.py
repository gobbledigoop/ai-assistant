# graph.py
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


# Import tools
from tools.math_tools import (
    derivative_tool,
    integral_tool,
    limit_tool,
    solve_tool,
)

# -----------------------------
# State
# -----------------------------
from typing import TypedDict, Optional

class State(TypedDict, total=False):
    input: str
    expression: Optional[str]
    point: Optional[str]
    equation: Optional[str]
    model_output: Optional[object]
    tool_output: Optional[object]


# -----------------------------
# Model Node
# -----------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def call_model(state: State):
    user_input = state["input"]
    response = model.invoke([HumanMessage(content=user_input)])
    return {"model_output": response}


# -----------------------------
# Tool Nodes
# -----------------------------
def call_derivative(state: State):
    expr = state["expression"]
    return {"tool_output": derivative_tool.invoke({"expression": expr})}

def call_integral(state: State):
    expr = state["expression"]
    return {"tool_output": integral_tool.invoke({"expression": expr})}

def call_limit(state: State):
    expr = state["expression"]
    point = state["point"]
    return {"tool_output": limit_tool.invoke({"expression": expr, "point": point})}

def call_solve(state: State):
    eq = state["equation"]
    return {"tool_output": solve_tool.invoke({"equation": eq})}


# -----------------------------
# Router
# -----------------------------
def router(state: State):
    """
    Always return:
        (next_node_name: str, state_update: dict)

    Never return a dict as the node name.
    Never return None.
    Never return malformed state.
    """

    # Always preserve the input so downstream nodes never lose it
    text = state.get("input", "")
    lowered = text.lower()

    # --- Derivative ---
    if "derivative" in lowered or "differentiate" in lowered:
        expr = (
            lowered.replace("derivative of", "")
            .replace("differentiate", "")
            .strip()
        )
        return "derivative", {"expression": expr, "input": text}

    # --- Integral ---
    if "integral" in lowered:
        expr = lowered.replace("integral of", "").strip()
        return "integral", {"expression": expr, "input": text}

    # --- Limit ---
    if "limit" in lowered:
        try:
            before, after = lowered.split("as x->")
            expr = before.replace("limit of", "").strip()
            point = after.strip()
            return "limit", {"expression": expr, "point": point, "input": text}
        except Exception:
            # fallback to model
            return "model", {"input": text}

    # --- Solve ---
    if "solve" in lowered:
        eq = lowered.replace("solve", "").strip()
        return "solve", {"equation": eq, "input": text}

    # --- Default fallback ---
    return "model", {"input": text}
    

# -----------------------------
# Build Graph
# -----------------------------
graph = StateGraph(State)

graph.add_node("model", call_model)
graph.add_node("derivative", call_derivative)
graph.add_node("integral", call_integral)
graph.add_node("limit", call_limit)
graph.add_node("solve", call_solve)

graph.set_entry_point("model")

graph.add_conditional_edges(
    "model",
    router,
    {
        "model": END,
        "derivative": "derivative",
        "integral": "integral",
        "limit": "limit",
        "solve": "solve",
    }
)

graph.add_edge("derivative", END)
graph.add_edge("integral", END)
graph.add_edge("limit", END)
graph.add_edge("solve", END)

app = graph.compile()
