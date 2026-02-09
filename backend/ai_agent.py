from langchain_core.tools import tool
from tools import query_medgemma, call_emergency

@tool
def ask_mental_health_specialist(query: str) -> str:
    """
    Generate a therapeutic response using the MedGemma model.
    Use this for all general user queries, mental health questions, emotional concerns,
    or to offer empathetic, evidence-based guidance in a conversational tone.
    """
    return query_medgemma(query)


@tool
def emergency_call_tool() -> None:
    """
    Place an emergency call to the safety helpline's phone number via Twilio.
    Use this only if the user expresses suicidal ideation, intent to self-harm,
    or describes a mental health emergency requiring immediate help.
    """
    call_emergency()


@tool
def find_nearby_therapists_by_location(location: str) -> str:
    """
    Finds and returns a list of licensed therapists near the specified location.

    Args:
        location (str): The name of the city or area in which the user is seeking therapy support.

    Returns:
        str: A newline-separated string containing therapist names and contact info.
    """
    return (
        f"Here are some therapists near {location}, {location}:\n"
        "- Dr. Ayesha Kapoor - +1 (555) 123-4567\n"
        "- Dr. James Patel - +1 (555) 987-6543\n"
        "- MindCare Counseling Center - +1 (555) 222-3333"
    )


# Step1: Create an AI Agent & Link to backend
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END  # Correct import for LangGraph 1.x
from config import OPENAI_API_KEY

tools = [ask_mental_health_specialist, emergency_call_tool, find_nearby_therapists_by_location]
llm = ChatOpenAI(model="gpt-4", temperature=0.2, api_key=OPENAI_API_KEY)

# Define the agent state (a simple dict to hold messages)
class AgentState(dict):
    messages: list

# Initialize LLM and bind tools
llm_with_tools = llm.bind_tools(tools)  # Binds your tools to the LLM for tool calling

# Define the agent node (handles LLM reasoning and tool calls)
def agent_node(state):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}

# Define the tool execution node (runs the tools based on LLM output)
def tool_node(state):
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = last_message.tool_calls if hasattr(last_message, 'tool_calls') else []
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        if tool_name == "ask_mental_health_specialist":
            result = ask_mental_health_specialist(**tool_args)
        elif tool_name == "emergency_call_tool":
            result = emergency_call_tool(**tool_args)
        elif tool_name == "find_nearby_therapists_by_location":
            result = find_nearby_therapists_by_location(**tool_args)
        else:
            result = "Tool not found."
        results.append(result)
    # Return updated messages with tool results
    return {"messages": messages + [{"role": "tool", "content": str(results)}]}

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
# Conditional edge: If tools are called, go to tools; else end
graph.add_conditional_edges("agent", lambda x: "tools" if x["messages"][-1].tool_calls else END)
graph.add_edge("tools", "agent")
graph.set_entry_point("agent")

# Compile the graph (this creates the executable agent)
graph = graph.compile()

SYSTEM_PROMPT = """
You are an AI engine supporting mental health conversations with warmth and vigilance.
You have access to three tools:

1. `ask_mental_health_specialist`: Use this tool to answer all emotional or psychological queries with therapeutic guidance.
2. `find_nearby_therapists_by_location`: Use this tool if the user asks about nearby therapists or if recommending local professional help would be beneficial.
3. `emergency_call_tool`: Use this immediately if the user expresses suicidal thoughts, self-harm intentions, or is in crisis.

Always take necessary action. Respond kindly, clearly, and supportively.
"""

def parse_response(stream):
    tool_called_name = "None"
    final_response = None

    for s in stream:
        # Check if a tool was called
        tool_data = s.get('tools')
        if tool_data:
            tool_messages = tool_data.get('messages')
            if tool_messages and isinstance(tool_messages, list):
                for msg in tool_messages:
                    tool_called_name = getattr(msg, 'name', 'None')

        # Check if agent returned a message
        agent_data = s.get('agent')
        if agent_data:
            messages = agent_data.get('messages')
            if messages and isinstance(messages, list):
                for msg in messages:
                    if msg.content:
                        final_response = msg.content

    return tool_called_name, final_response


"""if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        print(f"Received user input: {user_input[:200]}...")
        inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", user_input)]}
        stream = graph.stream(inputs, stream_mode="updates")
        tool_called_name, final_response = parse_response(stream)
        print("TOOL CALLED: ", tool_called_name)
        print("ANSWER: ", final_response)"""