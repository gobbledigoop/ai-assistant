from dotenv import load_dotenv
load_dotenv()

import re
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.agents import create_agent
#from langchain_core.agents import create_tool_calling_agent

# Math tools
from tools.math_tools import (derivative_tool, integral_tool, limit_tool, solve_tool)
# Internet tools
from tools.internet_tools import search_tool

#------------------------------------------
# This function is responsible for rendering the response when writing math.
# It looks for <math>...</math> tags in the response and renders the content as LaTeX.
#------------------------------------------
def render_response(response: str):
    # Split the response into parts: text and math
    parts = re.split(r'(<math>(.*?)</math>)', response, flags=re.DOTALL)
    
    for part in parts:
        if part.startswith("<math>") and part.endswith("</math>"):
            # Extract the math content and render it as LaTeX
            math_content = part[6:-7]  # Remove <math> and </math> tags
            st.latex(math_content)
        else:
            # Render regular text
            st.write(part)


#------------------------------------------
# This is the main file that runs the Streamlit app and serves as the entry point
# for the project. It sets up the agent, defines the user interface, and handles user interactions.
#------------------------------------------

@st.cache_resource
def load_agent():
    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    tools = [derivative_tool, integral_tool, limit_tool, solve_tool, search_tool]
    agent = create_agent(model=model, 
    tools=tools, 
    system_prompt="You are a math assistant. When displaying mathematical expressions or results, always wrap them in <math> and </math> tags. For example: The derivative is <math>2x</math>."
    )

    #------------------------------------------
    # This is the prompt, currently it's useless but I plan to use it later when adding 
    # some complexity and langgraph
    #------------------------------------------
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a new ai agent made for my first project! You can perform various mathematical operations using the provided tools."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    #------------------------------------------
    return agent

# Page Configuration
st.set_page_config(page_title="AI Assistant", page_icon=":robot_face:")
st.title("AI Assistant")

# Initialize conversation history in session_state if it doesn't exist yet
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redraw conversation history on every rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_response(message["content"])
        else:
            st.write(message["content"])
            
# Wait for user input
if user_input := st.chat_input("Type your message here..."):
    # Add user message to conversation history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Call the agent and render response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
    try:
        agent = load_agent()
        for chunk in agent.stream(
            {"messages": [{"role": "human", "content": user_input}]},
            {"recursion_limit": 8},
            stream_mode="updates"
        ):
            if "model" in chunk:
                content = chunk["model"]["messages"][-1].content
                if content:
                    full_response += content
                    message_placeholder.write(full_response + "▌")  # Show a cursor while streaming

        message_placeholder.empty()
        render_response(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(f"Something went wrong: {e}")


        # try:
        #     agent = load_agent()
        #     result = agent.invoke(
        #         {"messages": [{"role": "human", "content": user_input}]},
        #         {"recursion_limit": 8}
        #     )
        #     full_response = result["messages"][-1].content
        #     message_placeholder.empty()
        #     render_response(full_response)
        #     st.session_state.messages.append({"role": "assistant", "content": full_response})

        # except Exception as e:
        #     st.error(f"Something went wrong: {e}")