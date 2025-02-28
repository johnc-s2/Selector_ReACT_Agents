import os
import json
import logging
import streamlit as st
import urllib3
from dotenv import load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI

## Import Selector AI Agents
from selector_natural_language_agent import ask_selector_tool

# Import other AI Agents
from netbox_agent import tools as netbox_tools, prompt_template as netbox_prompt
from email_agent import send_email_tool  
from servicenow_agent import tools as servicenow_tools, prompt_template as servicenow_prompt

# ============================================================
# **🚀 Load Environment Variables**
# ============================================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ============================================================
# **🔧 Configure Logging & Security**
# ============================================================
logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================
# **🤖 Define the LLM**
# ============================================================
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

# ============================================================
# **📡 Initialize Agents**
# ============================================================
servicenow_agent = initialize_agent(
    tools=servicenow_tools, llm=llm,
    agent='structured-chat-zero-shot-react-description',
    prompt=servicenow_prompt, verbose=True
)

netbox_agent = initialize_agent(
    tools=netbox_tools, llm=llm,
    agent='structured-chat-zero-shot-react-description',
    prompt=netbox_prompt, verbose=True
)

# ============================================================
# **📡 Selector AI Agent Functions**
# ============================================================
def selector_agent_func(input_text: str) -> dict:
    """Calls the natural language Selector AI Agent."""
    response = ask_selector_tool.func({"input": input_text})
    return {"output": response}  # ✅ Wrap in a dict for consistency

## Other AI Agent Functions
def netbox_agent_func(input_text: str) -> str:
    return netbox_agent.invoke(f"NetBox: {input_text}")

def email_agent_func(input_data) -> dict:
    """Sends an email report via the email agent."""
    try:
        if isinstance(input_data, str):
            input_data = json.loads(input_data)
        if not isinstance(input_data, dict) or not all(k in input_data for k in ["recipient", "subject", "message"]):
            return {"status": "error", "error": "Invalid email data format"}
        return send_email_tool.func(input_data)
    except Exception as e:
        return {"status": "error", "error": str(e)}

def servicenow_agent_func(input_text: str) -> str:
    return servicenow_agent.invoke(f"ServiceNow: {input_text}")

# ============================================================
# **🔹 Create LangChain Tools**
# ============================================================
selector_tool = Tool(
    name="Selector AI Agent",
    func=selector_agent_func,
    description="Use this tool for AI-generated insights from Selector AI."
)

netbox_tool = Tool(
    name="NetBox Agent", func=netbox_agent_func,
    description="Use for NetBox operations and queries."
)

email_tool = Tool(
    name="Email Agent", func=email_agent_func,
    description="Send an email with 'recipient', 'subject', and 'message'."
)

servicenow_tool = Tool(
    name="ServiceNow Agent", func=servicenow_agent_func,
    description="Use for ServiceNow incident management operations."
)

# ============================================================
# **🤖 Main Parent Routing Agent**
# ============================================================
parent_tools = [selector_tool,netbox_tool, email_tool, servicenow_tool]

parent_agent = initialize_agent(
    tools=parent_tools, llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

logging.info(f"🚀 Main Parent Routing Agent Initialized with Tools: {[tool.name for tool in parent_tools]}")

# ============================================================
# **🛰️ Streamlit UI - Chat with Selector AI**
# ============================================================
st.title("🔍 Selector ReACT AI Agent")
st.write("Drive infrastructure with your natural language!")

# User input text area
user_input = st.text_area("💬 Enter your question:")

# Conversation History (Stored in Session)
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if st.button("Send"):
    if not user_input:
        st.warning("⚠️ Please enter a question.")
    else:
        # 🚀 Invoke the Parent Agent
        response = parent_agent.invoke(user_input)

        # ✅ Extract response text
        response_text = response if isinstance(response, str) else response.get("output", "No valid response received.")

        # ✅ Display AI Agent's response
        st.write(f"### **💬 Question:** {user_input}")
        st.write(f"### **📡 Response:** {response_text}")

        # ✅ Save conversation history
        st.session_state.conversation.append({"role": "user", "content": user_input})
        st.session_state.conversation.append({"role": "assistant", "content": response_text})

# ============================================================
# **📜 Display Conversation History**
# ============================================================
st.write("### 💬 Conversation History")
for chat in st.session_state.conversation:
    role = "👤 You" if chat["role"] == "user" else "🤖 Selector AI"
    st.write(f"**{role}:** {chat['content']}")
