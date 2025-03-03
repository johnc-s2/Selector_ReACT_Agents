import os
import json
import logging
import requests
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Slack API credentials
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL_ID = os.getenv("SLACK_CHANNEL_ID")

# Logging
logging.basicConfig(level=logging.INFO)

# ---------------------------- SLACK API FUNCTION ---------------------------- #
def send_slack_message(input_data):
    """
    Sends a message to a Slack channel.
    """
    logging.info(f"📥 Received input data for Slack message: {input_data}")

    # ✅ Ensure input is JSON
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data.replace("'", '"'))
        except json.JSONDecodeError:
            logging.error("❌ Invalid JSON input format.")
            return {"error": "Invalid JSON format."}

    if not isinstance(input_data, dict) or "message" not in input_data:
        logging.error("❌ Missing 'message' field in input.")
        return {"error": "Input must be a dictionary with a 'message' key."}

    message = input_data["message"]

    # ✅ Construct Slack API request
    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "channel": SLACK_CHANNEL_ID,
        "text": message
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        json_response = response.json()

        if not json_response.get("ok"):
            logging.error(f"❌ Slack API error: {json_response}")
            return {"error": "Failed to send message."}

        logging.info(f"✅ Message sent successfully: {json_response}")
        return {"status": "Message sent to Slack."}

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Slack API request failed: {str(e)}")
        return {"error": "Failed to communicate with Slack API."}

# ---------------------------- LANGCHAIN TOOL ---------------------------- #
send_slack_tool = Tool(
    name="send_slack_message",
    description="Sends a message to a Slack channel.",
    func=send_slack_message
)

# Define the tool list
tools = [send_slack_tool]

# Extract tool names and descriptions
tool_names = ", ".join([tool.name for tool in tools])
tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# ---------------------------- LLM & PROMPT TEMPLATE ---------------------------- #
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

slack_prompt = PromptTemplate(
    input_variables=["message", "agent_scratchpad", "tool_names", "tools"],
    template="""
    You are an AI Agent that can send messages to Slack.

    You have access to:
    - **send_slack_message**: Sends a message to a Slack channel.

    Example:
    - *Question*: Tell the team that the deployment was successful.
      Thought: I need to send a message to Slack.
      Action: send_slack_message
      Action Input: {{"message": "The deployment was successful!"}}
      Observation: [Slack API response]
      Final Answer: The message was sent to Slack.

    **Begin!**

    Message: {message}

    {agent_scratchpad}
    """
)

# ✅ Create LangChain Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=slack_prompt.partial(
        tool_names=tool_names,
        tools=tool_descriptions
    )
)

# ✅ Create AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=10,
    max_execution_time=60
)

logging.info("🚀 Slack AI Agent initialized.")

# ---------------------------- TEST EXECUTION ---------------------------- #
if __name__ == "__main__":
    test_input = {
        "message": "Hello team, this is an AI-powered message from Slack Agent!",
        "agent_scratchpad": ""
    }

    logging.info(f"🚀 Sending Input to AgentExecutor: {json.dumps(test_input, indent=2)}")

    response = agent_executor.invoke(test_input)

    logging.info(f"📩 Raw Response from AgentExecutor: {json.dumps(response, indent=2) if isinstance(response, dict) else response}")

    print("\n📝 AGENT RESPONSE:", response)
