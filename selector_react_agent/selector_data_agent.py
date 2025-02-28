import os
import json
import time
import logging
import requests
import urllib3
from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SELECTOR_AI_API_KEY = os.getenv("SELECTOR_AI_API_KEY")
SELECTOR_API_URL = os.getenv("SELECTOR_NL_URL")

logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------- SELECTOR RAW DATA API CLASS ---------------------------- #
class SelectorRawDataAPI:
    """
    Queries Selector AI API for raw data responses.
    """
    def __init__(self):
        self.api_url = SELECTOR_API_URL
        self.headers = {
            "Authorization": f"Bearer {SELECTOR_AI_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def ask(self, input_data):
        """
        Sends a command to the Selector AI API and retrieves raw JSON data.
        """
        payload = {"command": input_data["command"]}  # ✅ Change "content" to "command"

        # ✅ Set a higher timeout value (default: 20)
        timeout_value = int(os.getenv("SELECTOR_AI_TIMEOUT", 20))

        logging.info(f"📡 Requesting Selector AI RAW Data with payload: {json.dumps(payload, indent=2)}")
        logging.info(f"⏳ Timeout for request: {timeout_value} seconds")

        retries = 3  # Retry mechanism

        for attempt in range(retries):
            try:
                start_time = time.time()
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=timeout_value)
                elapsed_time = time.time() - start_time

                logging.info(f"✅ HTTP Status Code: {response.status_code}")
                logging.info(f"⏱️ API Response Time: {elapsed_time:.2f} seconds")

                response.raise_for_status()

                json_response = response.json()
                logging.info(f"✅ FULL RAW DATA RESPONSE FROM SELECTOR AI:\n{json.dumps(json_response, indent=4)}")

                return json_response  # ✅ Return full response

            except requests.exceptions.Timeout:
                logging.error(f"⏳ API request timed out after {timeout_value} seconds (Attempt {attempt+1}/{retries})")
            except requests.exceptions.RequestException as e:
                logging.error(f"❌ Selector AI RAW Data request failed (Attempt {attempt+1}): {e}")

            time.sleep(2)

        return {"error": f"Failed to fetch response after {retries} attempts."}

# ---------------------------- TOOL FUNCTION ---------------------------- #
def ask_selector_raw(input_data):
    """
    Queries Selector AI for raw JSON data and extracts the 'data' field.
    """
    logging.info(f"📥 Received input data for ask_selector_raw: {json.dumps(input_data, indent=2)}")

    # ✅ Ensure input is a dictionary
    if not isinstance(input_data, dict):
        logging.warning(f"⚠️ Converting input to dictionary. Received: {input_data}")
        input_data = {"command": input_data}  # Wrap string input in dictionary

    if "command" not in input_data:
        logging.error("❌ Missing 'command' field in ask_selector_raw.")
        return {"error": "Missing 'command' field in JSON object."}

    # ✅ Send request to the raw data API
    selector_client = SelectorRawDataAPI()
    response = selector_client.ask({"command": input_data["command"]})  # ✅ Send with correct API format

    # Log the raw API response
    logging.info(f"📊 Raw API Response: {json.dumps(response, indent=2)}")

    if isinstance(response, dict) and "error" in response:
        logging.warning(f"⚠️ API returned an error: {response['error']}")
        return response  # Pass error back up

    # ✅ Extract and return ONLY the "data" field
    extracted_content = response.get("data", "No valid response received.")

    logging.info(f"📝 Extracted Final Answer: {extracted_content}")

    return extracted_content  # ✅ Return only the actual raw data

# ---------------------------- LANGCHAIN TOOL ---------------------------- #
ask_selector_raw_tool = Tool(
    name="ask_selector_raw",
    description="Queries Selector AI API for raw JSON data.",
    func=ask_selector_raw
)

# ✅ Define the tools
tools = [ask_selector_raw_tool]

# Extract tool names and descriptions
tool_names = ", ".join([tool.name for tool in tools])
tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# ---------------------------- LLM & PROMPT TEMPLATE ---------------------------- #
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

selector_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
    template="""
    You are an AI Agent that interacts with the Selector AI API to retrieve raw data in JSON format.

    You have access to:
    - **ask_selector_raw**: Queries Selector AI API for raw JSON data.

    Example:
    - *Question*: Describe the Kubernetes pod status.
      Thought: I need to query Selector AI for raw data about Kubernetes pod status.
      Action: ask_selector_raw
      Action Input: {{"command": "#select describe the kubernetes pod status"}}
      Observation: [Selector AI API raw data response]
      Final Answer: The response from Selector AI is [parsed JSON data].

    **Begin!**
    
    Question: {input}

    {agent_scratchpad}
    """
)

# ✅ Ensure prompt variables are correctly set
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=selector_prompt.partial(
        tool_names=tool_names,
        tools=tool_descriptions
    )
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,
    max_iterations=25,
    max_execution_time=360
)

# Log initialization
logging.info("🚀 Selector RAW DATA AI Agent initialized.")

# ---------------------------- TEST EXECUTION ---------------------------- #
if __name__ == "__main__":
    test_command = "#select describe the kubernetes pod status"
    test_input = {
        "command": test_command,
        "agent_scratchpad": ""  # ✅ Ensure required keys are present
    }

    logging.info(f"🚀 Sending Input to AgentExecutor: {json.dumps(test_input, indent=2)}")

    response = agent_executor.invoke(test_input)

    logging.info(f"📩 Raw Response from AgentExecutor: {json.dumps(response, indent=2) if isinstance(response, dict) else response}")

    print("\n📝 AGENT RESPONSE:", response)
