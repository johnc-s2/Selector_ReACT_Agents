import os
import json
import time
import logging
import requests
import urllib3
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SELECTOR_AI_API_KEY = os.getenv("SELECTOR_AI_API_KEY")
SELECTOR_API_URL = os.getenv("SELECTOR_DATA_URL")

logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ---------------------------- SELECTOR AI API CLASS ---------------------------- #
class SelectorAPI:
    """
    Queries Selector AI API for natural language responses.
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
        Sends a natural language query to the Selector AI API and logs all requests and responses.
        Increases timeout to handle long API processing time.
        """
        payload = {"content": input_data["content"]}

        # ✅ Set a higher timeout value (default: 20)
        timeout_value = int(os.getenv("SELECTOR_AI_TIMEOUT", 20))  

        logging.info(f"📡 Requesting Selector AI with payload: {json.dumps(payload, indent=2)}")
        logging.info(f"⏳ Timeout for request: {timeout_value} seconds")

        retries = 3  # Retry mechanism

        for attempt in range(retries):
            try:
                start_time = time.time()  # Start timer
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=timeout_value)
                elapsed_time = time.time() - start_time  # End timer

                logging.info(f"✅ HTTP Status Code: {response.status_code}")
                logging.info(f"⏱️ API Response Time: {elapsed_time:.2f} seconds")

                response.raise_for_status()

                json_response = response.json()
                logging.info(f"✅ FULL RESPONSE FROM SELECTOR AI:\n{json.dumps(json_response, indent=4)}")

                return json_response  # ✅ Return full response

            except requests.exceptions.Timeout:
                logging.error(f"⏳ API request timed out after {timeout_value} seconds (Attempt {attempt+1}/{retries})")
            except requests.exceptions.RequestException as e:
                logging.error(f"❌ Selector AI request failed (Attempt {attempt+1}): {e}")

            time.sleep(2)  # Wait before retrying

        return {"error": f"Failed to fetch response after {retries} attempts."}

# ---------------------------- TOOL FUNCTION ---------------------------- #
def ask_selector(input_data):
    """
    Queries Selector AI for natural language responses and extracts only the 'content' field.
    """
    logging.info(f"📥 Received input data for ask_selector: {json.dumps(input_data, indent=2) if isinstance(input_data, dict) else input_data}")

    # ✅ Ensure the input is always a dictionary
    if not isinstance(input_data, dict):
        logging.warning(f"⚠️ Converting input to dictionary. Received: {input_data}")
        input_data = {"input": input_data}  # Wrap string input in dictionary

    if "input" not in input_data:
        logging.error("❌ Missing 'input' field in ask_selector.")
        return {"error": "Missing 'input' field in JSON object."}

    # ✅ Convert "input" -> "content" for Selector AI API
    selector_client = SelectorAPI()
    response = selector_client.ask({"content": input_data["input"]})

    # Log the raw API response
    logging.info(f"📊 Raw API Response: {json.dumps(response, indent=2)}")

    if isinstance(response, dict) and "error" in response:
        logging.warning(f"⚠️ API returned an error: {response['error']}")
        return response  # Pass error back up

    # ✅ Extract and return ONLY the "content" field
    extracted_content = response.get("content", "No valid response received.")

    logging.info(f"📝 Extracted Final Answer: {extracted_content}")

    return extracted_content  # ✅ Return only the actual answer

# ---------------------------- LANGCHAIN TOOL ---------------------------- #
ask_selector_tool = Tool(
    name="ask_selector",
    description="Queries Selector AI API for natural language responses.",
    func=ask_selector
)

# ✅ Define the tools
tools = [ask_selector_tool]

# Extract tool names and descriptions
tool_names = ", ".join([tool.name for tool in tools])
tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# ---------------------------- LLM & PROMPT TEMPLATE ---------------------------- #
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

selector_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
    template="""
    You are an AI Agent that interacts with the Selector AI API to retrieve insights in natural language.

    You have access to:
    - **ask_selector**: Queries Selector AI API for natural language responses.

    Example:
    - *Question*: What are the latest anomalies?
      Thought: I need to query Selector AI for information about devices or other network-related items.
      Action: ask_selector
      Action Input: {{"input": "{input}"}}
      Observation: [Selector AI API response]
      Final Answer: The response from Selector AI is [parsed answer].

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
logging.info("🚀 Selector AI Agent initialized.")

# ---------------------------- TEST EXECUTION ---------------------------- #
if __name__ == "__main__":
    test_question = "What can you tell me about device S6?"
    test_input = {
        "input": test_question,
        "agent_scratchpad": ""  # ✅ Fix: Ensure required keys are present
    }

    logging.info(f"🚀 Sending Input to AgentExecutor: {json.dumps(test_input, indent=2)}")

    response = agent_executor.invoke(test_input)

    logging.info(f"📩 Raw Response from AgentExecutor: {json.dumps(response, indent=2) if isinstance(response, dict) else response}")

    print("\n📝 AGENT RESPONSE:", response)
