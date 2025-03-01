import os
import json
import time
import logging
import requests
import urllib3
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenCVE API Credentials
OPENCVE_USER = os.getenv("OPENCVE_USER")
OPENCVE_PASSWORD = os.getenv("OPENCVE_PASSWORD")

logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ---------------------------- OPENCVE API CLASS ---------------------------- #
class OpenCVEAPI:
    """
    Queries OpenCVE API to retrieve CVEs based on vendor and version.
    """

    def __init__(self):
        self.base_url = "https://app.opencve.io/api"

    def get_cves(self, vendor, version):
        """
        Fetch CVEs for a given vendor and version.
        """
        url = f"{self.base_url}/cve?vendor={vendor}&version={version}"
        logging.info(f"📡 Requesting OpenCVE API: {url}")

        try:
            response = requests.get(url, auth=(OPENCVE_USER, OPENCVE_PASSWORD), timeout=20)
            response.raise_for_status()
            data = response.json()

            if "results" in data:
                cve_list = [
                    {"id": cve["cve_id"], "description": cve["description"]}
                    for cve in data["results"]
                ]
                return cve_list

            return {"error": "No CVEs found for the given vendor and version."}

        except requests.exceptions.RequestException as e:
            logging.error(f"❌ OpenCVE request failed: {e}")
            return {"error": str(e)}


# ---------------------------- TOOL FUNCTION ---------------------------- #
def fetch_cves_from_opencve(input_data):
    """
    Queries OpenCVE API for CVEs based on vendor and version extracted from Selector AI.
    """
    logging.info(f"📥 Raw input data received for fetch_cves_from_opencve: {input_data}")

    # ✅ Ensure input is a dictionary
    if isinstance(input_data, str):
        try:
            input_data = json.loads(input_data.replace("'", '"'))  # Convert string to JSON
            logging.info(f"✅ Parsed string into dictionary: {input_data}")
        except json.JSONDecodeError:
            logging.error(f"❌ Failed to parse input data: {input_data}")
            return {"error": "Invalid input format. Expected a valid JSON dictionary."}

    if not isinstance(input_data, dict):
        logging.error(f"❌ Still invalid input type: {type(input_data)}")
        return {"error": "Invalid input format. Expected a dictionary."}

    # ✅ Extract and normalize vendor and version
    vendor = input_data.get("vendor", "").strip().lower()
    version = input_data.get("version", "").strip()

    # ✅ Vendor normalization mapping
    vendor_mapping = {
        "juniper networks": "juniper",
        "juniper": "juniper",
        "cisco systems": "cisco",
        "cisco": "cisco",
        "arista networks": "arista",
        "arista": "arista"
    }
    vendor = vendor_mapping.get(vendor, vendor)

    if not vendor or not version:
        logging.error(f"❌ Missing required fields - Vendor: {vendor}, Version: {version}")
        return {"error": "Missing 'vendor' or 'version' field."}

    logging.info(f"🔍 Querying OpenCVE for Vendor: {vendor}, Version: {version}")

    # ✅ Construct OpenCVE API URL
    url = f"https://app.opencve.io/api/cve?vendor={vendor}&version={version}"
    logging.info(f"📡 OpenCVE API Request URL: {url}")

    # ✅ Ensure authentication is set
    opencve_user = os.getenv("OPENCVE_USER")
    opencve_password = os.getenv("OPENCVE_PASSWORD")
    if not opencve_user or not opencve_password:
        logging.error("❌ OpenCVE API credentials not set.")
        return {"error": "OpenCVE API credentials not found."}

    # ✅ Make the API request
    try:
        response = requests.get(url, auth=(opencve_user, opencve_password))
        response.raise_for_status()
        json_response = response.json()

        logging.info(f"📊 OpenCVE API Response: {json.dumps(json_response, indent=2)}")

        # ✅ Handle no results, try a fallback version (drop patch number)
        if json_response.get("count", 0) == 0:
            fallback_version = ".".join(version.split(".")[:2])
            if fallback_version != version:
                logging.warning(f"⚠️ No results for {version}. Retrying with {fallback_version}...")
                fallback_url = f"https://app.opencve.io/api/cve?vendor={vendor}&version={fallback_version}"
                response = requests.get(fallback_url, auth=(opencve_user, opencve_password))
                response.raise_for_status()
                json_response = response.json()
                logging.info(f"📊 OpenCVE Fallback API Response: {json.dumps(json_response, indent=2)}")

        # ✅ Extract CVEs
        cve_list = json_response.get("results", [])
        if not cve_list:
            return {"message": f"No CVEs found for {vendor} {version}."}

        formatted_cves = [{"cve_id": cve["cve_id"], "description": cve["description"][:200] + "..."} for cve in cve_list[:10]]

        logging.info(f"✅ Successfully retrieved {len(formatted_cves)} CVEs for {vendor} {version}.")
        return {"cve_count": json_response["count"], "cves": formatted_cves}

    except requests.exceptions.RequestException as e:
        logging.error(f"❌ OpenCVE API request failed: {str(e)}")
        return {"error": "Failed to fetch CVEs from OpenCVE."}

# ---------------------------- LANGCHAIN TOOL ---------------------------- #
fetch_cves_tool = Tool(
    name="fetch_cves",
    description="Fetches CVEs from OpenCVE API based on vendor and version.",
    func=fetch_cves_from_opencve
)

# ✅ Define tools
tools = [fetch_cves_tool]

# Extract tool names and descriptions
tool_names = ", ".join([tool.name for tool in tools])
tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# ---------------------------- LLM & PROMPT TEMPLATE ---------------------------- #
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

opencve_prompt = PromptTemplate(
    input_variables=["vendor", "version", "agent_scratchpad", "tool_names", "tools"],
    template="""
    You are an AI Agent that retrieves CVEs using the OpenCVE API.

    You have access to:
    - **fetch_cves**: Queries OpenCVE API based on vendor and version.

    Example:
    - *Question*: What are the CVEs for Juniper running Junos 21.4R5?
      Thought: I need to query OpenCVE for CVEs for Juniper Junos 21.4R5.
      Action: fetch_cves
      Action Input: {{"vendor": "{vendor}", "version": "{version}"}}
      Observation: [OpenCVE API response]
      Final Answer: The response from OpenCVE is [parsed CVE list].

    **Begin!**
    
    Vendor: {vendor}
    Version: {version}

    {agent_scratchpad}
    """
)

# ✅ Create LangChain Agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=opencve_prompt.partial(
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
    max_iterations=25,
    max_execution_time=360
)

# Log initialization
logging.info("🚀 OpenCVE AI Agent initialized.")

# ---------------------------- TEST EXECUTION ---------------------------- #
if __name__ == "__main__":
    test_input = {
        "vendor": "juniper",
        "version": "21.4R5",
        "agent_scratchpad": ""  # ✅ Fix: Ensure required keys are present
    }

    logging.info(f"🚀 Sending Input to AgentExecutor: {json.dumps(test_input, indent=2)}")

    response = agent_executor.invoke(test_input)

    logging.info(f"📩 Raw Response from AgentExecutor: {json.dumps(response, indent=2) if isinstance(response, dict) else response}")

    print("\n📝 AGENT RESPONSE:", response)
