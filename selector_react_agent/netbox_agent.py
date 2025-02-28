import os
import re
import time
import json
import logging
import requests
import textwrap
from langchain.tools import Tool  # Import Tool instead of using @tool decorator
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool, render_text_description
import urllib3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
os.environ['NETBOX_URL']  = os.getenv("NETBOX_BASE_URL")
os.environ['NETBOX_TOKEN'] = os.getenv("NETBOX_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Global variables for lazy initialization
llm = None
agent_executor = None

# NetBoxController for CRUD Operations
class NetBoxController:
    def __init__(self, netbox_url, api_token):
        self.netbox = netbox_url.rstrip('/')
        self.api_token = api_token
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f"Token {self.api_token}",
        }

    def get_api(self, api_url: str, params: dict = None):
        """
        Perform a GET request to the specified NetBox API endpoint.
        """
        full_url = f"{self.netbox}/{api_url.lstrip('/')}"
        logging.info(f"GET Request to URL: {full_url}")
        logging.info(f"Headers: {self.headers}")
        logging.info(f"Params: {params}")
        
        try:
            response = requests.get(full_url, headers=self.headers, params=params, verify=False)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"GET request failed: {e}")
            return {"error": f"Request failed: {e}"}

    def post_api(self, api_url: str, payload: dict):
        full_url = f"{self.netbox}{api_url}"
        logging.info(f"POST Request to URL: {full_url}")
        logging.info(f"Headers: {self.headers}")
        logging.info(f"Payload: {json.dumps(payload)}")
    
        try:
            response = requests.post(
                full_url,
                headers=self.headers,
                json=payload,
                verify=False
            )
            logging.info(f"Response Status Code: {response.status_code}")
            logging.info(f"Response Content: {response.text}")
    
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"POST request failed: {e}")
            return {"error": f"Request failed: {e}"}

    def delete_api(self, api_url: str):
        full_url = f"{self.netbox}{api_url}"
        logging.info(f"🗑️ DELETE Request to URL: {full_url}")
        logging.info(f"Headers: {self.headers}")

        try:
            response = requests.delete(
                full_url,
                headers=self.headers,
                verify=False
            )

            logging.info(f"📡 Response Status Code: {response.status_code}")
            logging.info(f"📦 Response Content: {response.text}")

            if response.status_code == 204:
                logging.info(f"✅ Deletion successful for {full_url}")
                return {"status": "success", "message": "Deletion successful."}
            else:
                logging.warning(f"⚠️ Deletion failed. Status code: {response.status_code}")
                return {"error": f"Failed to delete. Status code: {response.status_code}"}

        except requests.exceptions.RequestException as e:
            logging.error(f"❌ DELETE request failed: {e}")
            return {"error": f"Request failed: {e}"}
        
# Function to load supported URLs with their names from a JSON file
def load_urls(file_path='netbox_apis.json'):
    if not os.path.exists(file_path):
        return {"error": f"URLs file '{file_path}' not found."}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [(entry['URL'], entry.get('Name', '')) for entry in data]
    except Exception as e:
        return {"error": f"Error loading URLs: {str(e)}"}

get_netbox_data_tool = Tool(
    name="get_netbox_data_tool",
    description="Fetch data from NetBox using the correct API URL.",
    func=lambda input_data: get_data_directly(input_data["api_url"])  # Force use of a plain string
)

def validate_tool_input(input_data, expected_keys=None, max_retries=5):
    """
    Ensures the input_data is properly formatted and contains only expected keys.
    Retries up to max_retries times if too many arguments are present.
    """
    retries = 0

    while retries < max_retries:
        # Ensure input is a dictionary
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                logging.error("❌ Invalid JSON input.")
                return {"error": "Invalid JSON format."}

        if not isinstance(input_data, dict):
            logging.error(f"🚨 Expected dictionary input but got {type(input_data)}")
            return {"error": "Invalid input format. Expected JSON object."}

        # Check if there are too many keys
        if expected_keys and len(input_data) > len(expected_keys):
            logging.warning(f"⚠️ Too many arguments provided. Expected {expected_keys}, but got {list(input_data.keys())}")
            
            # Remove unnecessary keys
            input_data = {key: input_data[key] for key in expected_keys if key in input_data}
            
            retries += 1
            time.sleep(1)  # Add a short delay before retrying
            continue
        else:
            return input_data  # Input is valid, return it

    logging.error("❌ Maximum retries exceeded due to too many arguments.")
    return {"error": "Too many arguments provided, and retries exceeded."}

def get_data_directly(api_url: str):
    input_data = validate_tool_input({"api_url": api_url}, expected_keys=["api_url"])

    if "error" in input_data:
        return input_data

    api_url = input_data["api_url"]

    if not api_url or not isinstance(api_url, str):
        return {"error": "Invalid or missing `api_url`."}

    if not re.match(r"^/api/[a-z]+/[a-z\-]+", api_url):
        logging.error(f"🚨 Invalid API URL detected: {api_url}")
        return {"error": f"Invalid API URL format: {api_url}"}

    full_url = f"{os.getenv('NETBOX_URL').rstrip('/')}/{api_url}"

    netbox_controller = NetBoxController(
        netbox_url=os.getenv("NETBOX_URL"),
        api_token=os.getenv("NETBOX_TOKEN")
    )

    logging.info(f"🚀 Fetching data from API: {full_url}")

    retries = 5
    for attempt in range(1, retries + 1):
        try:
            response = netbox_controller.get_api(api_url)
            if isinstance(response, dict) and response.get("error"):
                logging.warning(f"⚠️ API request failed on attempt {attempt}/{retries}: {response['error']}")
                if attempt == retries:
                    return {"error": f"API request failed after {retries} attempts: {response['error']}"}
                time.sleep(2)
                continue
            return {"status": "success", "message": "Data fetched successfully.", "data": response}
        except Exception as e:
            logging.error(f"❌ Error while fetching data from API: {str(e)}")
            if attempt == retries:
                return {"error": f"GET request failed after {retries} attempts: {str(e)}"}
            time.sleep(2)
    return {"error": "Unexpected failure in API request."}

# ✅ Improved Create NetBox Data Tool
create_netbox_data_tool = Tool(
    name="create_netbox_data_tool",
    description="Create new data in NetBox. Requires 'api_url' and 'payload'.",
    func=lambda input_data: create_data_handler(input_data)
)

def create_data_handler(input_data):
    input_data = validate_tool_input(input_data, expected_keys=["api_url", "payload"])

    if "error" in input_data:
        return input_data

    api_url = input_data.get("api_url")
    payload = input_data.get("payload")

    if not api_url or not isinstance(payload, dict):
        return {"error": "Both 'api_url' and a valid 'payload' dictionary are required."}

    try:
        netbox_controller = NetBoxController(
            netbox_url=os.getenv("NETBOX_URL"),
            api_token=os.getenv("NETBOX_TOKEN")
        )
        response = netbox_controller.post_api(api_url, payload)
        return {
            "status": "success",
            "message": f"Successfully created resource at {api_url}.",
            "response": response
        }
    except requests.exceptions.HTTPError as http_err:
        return {"error": f"HTTP error occurred: {http_err}"}
    except Exception as e:
        return {"error": f"POST request failed: {str(e)}"}

def delete_data_handler(input_data):
    input_data = validate_tool_input(input_data, expected_keys=["api_url", "payload"])

    if "error" in input_data:
        return input_data

    api_url = input_data.get("api_url")
    payload = input_data.get("payload", {})
    name = payload.get("name")

    if not api_url or not name:
        logging.error("❌ Missing 'api_url' or 'name' in payload.")
        return {"error": "Both 'api_url' and 'payload' with 'name' are required."}

    try:
        logging.info(f"🔍 Looking up provider '{name}' at {api_url}")

        netbox_controller = NetBoxController(
            netbox_url=os.getenv("NETBOX_URL"),
            api_token=os.getenv("NETBOX_TOKEN")
        )

        lookup_response = netbox_controller.get_api(api_url, params={'name': name})
        logging.info(f"📦 Lookup response: {json.dumps(lookup_response, indent=2)}")

        if lookup_response.get('count', 0) == 0:
            logging.warning(f"⚠️ No provider found with the name '{name}'.")
            return {"error": f"No resource found at '{api_url}' with name '{name}'."}

        entity_id = lookup_response['results'][0]['id']
        delete_url = f"{api_url.rstrip('/')}/{entity_id}/"

        logging.info(f"🗑️ Preparing to DELETE at {delete_url}")

        delete_response = netbox_controller.delete_api(delete_url)
        logging.info(f"📝 DELETE response: {delete_response}")

        if delete_response.get("status") == "success":
            return {
                "status": "success",
                "message": f"Successfully deleted '{name}' at {api_url}."
            }
        else:
            return {"error": delete_response.get("error", "Unknown error during deletion.")}

    except Exception as e:
        logging.error(f"❌ Error in delete_data_handler: {e}")
        return {"error": f"Error deleting data: {str(e)}"}
    
delete_netbox_data_tool = Tool(
    name="delete_netbox_data_tool",
    description="Delete data in NetBox. Requires 'api_url' and 'payload' with 'name'.",
    func=delete_data_handler
)

def process_agent_response(response):
    if not isinstance(response, dict):
        logging.error(f"Unexpected response format: {response}")
        return {"error": "Unexpected response format. Please check the input."}

    if response.get("status") == "success":
        return response

    if response.get("status") == "supported" and "next_tool" in response.get("action", {}):
        next_tool = response["action"]["next_tool"]
        tool_input = response["action"]["input"]

        return agent_executor.invoke({
            "input": tool_input,
            "chat_history": "",
            "agent_scratchpad": "",
            "tool": next_tool
        })

    return response

# Initialize the LLM (you can replace 'gpt-3.5-turbo' with your desired model)
#llm = Ollama(model="command-r7b", base_url="http://ollama:11434")
llm = ChatOpenAI(model_name="gpt-4o", temperature="0.1")
# ✅ Define the tools
tools = [
    Tool(name="get_netbox_data_tool", func=get_data_directly, description="Fetch data from NetBox using a valid API URL."),
    Tool(name="create_netbox_data_tool", func=create_data_handler, description="Create new data in NetBox with an API URL and payload."),
    Tool(name="delete_netbox_data_tool", func=delete_data_handler, description="Delete data in NetBox with an API URL and payload."),
]
# Extract tool names and descriptions
tool_names = ", ".join([tool.name for tool in tools])

tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
# ✅ Updated PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
    template='''
    You are a network assistant managing NetBox data using CRUD operations.

    **Strict URL Formatting Rules**  
    - Always provide URLs **exactly** as `/api/section/resource/` (e.g., `/api/ipam/ip-addresses/`)
    - Never wrap URLs in JSON objects or add `tool_input`
    - Never use `%22`, `%7B`, or any special encoding in URLs
    - Example Correct Format:
      Action Input: {{ "api_url": "/api/ipam/ip-addresses/?address=10.10.10.100" }}    

    Always use the API URL provided in the Action Input without modifying it or validating it.

    ** Assistant must strictly return plain text with no markdown formatting. 
    ** Do NOT use **bold**, *italics*, `code blocks`, bullet points, or any special characters.
    ** Only return responses in raw text without any additional formatting.
        
    **TOOLS:**  
    {tools}

    **Available Tool Names (use exactly as written):**  
    {tool_names}

    **FORMAT:**  
    Thought: [Your reasoning]  
    Action: [Tool Name]  
    Action Input: {{ "api_url": "YOUR_API_URL" }}  
    Observation: [Result]  
    Final Answer: [Answer to the User]  

    **Examples:**
    - To get interfaces from a device called R1:  
      Thought: I need to get the interfaces from R1.  
      Action: get_netbox_data_tool  
      Action Input: {{ "api_url": "/api/dcim/interfaces/?device=R1" }}
         
    - To fetch all circuits:  
      Thought: I need to retrieve all circuits from NetBox.  
      Action: get_netbox_data_tool  
      Action Input: {{ "api_url": "/api/circuits/" }}
    
    - To create a provider called "Bell Canada":  
      Thought: I need to create a provider named 'Bell Canada' with the slug 'bell'.  
      Action: create_netbox_data_tool  
      Action Input: {{ 
        "api_url": "/api/circuits/providers/", 
        "payload": {{ 
          "name": "Bell Canada", 
          "slug": "bell" 
        }} 
      }}
    
    - To delete a provider called "Bell Canada":  
      Thought: I need to create a provider named 'Bell Canada' with the slug 'bell'.  
      Action: delete_netbox_data_tool  
      Action Input: {{ 
        "api_url": "/api/circuits/providers/", 
        "payload": {{ 
          "name": "Bell Canada",
          "slug": "bell" 
        }} 
      }}
    
    **Begin!**
    
    Question: {input}  
    
    {agent_scratchpad}
    
    '''
)
logging.info(f"🛠️ Registered tools: {[tool.name for tool in tools]}")
# ✅ Pass 'tool_names' and 'tools' to the agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template.partial(
        tool_names=tool_names,
        tools=tool_descriptions
    )
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,  # Enable detailed logs
    max_iterations=2500,
    max_execution_time=1800
)
logging.info("🚀 AgentExecutor initialized with tools.")