# Selector_ReACT_Agents
Selector NL and Raw Data ReACT Agents including email, NetBox, and ServiceNow agents


## Setup .env 

For all the agents to work you will need a .env file located inside the select_react_agents subfolder (where the .py files are). There is already a .gitignore so your credentials are safe in this file 

OPENAI_API_KEY=<openai API key>

NETBOX_BASE_URL=<NetBox URL>

NETBOX_TOKEN=<Netbox API key>

SMTP_RELAY_SERVER=smtp.gmail.com

SMTP_RELAY_PORT=587

SMTP_RELAY_USERNAME=<your application email>

SMTP_RELAY_PASSWORD=<your application password>

SERVICENOW_URL=<ServiceNow URL>

SERVICENOW_USER=<service now user>

SERVICENOW_PASSWORD=<service now password>

SELECTOR_AI_API_KEY=<selector API key>

SELECTOR_NL_URL="https://<selector URL>/api/collab2-slack/command"

SELECTOR_DATA_URL="https://<selector URL>/internal/collab/copilot/v1/chat"

## Docker 

Works on Mac, Windows, or Linux with docker-compose

Please install Docker Desktop for Windows or Mac 

## Bring up the system

1. Git clone the repo 

2. Navigate to Selector_ReACT_Agents

3. docker-compose up

3a. If you make modifications to the code use docker-compose up --build to include your changes

4. Visit localhost:8501 

5. Ask questions about devices in Selector

6. Watch the AI reasoning in the terminal logs 

## Selector AI Agents

In your phrasing / prompt you have access to 2 Selector AI Agents: Natural Language and Raw Data 

You also have access to a NetBox Agent; a ServiceNow Problems Agent; and an E-Mail Agent (with more to come)

You can ask "What can you tell me about device S6?" or "What can you tell me about device S6 using the raw data?" 

You can ask "Gather what you know about device S6 in Selector and NetBox and create a problem in ServiceNow if there are descrepancies; email a comprehensive report to John at johnc@selector.ai" 

Explore and share your findings and contact John directly with issues or suggestions or whatever