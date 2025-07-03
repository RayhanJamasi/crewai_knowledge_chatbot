import sys
import os
import warnings

#ignoring warnings 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

#using the absolute path of one folder above, 
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
from crewai_knowledge_chatbot.crew import CrewaiMasterAgentCrew
from openai import OpenAI
import yaml
from pathlib import Path

#setting base directory to location of main
base_dir = Path(__file__).resolve().parent

#getting path for yaml files
agents_path = base_dir / "config" / "agents.yaml"
tasks_path = base_dir / "config" / "tasks.yaml"

#loading yaml files
with agents_path.open() as agent_file:
    agents_data = yaml.safe_load(agent_file)

with tasks_path.open() as task_file:
    tasks_data = yaml.safe_load(task_file)

#getting the info from the agents and task file which will be given to the master agent
research_agent_info = agents_data['researcher']
research_task_info = tasks_data['research_task']
reporting_analyst_agent_info = agents_data['reporting_analyst']
reporting_analyst_task_info = tasks_data['reporting_task']
support_QA_agent_info = agents_data['support_QA_agent']
support_QA_agent_task_info = tasks_data['support_task']

#adding them all to the crew_info array
crew_info = [
    research_agent_info,
    research_task_info,
    "\n\n",
    reporting_analyst_agent_info,
    reporting_analyst_task_info,
    "\n\n",
    support_QA_agent_info,
    support_QA_agent_task_info
]

#sensitive categories that the master agent will have
sensitive_info = [
    "Hate / Identity Hate",
    "Sexual",
    "Violence",
    "Suicide and Self Harm",
    "Threat",
    "Sexual Minor",
    "Guns / Illegal Weapons",
    "Controlled / Regulated Substances",
    "Criminal Planning / Confessions",
    "PII (Personally Identifiable Information)",
    "Harassment",
    "Profanity",
    "Security",
    "Bias"
]

#creating instance of OpenAI for my guardrails
openai_client = OpenAI()

#loading the environment variables from the .env file
load_dotenv()

#getting the API keys securely (compared to hardcoding which is not good for security)
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

#setting the environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPER_API_KEY"] = serper_api_key
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

#declaring the iteration number variable for training and the inputs 
MA_i = 5
MA_inputs = { 
            "crew_info": f"{crew_info}",
            "sensitive_info": f"{sensitive_info}",
        }

#training
try:
    CrewaiMasterAgentCrew().crew().train(
        n_iterations=MA_i,
        inputs=MA_inputs,
        filename="master_agent_training.pkl"
    )
    print("Training completed successfully.")

#throw exception in case of error
except Exception as e:
    print(f"Error during training: {e}")

