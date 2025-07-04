import sys
import os
import logging
import warnings

# ChatGPT code to remove the warnings for demo purposes (mem0 v1.0 is deprecated and it is defaulting
# to v1.1)
# logging.basicConfig(
#     filename='warnings.log',  
#     level=logging.WARNING,   
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )

# Redirect warnings to the logging system
logging.captureWarnings(True)

#ignoring warnings 
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# using the absolute path of one folder above, 
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# using SSL and Certifi to remove the error where I was unable to download WhisperAI
# base model from the internet due to security reasons.
import ssl
import certifi
ssl._create_default_https_context = ssl._create_unverified_context

import glob
import json
from datetime import datetime
import whisper
import platform
from dotenv import load_dotenv
from crewai_knowledge_chatbot.crew import CrewaiKnowledgeChatbotCrew, CrewaiMasterAgentCrew
from mem0 import MemoryClient
from openai import OpenAI
import threading
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

#loading whisper base english model
model = whisper.load_model("base.en")

#creating instance of OpenAI for my guardrails
openai_client = OpenAI()

#loading the environment variables from the .env file
load_dotenv()

#getting the API keys securely (compared to hardcoding which is not good for security)
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")
memo_api_key = os.getenv("MEM0_API_KEY")

#setting the environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPER_API_KEY"] = serper_api_key
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'
os.environ["MEM0_API_KEY"] = memo_api_key

#instance of memory client for mem0
client = MemoryClient()

#creating an instance of the crews outside the run method, this way a new instance of the crew
#is not needed ever time I run the code
crew_instance = CrewaiKnowledgeChatbotCrew().crew()
MA_crew_instance = CrewaiMasterAgentCrew().crew()

#if memory_type = 0, it will use short term memory. if memory_type = 1, it 
#will use long term memory
memory_type = 0 

#main logic flow for the chatbot
def run():
    #creating array that holds short term history
    history = []

    MA_inputs = { 
            "crew_info": f"{crew_info}",
            "sensitive_info": f"{sensitive_info}",
                }

    MA_response = MA_crew_instance.kickoff(inputs=MA_inputs)

    #splitting the result and assigning it to each variable
    researcher, summarizer, support = str(MA_response).split(",", 2)

    # #adding the categories for each agent to an array
    requirements = [researcher, summarizer, support]

    #formatting the requirements into one message
    requirements_printed_msg = f"""
    \n\nThe requirements needed to download for each agent are:
    
    Researcher Agent: {requirements[0].strip()}\n
    Summarizer Agent: {requirements[1].strip()}\n
    Support QA Agent: {requirements[2].strip()}\n\n
    """

    # printing the requirements that need to be
    print(requirements_printed_msg)
    
    current_dir = os.getcwd() # get current directory
    probe_info_list_qa = [] #each element looks like: [<probe_name>, <detector_name>]
    results_qa = [] #each element looks like: [<tests_passed>, <total_tests_ran>]

    #checking which agents need to be checked for security and accessing that specific agent
    if "Security" in requirements[0].strip():
        pass

    if "Security" in requirements[1].strip():
        pass
    
    if "Security" in requirements[2].strip():
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix =f"{current_dir}/src/crewai_knowledge_chatbot/garak_runs_qa/{time}log"

        os.system("pip install --upgrade tokenizers")
        os.system(f"garak --model_type rest -G src/crewai_knowledge_chatbot/garak_configs/garak_config_guard_qa.json \
                    --probes lmrc.Profanity \
                    --generations 2 \
                    --report_prefix='{prefix}'") 

        list_of_reports_qa = glob.glob(f'{current_dir}/src/crewai_knowledge_chatbot/garak_runs_qa/*.report.jsonl')

        latest_qa_report = max(list_of_reports_qa, key=os.path.getmtime)

        report_info = []

        #opening the report and parsing it
        with open(latest_qa_report, "r") as report:
            for line in report:
                line_info = json.loads(line)
                report_info.append(line_info)

        for dict_log in report_info:
            try:
                if dict_log["entry_type"] == "eval":
                    probe_info_list_qa.append([dict_log["probe"], dict_log["detector"]])
                    results_qa.append([dict_log["passed"], dict_log["total"]])
            except KeyError:
                pass

    #intro message
    print("Welcome to TD Bank, how may I help you? (type \"voice\" to do speech to text for 6 seconds)")

    #for loop which contains the conversation logic
    while True:

        #get user input
        user_input = input("\n\nUser (type \"voice\" to do speech-to-text for 6 seconds): ")

        #leave conversation if user says any of the specific words
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! It was nice talking to you.")
            break

        #performing speech to text if the user types in the string "voice"
        if user_input.strip().lower() == "voice":
            user_input = voice_to_text()

        #ensures chat history only holds the last 4 exchanges
        if len(history) >= 8:
            chat_history = "\n".join(history[-8:])
        else:
            chat_history = "\n".join(history)

        #ensuring the history passed to the guardrail agent only has the last 3 exchanges
        if len(history) >= 6:
            
            #1 exchange is equal to 1 user prompt and 1 chatbot output
            last_3_exchanges = "\n".join(history[-6:])
            input_validator, user_input = is_question_understandable(user_input, last_3_exchanges)

        else:
            input_validator, user_input = is_question_understandable(user_input, chat_history)

        #ensures the string is not empty and the guardrail agent has approved of the prompt
        if user_input.strip() and input_validator:

            #if memory type was long term:
            if memory_type == 1:

                #searches semantically for top 2 best results from both the user and chatbot
                past_memories_TD = client.search(user_input, user_id="TD_Assistant")[:2]
                past_memories_user = client.search(user_input, user_id="User")[:2]

                #declaring past_memories_str which will hold the snippets of long-term memory
                past_memories_str = ""

                #going through the memory field for all 4 memories
                for i, memory in enumerate(past_memories_TD + past_memories_user):

                    #adding them to the variable
                    past_memories_str += f"Memory {i+1}: {memory['memory']}\n\n"

                #sending user prompt and results from long term memory semantic search
                inputs = {
                    "topic": f"{user_input}",
                    "memories": f"top 2 results from long term memory: {past_memories_str}",
                }

                #launching the crew
                response = crew_instance.kickoff(inputs=inputs)

                #printing response
                print(f"\nAssistant: {response}\n\n")

                #calling function in charge for adding user/chatbot messages to mem0
                add_memories(user_input, response, client)

            #if short term memory is enabled
            elif memory_type == 0:   

                #sending user prompt and short term history of conversation             
                inputs = {
                    "topic": f"{user_input}",
                     "memories": f"{chat_history}"
                }

                #launching the crew
                response = crew_instance.kickoff(inputs=inputs)

                #printing the response
                print(f"\nAssistant: {response}\n\n")

            #adding the exchange between user and chatbot to short term memory 
            history.append(f"User: {user_input}")
            history.append(f"Assistant: {response}")

        #if the guardrail agent did not approve of prompt and/or the prompt was empty:
        #the chatbot outputs this hard coded message asking for clarification
        else:
            print("\nI'm afraid I can only assist with proper questions related to banking and finance. Could you clarify your request?")

#voice to text method which uses FFmpeg and WhisperAI
def voice_to_text():
    #informs the user that they can speak
    print("Speak now... (6 seconds)")

    #getting the os that they are using
    os_name = platform.system()

    #running the correct recording command depending on their os
    if os_name == "Darwin":
        os.system("ffmpeg -y -f avfoundation -i ':0' -t 6 output.mp3")

    #WARNING: the line below has not been checked as I do not own a windows device
    elif os_name == "Windows":
        os.system('ffmpeg -y -f dshow -i audio="default" -t 6 output.mp3')

    else:
        print("Sorry, voice is not supported on your device")

    #tells the user they are finished recording
    print("Recording complete, transcribing...")

    #using whisperAI to transcribe the audio
    result = model.transcribe("output.mp3")

    #printing the transcribed message
    print("\n" + result["text"])

    #returning the transcribed message so the chatbot take the prompt
    return result["text"]

#guardrail agent in charge of deciding whatever if a prompt should be allowed to pass or not. This
#agent also will give a rephrased prompt of what was said and help add context for the agents
def is_question_understandable(user_input, last_3_exchanges):
    system_msg = """
    You work at TD Bank as a compliance-focused prompt reviewer. Your job is to filter out
    irrelevant or risky topics before they reach internal systems. Determine if the user input
    is relevant to TD, TD services, banking or finance. Your other job is to look at the user prompt
    and rephrase it based off the memories you've been given (if any). Use the current chat history to help
    determine if the user is continuing a relevant conversation to the conditions mentioned before.
    For example, they may ask question which may not be finance related directly, but by referencing
    the memories you've been given, in reality, it actually was finance-related. Use that knowledge to
    rephrase the prompt and and give it context to what the user is actually asking.
    
    If it's not clearly related, or it's  incoherent, fragmented, or vague, respond with (case sensitive):
    no, <reason>
    If it was related, respond exactly with (case sensitive):
    yes, <rephrased_prompt>

    respond in only this format, the comma between the yes/no and the reason/prompt is very very important as 
    I am using this to split the string later on. Without that, I will recieve an error
    """

    #declaring the model and giving it the last 3 messages as well as the current user
    #prompt. This is to help the model decide if the current prompt is relevant or not
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Current conversation so far:\n{last_3_exchanges}\n\nUser's new message: {user_input}"}
        ]
    )

    #getting the message from the chatbot
    result = response.choices[0].message.content.strip()

    #splitting the guardrail response and the rephrased prompt
    decision, rephrased_prompt = result.split(",", 1)

    #returning True if response is "yes", else return False
    return decision.strip().lower() == "yes", rephrased_prompt.strip()

#function that uses threading to add the memories to mem0 at the same time. This is because if I do 
#not use this method, it can take up to 25 seconds before the user can type a new output
def add_memories(user_input, response, client):
    def add_user_memories():
        client.add(user_input, user_id = "User")

    def add_chatbot_memories():
        client.add(str(response), user_id = "TD_Assistant")

    threading.Thread(target = add_user_memories).start()
    threading.Thread(target = add_chatbot_memories).start()

