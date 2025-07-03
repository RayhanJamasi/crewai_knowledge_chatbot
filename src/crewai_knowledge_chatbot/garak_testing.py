from crewai_knowledge_chatbot.crew import ResearcherCrew, SummarizerCrew, QASupportCrew
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os

app = FastAPI()

#loading the environment variables from the .env file
load_dotenv()

#getting the API keys securely (compared to hardcoding which is not good for security)
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

#setting the environment variables
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["SERPER_API_KEY"] = serper_api_key
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

class prompt_input(BaseModel):
    prompt: str

@app.post("/test_rsch")
def garak_test_rsch(request: prompt_input):
    prompt = request.prompt
    
    chosen_crew = ResearcherCrew().crew()

    inputs = {"topic": prompt, "memories": "(N/A)"}
    response = chosen_crew.kickoff(inputs=inputs)

    return{"text": str(response)}

@app.post("/test_guard")
def garak_test_guard(request: prompt_input):
    prompt = request.prompt
    
    from crewai_knowledge_chatbot.main import is_question_understandable
    choice, rephrased_prompt = is_question_understandable(prompt, "")

    return{"text": f"Choice: {choice}, rephrased_prompt: {rephrased_prompt}"}

@app.post("/test_guard_and_rsch")
def garak_test_guard_and_rsch(request: prompt_input):
    prompt = request.prompt

    from crewai_knowledge_chatbot.main import is_question_understandable
    choice, rephrased_prompt = is_question_understandable(prompt, "")

    if choice == True:
        chosen_crew = ResearcherCrew().crew()

        inputs = {"topic": prompt, "memories": "(N/A)"}
        response = chosen_crew.kickoff(inputs=inputs)

        return{"text": str(response)}

    elif choice == False:
        return {"text": "I'm afraid I can only assist with proper questions related to banking and finance. Could you clarify your request?"}

    return {"text" : "ERROR"}

@app.post("/test_guard_and_sum")
def garak_test_guard_and_sum(request: prompt_input):
    prompt = request.prompt

    from crewai_knowledge_chatbot.main import is_question_understandable
    choice, rephrased_prompt = is_question_understandable(prompt, "")

    if choice == True:
        chosen_crew = SummarizerCrew().crew()

        inputs = {"topic": prompt, "memories": "(N/A)"}
        response = chosen_crew.kickoff(inputs=inputs)

        return{"text": str(response)}

    elif choice == False:
        return {"text": "I'm afraid I can only assist with proper questions related to banking and finance. Could you clarify your request?"}

    return {"text" : "ERROR"}

@app.post("/test_guard_and_qa")
def garak_test_guard_and_qa(request: prompt_input):
    prompt = request.prompt

    from crewai_knowledge_chatbot.main import is_question_understandable
    choice, rephrased_prompt = is_question_understandable(prompt, "")

    if choice == True:
        chosen_crew = QASupportCrew().crew()

        inputs = {"topic": prompt, "memories": "(N/A)"}
        response = chosen_crew.kickoff(inputs=inputs)

        return{"text": str(response)}

    elif choice == False:
        return {"text": "I'm afraid I can only assist with proper questions related to banking and finance. Could you clarify your request?"}

    return {"text" : "ERROR"}
