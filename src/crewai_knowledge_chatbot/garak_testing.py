from crewai_knowledge_chatbot.crew import ResearcherCrew, SummarizerCrew, QASupportCrew
from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from fastapi.responses import HTMLResponse


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

@app.get("/demo", response_class=HTMLResponse)
def demo():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Simple Form</title>
        <!-- Bootstrap CDN for styling -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-KyZXEJv5l1+K7SB5Rlz43k2f2r4iJ9L0vYtSaMkA9t2zA7czlf/WP/OeQ6iZDZ5b" crossorigin="anonymous">
        <style>
            body {
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f1f1f1;
            }

            .container {
                text-align: center;
                background: #fff;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                max-width: 400px;
                width: 100%;
            }

            .form-control {
                width: 100%;
                padding: 15px;
                font-size: 18px;
                border-radius: 8px;
                background-color: #fafafa;
                border: 1px solid #ddd;
            }

            .form-control:focus {
                border-color: #007bff;
                background-color: #fff;
            }

            h1 {
                margin-bottom: 30px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Simple Form</h1>
            <form>
                <!-- Normal Textbox 1: Name -->
                <div class="form-group mb-3">
                    <label for="nameInput">Name</label>
                    <input type="text" class="form-control" id="nameInput" placeholder="Enter your name">
                </div>

                <!-- Normal Textbox 2: Email -->
                <div class="form-group mb-3">
                    <label for="emailInput">Email address</label>
                    <input type="email" class="form-control" id="emailInput" placeholder="Enter your email">
                </div>
            </form>
        </div>

        <!-- Bootstrap JS (optional, for interactivity) -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js" integrity="sha384-pzjw8f+ua7Kw1TIq0r6p8cCUA2+k4hYmZ+fOdY+RvdHwXk5b6nd6yDE9Ff5B1S4X" crossorigin="anonymous"></script>
    </body>
    </html>
    """

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
