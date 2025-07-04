## ################### NAVIGATION ###################
# Navigate to grandchild directory 
cd src/crewai_knowledge_chatbot


## ################### STARTING SERVER ###################
# Launch uvicorn garak_testing
uvicorn garak_testing:app --reload


## ################### SINGLE AGENTS ###################
# Garak test Rsch_crew
garak --model_type rest -G garak_configs/garak_config_rsch.json \
    --probes donotanswer.MisinformationHarms \
    --generations 2


# Garak test Guard agent
garak --model_type rest -G garak_configs/garak_config_guard.json \
    --probes donotanswer.MisinformationHarms \
    --generations 2



## ################### GUARDRAIL AND AGENT ###################
# Garak test GUARD + RSCH
garak --model_type rest -G garak_configs/garak_config_guard_rsch.json \
    --probes lmrc.Profanity \
    --generations 2


# Garak test GUARD + SUM
garak --model_type rest -G garak_configs/garak_config_guard_sum.json \
    --probes lmrc.Profanity \
    --generations 2


# Garak test GUARD + QA
garak --model_type rest -G garak_configs/garak_config_guard_qa.json \
    --probes lmrc.Profanity \
    --generations 2 \ 
    --report_prefix src/crewai_knowledge_chatbot/garak_runs_qa/report







### details needed for CrewAI and Asenion Integration ###

API Url

Bearer Token

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}",
    "Content-Type": "application/json"
}

required inputs --> retried from requests.get(f"{API_URL}/inputs") --> parse it with .json()



# example input:
kickoff_data = {
    "inputs": {
        "topic": "AI trends",  #replace it with the actual inputs
        "year": "2025"
    }
}
