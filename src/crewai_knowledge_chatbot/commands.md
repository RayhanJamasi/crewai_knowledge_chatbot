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


