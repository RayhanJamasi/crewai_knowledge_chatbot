from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

# Uncomment the following line to use an example of a custom tool
# from crewai_knowledge_chatbot.tools.custom_tool import MyCustomTool

#use the line below for training the master agent crew
# crewai train -n 2 crew_file.py --crew CrewaiMasterAgentCrew

#specifications for the memory
memory_config = {
    "provider": "mem0",
    "config": {"user_id": "Sony_Client"}
}

#Making the Crew Class
@CrewBase
class CrewaiKnowledgeChatbotCrew():
    """CrewaiKnowledgeChatbotSony crew"""

    #accessing the agent and yask YAML files
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    #this tool can be used if you want the LLM to research from a specific docs
    #scrape_tool = ScrapeWebsiteTool() 

    #creating the Agents, giving access to memory, choosing max iterations, etc
    #researcher agent
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            #giving access to serper search tool, with searching for things in toronto ontario
            tools=[SerperDevTool(n_results=2,country="ca", locale="en-CA", location="Toronto, Ontario, Canada")], 
            memory=True, 
            memory_config=memory_config, 
            verbose=True,
            max_iter = 3
        )

    #summarizer agent
    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            memory=False,
            verbose=True,
            max_iter = 2
        )

    #QA agent
    @agent
    def support_QA_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['support_QA_agent'],
            memory=False,
            verbose=True,
            max_iter = 2
        )

    #accessing research task and assigning to researcher agent
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'],
            agent=self.researcher()
        )

    #accessing summarizing and assigning to summarizing agent
    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'],
            agent=self.reporting_analyst(),
        )

    #accessing QA check task and assigning to QA agent
    @task
    def support_task(self) -> Task:
        return Task(
            config=self.tasks_config['support_task'],
            agent=self.support_QA_agent(),
            #output_file='report.md'
        )

    #creating the crew
    @crew
    def crew(self) -> Crew:
        """Creates the CrewaiKnowledgeChatbotSony crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=False,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
