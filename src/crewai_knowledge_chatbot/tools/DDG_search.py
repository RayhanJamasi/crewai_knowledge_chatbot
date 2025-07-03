from crewai_tools import RagTool
from duckduckgo_search import DDGS

class DuckDuckGoTool(RagTool):
    name: str = "DuckDuckGo Search"
    description: str = "Useful for searching recent and relevant information from the internet."

    def _run(self, query: str) -> str:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            if not results:
                return "No results found."
            return "\n\n".join([
                f"{r['title']}\n{r['href']}\n{r['body']}" for r in results
            ])


# General format
# class MyCustomTool(BaseTool):
#     name: str = "Name of my tool"
#     description: str = "Clear description for what this tool is useful for, you agent will need this information to use it."

#     def _run(self, argument: str) -> str:
#         # Implementation goes here
#         return "this is an example of a tool output, ignore it and move along."
