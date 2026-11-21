from langchain_tavily import TavilySearch

search_tool = TavilySearch(max_results=3)

__all__ = ["search_tool"]