import requests
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field


# def get_country_gdp(country_code: str, year: int = None):
#     """
#     从世界银行 API 获取指定国家的 GDP 数据
#     :param country_code: 国家代码（如 'CN'、'US'、'JP'）
#     :param year: 年份（如 2022），默认返回最新年份
#     :return: GDP 数值（美元）
#     """
#     url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/NY.GDP.MKTP.CD?format=json&per_page=100"
#
#     response = requests.get(url)
#     data = response.json()
#
#     if not data or len(data) < 2:
#         return f"未找到国家代码 {country_code} 的数据"
#
#     gdp_data = data[1]
#
#     for entry in gdp_data:
#         if year:
#             if str(entry["date"]) == str(year) and entry["value"] is not None:
#                 return {
#                     "country": country_code,
#                     "year": year,
#                     "gdp": entry["value"]
#                 }
#         else:
#             if entry["value"] is not None:
#                 return {
#                     "country": country_code,
#                     "year": entry["date"],
#                     "gdp": entry["value"]
#                 }
#
#     return f"{country_code} 在 {year} 没有有效的 GDP 数据"
#
# print(get_country_gdp("China",2020))

# get_gdp_tool = StructuredTool.from_function(
#     get_gdp,
#     name="get_administrative_division_tool",
#     description="用于获取某个国家的 GDP 数据",
# )
# tools = [
# get_gdp_tool
# ]
#
# llm_GPT = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_retries=3)
# memory = MemorySaver()
# graph = create_react_agent(llm_GPT, tools=tools, checkpointer=memory)
#
# # 简易流式问答
# def stream_graph_updates(user_input):
#     events = graph.stream(
#         {"messages": [("user", user_input)]}, {"configurable": {"thread_id": "sgh","recursion_limit": 7}}, stream_mode="values"
#     )
#     for event in events:
#         event["messages"][-1].pretty_print()
# print("Starting interactive session...")
# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break
#         print(f"Received input: {user_input}")  # 调试信息
#         stream_graph_updates(user_input)
#     except Exception as e:
#         print(f"An error occurred: {e}")
