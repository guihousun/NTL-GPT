from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
import os
from langchain_core.documents import Document
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:/NTL_Agent/tool/bigquery/ee-guihousun-f062ac4ad3a3.json"
os.environ["RIZA_API_KEY"] = "riza_01JAGAHR1XSD9G7T3FFZ7NBVA2_01JAQK95F7463W3WD4Z74NCBSF"
os.environ["EXA_API_KEY"] = "f2a2c764-dda5-4207-97c5-f5715be969f4"
os.environ["SERPER_API_KEY"] = "02c7e5622679e5e0dff090dd034a6a651cd90c1b"
# 创建 BigQuery 客户端
from langchain.tools import StructuredTool
from langchain_google_community import BigQueryLoader
import os
# from exa_py import Exa
from langchain_core.tools import tool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import (
    create_async_playwright_browser,  # A synchronous browser is available, though it isn't compatible with jupyter.\n",	  },
)# This import is required only for jupyter notebooks, since they have their own eventloop
# import nest_asyncio

# project_id = 'empyrean-caster-430308-m2'
# ee.Initialize(project=project_id)
### Build search tool ###

# 定义BigQuery查询工具
def query_bigquery(query: str) -> list[Document]:
    loader = BigQueryLoader(query)  # 使用BigQueryLoader工具
    data = loader.load()  # 加载查询结果
    return data  # 将结果转换为字符串形式返回
# 创建StructuredTool
gdelt_query_tool = StructuredTool.from_function(
    query_bigquery,
    name="gdelt_query_tool",
    description=(
        "This tool allows you to query GDELT datasets on Google BigQuery. The queries should be written in standard "
        "SQL syntax. Since quota is limited, queries should be scoped to the necessary thing relevant to the analysis.\n\n"
        "Example:\n"
        "- Query: 'SELECT GlobalEventID, SQLDATE, EventCode, EventBaseCode, Actor1Name, Actor1CountryCode, Actor2Name, "
        "Actor2CountryCode, Actor1Geo_FullName, ActionGeo_FullName, ActionGeo_Lat, ActionGeo_Long, NumMentions, "
        "NumSources, NumArticles, AvgTone, SOURCEURL FROM gdelt-bq.gdeltv2.events WHERE (EventCode IN ('051', '030', '014', "
        "'010', '015', '140') OR EventBaseCode IN ('051', '051') OR (EventCode = '060' AND EventBaseCode = '060')) AND "
        "(Actor1Name LIKE '%concert%' OR Actor2Name LIKE '%concert%' OR ActionGeo_FullName LIKE '%fireworks%' OR "
        "ActionGeo_FullName LIKE '%light show%' OR EventCode = '060' OR EventBaseCode = '015' OR EventCode = '140') AND "
        "SQLDATE BETWEEN 20240101 AND 20241231 AND NumMentions > 5 AND NumSources > 2 ORDER BY SQLDATE DESC LIMIT 100;'\n\n"
    ),
    input_type=str  # 输入类型是查询字符串
)

from typing import Any, Dict, Union
import requests
import yaml


def _get_schema(response_json: Union[dict, list]) -> dict:
    if isinstance(response_json, list):
        response_json = response_json[0] if response_json else {}
    return {key: type(value).__name__ for key, value in response_json.items()}

def _get_api_spec() -> str:
    base_url = "https://jsonplaceholder.typicode.com"
    endpoints = [
        "/posts",
        "/comments",
    ]
    common_query_parameters = [
        {
            "name": "_limit",
            "in": "query",
            "required": False,
            "schema": {"type": "integer", "example": 2},
            "description": "Limit the number of results",
        }
    ]
    openapi_spec: Dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "JSONPlaceholder API", "version": "1.0.0"},
        "servers": [{"url": base_url}],
        "paths": {},
    }
    # Iterate over the endpoints to construct the paths
    for endpoint in endpoints:
        response = requests.get(base_url + endpoint)
        if response.status_code == 200:
            schema = _get_schema(response.json())
            openapi_spec["paths"][endpoint] = {
                "get": {
                    "summary": f"Get {endpoint[1:]}",
                    "parameters": common_query_parameters,
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object", "properties": schema}
                                }
                            },
                        }
                    },
                }
            }
    return yaml.dump(openapi_spec, sort_keys=False)
api_spec = _get_api_spec()

from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
ALLOW_DANGEROUS_REQUEST = True
toolkit = RequestsToolkit(
    requests_wrapper=TextRequestsWrapper(headers={}),
    allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
)
Browser_Toolkit = toolkit.get_tools()


# from langchain_community.tools import DuckDuckGoSearchResults
#
# DuckDuckGoSearch = DuckDuckGoSearchResults(output_format="list",max_results=8)
#

from langchain_community.utilities import GoogleSerperAPIWrapper

GoogleSerpersearch = GoogleSerperAPIWrapper()
GoogleSerper_search=Tool(
        name="GoogleSerper_search",
        func=GoogleSerpersearch.run,
        description="useful for when you need to ask with search; especial query Google question and Google Places",
    )

# from langchain_community.tools.riza.command import ExecPython
# from langchain_community.tools.riza.command import ExecPython
# from langchain.agents import AgentExecutor, create_tool_calling_agent, create_openai_functions_agent
# from langchain_anthropic import ChatAnthropic
# from langchain_core.prompts import ChatPromptTemplate
# Initialize Python REPL to execute code
python_repl = PythonREPL()
### Build Python repl ###
# You can create the tool to pass to an agent
repl_tool = Tool(
    name="python_repl",
    description=(
        """
        A Python shell. Use this to execute Python commands. Input should be a valid Python command. 
        To ensure correct path handling, all file paths should be provided as raw strings (e.g., r'C:\\path\\to\\file'). 
        Paths should also be constructed using os.path.join() for cross-platform compatibility. 
        Additionally, verify the existence of files before attempting to access them.
        After executing the command, use the `print` function to output the results. 
        Finally print('Task Completed').
        For tasks related to nighttime light processing and visualization, please first seek help from the code assistant!!!
        """

    ),
    func=python_repl.run,
)

# ExecPython = ExecPython()
# Riza_tool=Tool(
#         name="Riza_Code_Interpreter",
#         description="The Riza Code Interpreter is a WASM-based isolated environment for running Python or JavaScript generated by AI agents.Only return text",
#         func=ExecPython.run,
#     )


# from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun

# wikidata = WikidataQueryRun(api_wrapper=WikidataAPIWrapper())

# print(wikidata.run("Alan Turing"))



# wolfram_alpha = WolframAlphaAPIWrapper()