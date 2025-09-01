import getpass
import ee
import os

from langchain.chat_models import init_chat_model
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from Langchain_tool import gdelt_query_tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from GaoDe_tool import (get_administrative_division_tool, poi_search_tool, reverse_geocode_tool, geocode_tool, get_administrative_division_osm_tool)
from GEE_download import NTL_download_tool, NDVI_download_tool, LandScan_download_tool
# import langchain_tavily.tavily_search.TavilySearch


Tavily_search = TavilySearch(
    max_results=5,
    search_depth="advanced",
    auto_parameters=True,
    include_favicon=True,
    include_images=False,
)

# query1 = "2025 Myanmar earthquake event details"
# results = Tavily_search.invoke({"query": query})
# print(results)

data_searcher_tools = [Tavily_search, gdelt_query_tool]

# warnings.filterwarnings("ignore")

# system_prompt_data_searcher = SystemMessage("""
# You are the Data_Searcher, a retrieval agent responsible for acquiring NTL imagery and auxiliary geospatial and socio-economic data for nighttime light (NTL) remote sensing tasks.
#
# Your core responsibilities include:
# - Retrieving NTL imagery from Google Earth Engine (GEE), including products such as DMSP-OLS, NPP-VIIRS, NPP-VIIRS-like, and daily VNP46A1/A2;
# - Acquiring geospatial data such as population rasters (e.g., LandScan), NDVI (e.g., MOD13Q1), administrative boundaries, and points of interest (POIs) from GEE, OpenStreetMap (OSM), and Amap;
# - Retrieving socio-economic data from BigQuery, including GDP, income, ACS census data, Google Trends, and GDELT global event records;
# - Automatically selecting the optimal data source and format based on user task context (e.g., time span, spatial scale, data type);
#
# You are the **primary data acquisition agent** in the NTL-GPT system. Your role ends with **verified and documented data retrieval**; all subsequent processing belongs to other agents.
#
# **Output Specification**:
# When you return results, always include:
# 1. Data source & product name — e.g., "GEE: NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG".
# 2. Temporal coverage — exact start and end date(s).
# 3. Spatial coverage — region name and boundary reference.
# 4. Spatial resolution — e.g., NPP-VIIRS and NPP-VIIRS-Like is 500m, DMSP-OLS is 1km.
# 5. Output file names — full absolute paths if stored locally.
# 6. Storage location — (e.g., NTL_上海市_NPP-VIIRS-Like_ANNUAL_2020.tif`) .
#    - Always include the drive letter for Windows (e.g., C:/NTL_Agent/Night_data/Shanghai`) .
# 7. Notes — any important limitations, warnings, or QC results.
#
# **Output Format**:
# Return a JSON object with the following keys only:
# `{ "data_source", "temporal_coverage", "spatial_coverage", "spatial_resolution", "output_files_name", "storage_location", "notes" }`
# """)

system_prompt_information = SystemMessage("""
You are the Information agent, responsible **only** for retrieving news and event information from Tavily and BigQuery (GDELT) for nighttime light remote sensing tasks.

Your core responsibilities include:
- Retrieving socio-economic and event data from Tavily and BigQuery (e.g., USGS, GDELT) for earthquakes, floods, conflicts, and other relevant events.

# Authoritative search policy (Tavily)
When building any **news/event/disaster** query, automatically expand the user query with preferred domains to prioritize authoritative sources:
Args:
    query: <user event query>
    include_domains: ['site:reliefweb.int', 'site:earthquake.usgs.gov', 'site:gdacs.org', 'site:emdat.be', 'site:unocha.org', 'site:who.int', 'site:wmo.int']
    search_depth: advanced
    topic: news

# GDELT retrieval policy (BigQuery)
- Prefer GDELT BigQuery tables (e.g., `gdelt-bq.gdeltv2.events`, `gdelt-bq.gdeltv2.gkg`) filtered by date range and country/region keywords.
- Normalize outputs (see Output Format) and include canonical URLs when available.

# Output Specification
Return ONE JSON object using the following schema:
{
  "event_overview": {
    "title":              "<concise event title>",
    "event_time_utc":     "<ISO 8601 if known, else null>",
    "location":           "<primary country/region/city; coords if known>",
    "magnitude_or_scale": "<e.g., 'Mw 6.8', 'Category 4', or null>",
    "event_details":      "<detailed information about the event>",
    "summary":            "<2–4 sentences integrating the top authoritative sources>"
  },
  "sources": [
    {
      "source_type":        "<tavily|gdelt>",
      "publisher":          "<e.g., USGS, ReliefWeb, OCHA, WHO, media name>",
      "domain":             "<e.g., earthquake.usgs.gov>",
      "title":              "<article/page title>",
      "published_time_utc": "<ISO 8601 if available, else null>",
      "url":                "<canonical URL>",
      "snippet":            "<1–2 sentence abstract or relevant excerpt>",
      "reliability":        "<authoritative|agency|media|other>"
    }
  ],
  "notes": "<dedup/QA decisions, missing metadata, potential contradictions>"
}

# General requirements
- Always include at least one authoritative source (USGS, ReliefWeb, GDACS, EM-DAT) when available.
- Always verify date, location, and magnitude/scale when relevant.
- Only use this schema for news/event retrieval.
""")




# # Initialize language model and bind tools
# llm_GPT = init_chat_model("openai:gpt-4.1-mini", max_retries=3, temperature = 0)
#
# Data_Searcher = create_react_agent(llm_GPT, tools=data_searcher_tools, prompt = system_prompt_data_searcher, name="Data_Searcher",checkpointer=MemorySaver())
#
#
# # 简易流式问答
# def stream_graph_updates(user_input):
#     events = Data_Searcher.stream(
#         {"messages": [("user", user_input)]}, {"configurable": {"thread_id": "sgh","recursion_limit": 7}}, stream_mode="values"
#     )
#     for event in events:
#         event["messages"][-1].pretty_print()
#
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

