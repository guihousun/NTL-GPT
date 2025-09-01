# ntl_knowledge_base_searcher.py
# -*- coding: utf-8 -*-

import os
from typing import Annotated, Literal, Sequence, Optional
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import os


# tool_registry.py
TOOL_REGISTRY = {
    "NTL_Knowledge_Base": {},
    "SDGSAT-1_strip_removal_tool": {},
    "extract_urban_area_use_change_point": {},
    "NTL_trend_analysis": {},
    "noaa20_VIIRS_preprocess": {},
    "SDGSAT1_radiometric_calibration_tool": {},
    "classify_light_types_from_rrli_rbli": {},
    "NTL_raster_statistics": {},
    "VNP46A2_angular_correction_tool": {},
    "SDGSAT1_compute_index": {},
    "ContourTree_NoArcpy": {},
    "bvntl_index_tool": {},
    "vnci_index_tool": {},
    "dmsp_preprocess_tool": {},
    "extract_road_mask_from_grayscale_using_otsu": {},
    "NTL_composite_local_tool": {},
    "NTL_composite_GEE_tool": {},
    "NTL_anomaly_detection_tool": {},
    "reverse_geocode_tool": {},
    "geocode_tool": {},
    "NTL_download_tool": {},
    "NDVI_download_tool": {},
    "LandScan_download_tool": {},
    "get_administrative_division_Amap_tool": {},
    "get_administrative_division_osm_tool": {},
    "poi_search_tool": {},
    "tavily_search": {},
    "gdelt_query_tool": {},
}

REGISTRY_NAMES = sorted(TOOL_REGISTRY.keys())

# ---- Your internal RAG tools (as provided in your project) ----
# They must be StructuredTool or compatible tools.
from NTL_Knowledge_Base import (
    NTL_Literature_Knowledge,  # theory, equations, definitions
    NTL_Solution_Knowledge,    # workflows, tools, datasets, parameter patterns
    NTL_Code_Knowledge,        # concise Python/GEE code snippets
)

# ----------------------------------------------------------------
# State
# ----------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]
    response_mode: Optional[str]  # auto | workflow | theory | code | mixed
    locale: Optional[str]         # en | zh, etc.
    need_citations: Optional[bool]


# ----------------------------------------------------------------
# Tools registry for the ToolNode
# ----------------------------------------------------------------

TOOLS = [NTL_Literature_Knowledge, NTL_Solution_Knowledge, NTL_Code_Knowledge]
# TOOLS = [NTL_Solution_Knowledge]

# ----------------------------------------------------------------
# Relevance grader
# ----------------------------------------------------------------

# def grade_documents(state) -> Literal["agent", "rewrite"]:
#     print("---CHECK RELEVANCE---")
#
#     class Grade(BaseModel):
#         """Binary score for relevance check."""
#         binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")
#
#     model = init_chat_model("openai:gpt-4.1-mini", max_retries=3, temperature=0)
#     llm_with_tool = model.with_structured_output(Grade)
#
#     prompt = PromptTemplate(
#         template=(
#             "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
#             "Retrieved document:\n{context}\n\n"
#             "User question: {question}\n\n"
#             "If the document contains keywords or semantic meaning related to the question, grade it as relevant.\n"
#             "This is not a stringent test; the goal is to filter out erroneous retrievals.\n"
#             "Return 'yes' or 'no' in field 'binary_score'."
#         ),
#         input_variables=["context", "question"],
#     )
#     chain = prompt | llm_with_tool
#
#     messages = state["messages"]
#     last_message = messages[-1]
#     question = messages[0].content
#     docs = last_message.content
#
#     scored = chain.invoke({"question": question, "context": docs})
#     score = (scored.binary_score or "").strip().lower()
#
#     if score == "yes":
#         print("---DECISION: DOCS relevant---")
#         return "agent"
#     else:
#         print("---DECISION: DOCS NOT relevant---")
#         return "rewrite"


# ----------------------------------------------------------------
# Agent node (decides what to retrieve and how to answer)
# ----------------------------------------------------------------

def agent(state: State):
    print("---CALL AGENT---")

    # Pull mode/flags from state (with safe defaults)
    mode = (state.get("response_mode") or "auto").lower()
    need_citations = bool(state.get("need_citations", True))
    locale = state.get("locale", "en")

    system_prompt_text = SystemMessage(f"""
    You are the NTL_Knowledge_Base, a retrieval-first assistant with three internal sources:
    - NTL_Literature_Knowledge (theory, equations, definitions)
    - NTL_Solution_Knowledge (workflows, tools, datasets, parameter patterns)
    - NTL_Code_Knowledge (concise Python/GEE code snippets)
    - Prioritize searching the NTL_Solution_Knowledge database and NTL_Code_Knowledge. Only when there is a strong need for NTL_Literature_Knowledge should the databases be consulted.
    
    TOOL REGISTRY (allowed tool names only):
    {REGISTRY_NAMES}

    Return content in one of four modes:

    1) workflow → STRICT JSON **object** with keys:
    {{
      "task_id": "...", "task_name": "...", "category": "...", "description": "...",
      "steps": [
        // builtin tool
        {{
          "type": "builtin_tool",
          "name": "<one of registry tools>",
          "input": {{...}},
          "note": "<≤100 chars>",
          "sources": ["Store:DocIdOrTitle", "..."]  // optional but recommended
        }},
        // geospatial code
        {{
          "type": "geospatial_code",
          "language": "Python",
          "description": "...",
          "code": "...",
          "note": "<≤100 chars>",
          "sources": ["Store:DocIdOrTitle", "..."]  // optional but recommended
        }}
      ],
      "output": "..."
    }}
    No prose outside JSON. No unregistered tools. If none fit, return:
    {{
      "status":"no_valid_tool",
      "reason":"<short>",
      "sources":["Store:Doc"]
    }}

    2) theory → concise bullets grounded in retrieved sources; cite store+doc id/title.
    3) code → minimal runnable snippet(s) + short comments; cite APIs/sources.
    4) mixed → (A) workflow JSON (with note per step); (B) '---'; (C) 4–6 theory bullets; (D) '---'; (E) tiny code.

    Rules:
    - Prefer Solution for workflows, Literature for theory, Code for code.
    - Be grounded: do NOT invent tools/params or code beyond retrieved content and the registry.
    - If a required parameter is missing, propose a SAFE DEFAULT and mark it explicitly as "(default)" in note.
    - If response_mode='auto', infer:
      • best-way/tools/how-to-run → workflow
      • why/what/definition/compare → theory
      • "example code"/"sample script" → code
      • mixed needs → mixed.

    Locale: {locale}
    Need citations: {need_citations}
    Current mode: {mode}
    """)

    messages = state["messages"]
    prompt_template = ChatPromptTemplate.from_messages([system_prompt_text] + messages)
    formatted_prompt = prompt_template.format_prompt()

    # Choose base model; keep your originals for flexibility
    llm_gpt = init_chat_model("openai:gpt-4.1-mini", max_retries=3, temperature=0)
    # Optional alternates:
    # llm_qwen = ChatOpenAI(api_key=os.getenv("DASHSCOPE_API_KEY"),
    #                       base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    #                       model="qwen-max")
    # llm_claude = ChatAnthropic(model="claude-3-5-sonnet-latest")

    model = llm_gpt.bind_tools(TOOLS)
    response = model.invoke(formatted_prompt)

    # Return new message plus keep mode info in the state
    return {"messages": [response], "response_mode": mode, "need_citations": need_citations, "locale": locale}


# ----------------------------------------------------------------
# Rewrite node (clarify/upgrade the question if retrieval is weak)
# ----------------------------------------------------------------



# ----------------------------------------------------------------
# Graph assembly
# ----------------------------------------------------------------

workflow = StateGraph(State)
workflow.add_node("agent", agent)
workflow.add_node("tools", ToolNode(tools=TOOLS))
# workflow.add_node("generate", generate)

workflow.add_edge(START, "agent")

def select_next_node(state: State):
    return tools_condition(state)

workflow.add_conditional_edges("agent", select_next_node,
                               {"tools": "tools", "__end__": "__end__"})

workflow.add_edge("tools", "agent")
workflow.add_edge("agent", END)

memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)


# ----------------------------------------------------------------
# Public tool wrapper
# ----------------------------------------------------------------

class NTL_Knowledge_Searcher_Input(BaseModel):
    query: str = Field(..., description="Your question plus brief context.")
    response_mode: str = Field(
        default="auto",
        description="One of: auto|workflow|theory|code|mixed"
    )
    locale: str = Field(default="en", description="Output language preference (e.g., en|zh).")
    need_citations: bool = Field(default=True, description="Whether to include simple store/doc citations.")

def _NTL_Knowledge_Searcher(
    query: str,
    response_mode: str = "auto",
    locale: str = "en",
    need_citations: bool = True,
) -> str:
    events = graph.stream(
        input={
            "messages": [("user", query)],
            "response_mode": response_mode,
            "locale": locale,
            "need_citations": need_citations,
        },
        config={"configurable": {"thread_id": "NTL-RAG-1"}, "recursion_limit": 11},
        stream_mode="values",
    )
    final_answer = ""
    for event in events:
        # Each event includes the latest model/tool message
        try:
            event["messages"][-1].pretty_print()
        except Exception:
            pass
        final_answer = event["messages"][-1].content
    return final_answer

NTL_Knowledge_Base = StructuredTool.from_function(
    func=_NTL_Knowledge_Searcher,
    name="NTL_Knowledge_Base",
    description=(
        "Retrieve grounded knowledge for Nighttime Light (NTL) tasks from three internal stores:\n"
        "- Literature (theory, equations)\n- Solution (workflows, tools, datasets)\n- Code (concise Python/GEE snippets)\n\n"
        "Supports four response modes:\n"
        "• workflow → strict JSON of tool steps\n"
        "• theory   → concise grounded bullets\n"
        "• code     → minimal runnable snippet(s)\n"
        "• mixed    → workflow JSON + bullets + small code\n"
        "Set response_mode='auto' to infer intent from the question."
    ),
    args_schema=NTL_Knowledge_Searcher_Input,
)


# workflow-only (for NTL_Engineer)
# NTL_Knowledge_Base.run({
#     "query": "Assess the impact of the 2025 Myanmar earthquake on nighttime light levels using daily NPP-VIIRS VNP46A2 imagery. Retrieve authoritative event details from official sources (USGS, ReliefWeb, OCHA, WHO, GDACS) to identify the most severely affected provinces. Use GEE python API to composite median values for each period, compute brightness change and percentage change, and summarize results for damage assessment.",    # "query": "请你介绍一下夜间灯光遥感"
#     "response_mode": "auto",
#     "locale": "en",
#     "need_citations": True
# })

