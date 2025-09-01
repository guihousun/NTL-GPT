import h5py
from langchain.chat_models import init_chat_model
from NTL_Knowledge_Base_Searcher import NTL_Knowledge_Base
from NTL_Composite import NTL_composite_local_tool, NTL_composite_GEE_tool
from NTL_preprocess import SDGSAT1_strip_removal_tool, SDGSAT1_radiometric_calibration_tool, VNP46A2_angular_correction_tool, noaa20_sdr_preprocess_tool, dmsp_preprocess_tool
from NTL_classification import classify_light_types_from_rrli_rbli
from SDGSAT1_INDEX import SDGSAT1_index_tool
from NPP_viirs_index_tool import bvntl_index_tool, vnci_index_tool
from built_up_area_extract import urban_extraction_tool
from NTL_raster_stats import NTL_raster_statistics
from NTL_trend_detection_tool import NTL_trend_analysis_tool
from main_road import otsu_road_extraction_tool
from NTL_anomaly_detection_tool import simple_ntl_anomaly_detection_tool
from langchain_core.messages import SystemMessage


tools = [
    NTL_Knowledge_Base, SDGSAT1_strip_removal_tool, urban_extraction_tool, NTL_trend_analysis_tool,
    noaa20_sdr_preprocess_tool, SDGSAT1_radiometric_calibration_tool, classify_light_types_from_rrli_rbli, NTL_raster_statistics,
    VNP46A2_angular_correction_tool, SDGSAT1_index_tool, bvntl_index_tool, vnci_index_tool, dmsp_preprocess_tool,
    otsu_road_extraction_tool, NTL_composite_local_tool, NTL_composite_GEE_tool, simple_ntl_anomaly_detection_tool]

system_prompt_text = SystemMessage("""
You are the NTL Engineer, a supervisor agent responsible for coordinating nighttime light (NTL) remote sensing tasks in the NTL-GPT multi-agent system.

You manage two specialized agents:
- **Data_Searcher**: Handles all data retrieval tasks, including acquiring NTL imagery, population rasters, POIs, NDVI, land cover, administrative boundaries, and socio-economic indicators from platforms such as GEE, OSM, Amap, and BigQuery.
- **Code_Assistant**: Handles all geospatial code validation and execution. In addition to running local Python geospatial libraries (e.g., rasterio, geopandas, shapely, numpy), it can also run Google Earth Engine (GEE) Python API code for cloud-based geospatial computation and data processing. You must submit all generated geospatial Python code to Code_Assistant for thorough testing before execution.

You also have access to:
- **NTL_Knowledge_Base**: A domain-specific tool for retrieving workflows, indices, methods, definitions, and best practices in NTL research.

Your responsibilities include:
- Designing the analysis plan using modular, step-by-step reasoning (Chain-of-Thought).
- Generating Python geospatial code in small modular blocks (avoid function definitions).
- Always begin each task by querying the NTL_Knowledge_Base to ground your plan in domain knowledge.

### Interacting with Code_Assistant

1. Submit all geospatial code (origin code) and all required analysis files with their complete absolute file paths to Code_Assistant for Geo-CodeCoT validation.
2. Code_Assistant will automatically decompose the origin code into logical blocks (e.g., raster loading, field checking, zonal statistics), and test them using mini test inputs.
3. You will receive a structured validation report:
    - If the report indicates "status: fail", you must revise and regenerate the code based on the reported error type, traceback, and Code_Assistant's suggested fixes.

Rules:
- Assign work to only one agent at a time. Do not call agents in parallel.
- Always query the NTL_Knowledge_Base before beginning a task.
- Use sequential agent coordination to complete tasks accurately and efficiently.
- Avoid assumptions—always verify the spatial scope, temporal coverage, and data format of every dataset before using it.
""")






# Initialize language model and bind tools
# llm_GPT = init_chat_model("openai:gpt-5-mini", max_retries=3, temperature = 0)
#
# # llm_GPT = ChatOpenAI(model="gpt-5-mini")
# llm_qwen = ChatOpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#     model="qwen-max",
#     max_retries=3
# )
# llm_claude = ChatAnthropic(model="claude-3-5-sonnet-latest", temperature=0, max_retries=3)

# memory = MemorySaver()
# graph = create_react_agent(llm_GPT, tools=tools, state_modifier=system_prompt_text, checkpointer=memory)

# from langgraph_supervisor import create_supervisor
# for tool in tools:
#     if tool.__doc__ is None:
#         print(f"Tool function {tool.__name__} is missing a docstring!")

# NTL_Engineer = create_supervisor(
#     model=llm_GPT,
#     agents=[Data_Searcher, Code_Assistant],
#     prompt=system_prompt_text,
#     add_handoff_back_messages=True,
#     output_mode="last_message",
#     tools = tools,  # 传入记忆持久化器
#     supervisor_name= "NTL_Engineer"
# ).compile(checkpointer=MemorySaver(),name = "NTL-GPT")

# from IPython.display import display, Image
#
# display(Image(NTL_Engineer.get_graph().draw_mermaid_png()))

# graph = create_react_agent(llm_GPT, tools=tools, state_modifier=system_prompt_text, checkpointer=memory)
# graph = create_react_agent(llm_claude, tools=tools, checkpointer=memory)
# from langchain_core.messages import convert_to_messages
#

#
# def stream_graph_updates(user_input):
#     events = NTL_Engineer.stream(
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
# hello,please always reply in English and tell me your action plan after you finish your task
# continue,don't ask me again until you finish
# Could you please tell me the total number of subdivided steps that were performed?



# import json
# import os
# import inspect
#
# # 兼容不同版本的 LangChain 包路径
# try:
#     from langchain_core.tools import BaseTool
# except Exception:
#     try:
#         from langchain.tools.base import BaseTool
#     except Exception:
#         BaseTool = None  # 老版本兜底
#
# from langchain.tools import StructuredTool
# from pydantic import BaseModel
# #
# def _extract_params_schema(args_schema_cls):
#     """
#     兼容 pydantic v1/v2：优先用 v2 的 model_json_schema()，否则回退到 v1 的 schema()。
#     如果没有 args_schema，则返回空 dict。
#     """
#     if not args_schema_cls:
#         return {}
#     try:
#         # pydantic v2
#         return args_schema_cls.model_json_schema()
#     except Exception:
#         try:
#             # pydantic v1
#             return args_schema_cls.schema()
#         except Exception:
#             return {}
#
# def _normalize_tool(tool):
#     """
#     确保返回 StructuredTool 或 BaseTool：
#     - 若本来就是 BaseTool/StructuredTool，原样返回
#     - 若是可调用函数，包装成 StructuredTool
#     - 其他类型则抛错
#     """
#     if BaseTool is not None and isinstance(tool, BaseTool):
#         return tool
#     if isinstance(tool, StructuredTool):
#         return tool
#     if callable(tool):
#         # 尝试从函数 docstring 提取简要描述
#         desc = (tool.__doc__ or "").strip()
#         try:
#             return StructuredTool.from_function(tool, description=desc or None)
#         except TypeError:
#             # 某些旧版需要只传函数
#             return StructuredTool.from_function(tool)
#     raise TypeError(f"Unsupported tool type: {type(tool)}")
#
# #
# def tools_to_json(tools, save_path):
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#
#     # 先规范化（这里也能顺便帮你定位谁是“裸函数”）
#     normalized = []
#     for idx, t in enumerate(tools):
#         try:
#             nt = _normalize_tool(t)
#             normalized.append(nt)
#         except Exception as e:
#             # 带上索引和类型，方便你快速定位异常项
#             raise RuntimeError(f"工具列表第 {idx} 项无法规范化，类型为 {type(t)}，错误：{e}")
#
#     tool_list = []
#     for t in normalized:
#         # 名称：StructuredTool 一定有 .name
#         name = getattr(t, "name", None) or getattr(t, "func", None).__name__
#         # 描述：优先 StructuredTool 的 description，再退回函数 docstring
#         desc = (getattr(t, "description", None) or "").strip()
#         if not desc and hasattr(t, "func") and t.func.__doc__:
#             desc = t.func.__doc__.strip()
#
#         # 参数 schema：来自 args_schema（可能为 None）
#         args_schema_cls = getattr(t, "args_schema", None)
#         params_schema = _extract_params_schema(args_schema_cls)
#
#         # 自定义分类（如果你给工具挂了 .category）
#         category = getattr(t, "category", None)
#
#         tool_list.append({
#             "tool_name": name,
#             "description": desc,
#             "parameters": params_schema,
#             "category": category
#         })
#
#     with open(save_path, "w", encoding="utf-8") as f:
#         json.dump(tool_list, f, ensure_ascii=False, indent=2)
#
#     print(f"工具信息已保存到: {save_path}")
#
# # ==== 用法 ====
# save_file_path = r"E:\NTL_Agent\workflow\tools.json"
# tools_to_json(tools, save_file_path)
