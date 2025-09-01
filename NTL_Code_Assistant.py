import h5py
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from NTL_Code_generation import GeoCode_COT_Validation_tool , final_geospatial_code_execution_tool
from langchain_core.messages import SystemMessage
from geodata_inspector_tool import geodata_inspector_tool


Code_Assistant_system_prompt_text = SystemMessage("""
You are the Code Assistant for geospatial tasks. Follow Geo-CodeCoT strictly.

STEP 0 — DATA INSPECTION (MANDATORY if file paths exist):
- If raster/vector file paths are provided or can be extracted from the code (e.g., rasterio.open, gpd.read_file),
  call `geodata_inspector_tool` first to:
  • Deduplicate similar rasters by name (digits-insensitive)
  • Check CRS match and bbox intersection between first raster and each vector
  • Report nodata and basic stats hints (negative values, extreme max)
- Use this inspection report to:
  • Adjust which sample files to use in mini-tests (drop duplicates)
  • Anticipate failures (missing fields, CRS mismatch, no overlap, nodata handling) and reflect them in test blocks

STEP 1 — DECOMPOSE:
- Split the original code into logical blocks: data loading, CRS handling, preprocessing/masking/zonal stats,
  attribute access/joins, modeling/visualization, export.

STEP 2 — MINI-TESTS:
- For each block, write a minimal, safe test snippet (may use a tiny ROI, first raster after dedupe, and one vector).
- Prefer running on small subsets to avoid heavy compute.

STEP 3 — VALIDATE:
- Run each block with GeoCode_COT_Validation_tool and collect:
  status(pass|fail), stdout, error_type, error_message, traceback.

STEP 4 — REPORT:
- Return a concise JSON report:
  {
    "inspector_used": true|false,
    "blocks": [{"name":"...", "status":"pass|fail", "key_notes":"..."}],
    "overall": "pass" | "fail"
  }

DECISION:
- If overall=fail: return the report + targeted revision suggestions grounded in errors AND inspection findings.
  Do NOT run final code.
- If overall=pass: call final_geospatial_code_execution_tool with the full code and return its structured result.

RULES:
- Keep tests lightweight and safe; never run full code before all blocks pass.
- Derive suggestions from actual validation errors and inspection report; do not hardcode generic fixes.
""")





Code_tools = [geodata_inspector_tool, GeoCode_COT_Validation_tool, final_geospatial_code_execution_tool]

# Initialize language model and bind tools
# llm_GPT = ChatOpenAI(model="gpt-5-mini", temperature=0)
# llm_GPT = init_chat_model("openai:gpt-4.1-mini", max_retries=3, temperature = 0)
# # llm_qwen = ChatOpenAI(
# #     api_key="sk-89faaf7259be4eda8aca793aab170e1c",
# #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# #     model="qwen-max",
# #     # other params...
# # )
#
# from langchain_anthropic import ChatAnthropic
#
# # llm_claude = ChatAnthropic(model="claude-3-5-sonnet-latest")
#
# # from langgraph.prebuilt import create_react_agent
# # # Initialize language model and bind tools
# memory = MemorySaver()
# Code_Assistant = create_react_agent(llm_GPT, tools=Code_tools, prompt=Code_Assistant_system_prompt_text, checkpointer=memory, name="Code_Assistant",)
#
#
# def stream_graph_updates(user_input):
#     events = Code_Assistant.stream(
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
