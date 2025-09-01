import os

from langchain.agents import initialize_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool, create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
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
from langchain_community.vectorstores import FAISS

env_vars = {
    "OPENAI_API_KEY": "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",
}
os.environ.update({k: os.environ.get(k) or v for k, v in env_vars.items()})

# Initialize the embeddings
embeddings = OpenAIEmbeddings()

# 文献知识库
literature_vector_store = FAISS.load_local(
    r"C:\NTL-CHAT\NTL-GPT\RAG_Faiss\Literature_RAG",
    embeddings,
    allow_dangerous_deserialization=True,  # 读取本地 pkl 所需
)

NTL_Literature_Knowledge = create_retriever_tool(
    literature_vector_store.as_retriever(k=3,score_threshold=0.5),
    name="NTL_Literature_Knowledge",
    description="""
    Use this tool to retrieve peer-reviewed academic literature related to Nighttime Light (NTL) remote sensing.
    Includes 90+ papers focused on urban structure, socio-economic modeling, disaster assessment, light-type analysis, etc.
    Best for understanding theoretical foundations, indicator definitions, and research design patterns.
    Does NOT provide operational workflows or executable code.
    """
)

# 解决方案知识库（workflow、数据源、GEE使用、操作指南）
solution_vector_store = FAISS.load_local(
    r"C:\NTL-CHAT\NTL-GPT\RAG_Faiss\Solution_RAG",
    embeddings,
    allow_dangerous_deserialization=True,  # 读取本地 pkl 所需
)

NTL_Solution_Knowledge = create_retriever_tool(
    solution_vector_store.as_retriever(k=2,score_threshold=0.5),
    name="NTL_Solution_Knowledge",
    description="""
    Use this tool to retrieve structured workflows, tool usage guides, dataset access instructions, and end-to-end NTL application solutions.
    Ideal for finding:
    - How to download and preprocess specific NTL imagery (e.g., VIIRS, SDGSAT-1),
    - How to build analysis pipelines for classification, time-series, and urban growth,
    - Proper tool/parameter usage within geospatial tasks.
    Does NOT contain detailed code implementations.
    """
)

# 代码知识库（Python/GEE 代码片段和模块范例）
code_vector_store = FAISS.load_local(
    r"C:\NTL-CHAT\NTL-GPT\RAG_Faiss\Code_RAG",
    embeddings,
    allow_dangerous_deserialization=True,  # 读取本地 pkl 所需
)

NTL_Code_Knowledge = create_retriever_tool(
    code_vector_store.as_retriever(),
    name="NTL_Code_Knowledge",
    description="""
    Use this tool to retrieve Python and GEE code snippets relevant to NTL tasks.
    Includes raster processing (e.g., Rasterio, numpy), vector operations (GeoPandas, Shapely), GEE scripts (JavaScript/Python),
    and modeling implementations (e.g., regression, clustering, zonal stats).
    Focused on executable logic, NOT theoretical explanation or tool descriptions.
    """
)

