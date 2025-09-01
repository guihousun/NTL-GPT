import h5py
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
# 这些 import 要根据你的项目实际结构调整：
from langgraph_supervisor import create_supervisor
import pandas as pd
from session_manager import init_session_state, reset_session, save_history, load_history, export_history
import json
# 遍历当前事件中的 messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, FunctionMessage
import os
import streamlit as st
from PIL import Image
from NTL_Engineer import system_prompt_text, tools  # 用你的真实模块替换
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


# 1. 页面配置部分
st.set_page_config(page_title="NTL-GPT: A Collaborative Multi-Agent System for Automating Nighttime Light Remote Sensing Tasks", page_icon=":robot:", layout="wide")
# """
# 设置Streamlit应用的页面配置。这里设置了页面标题、页面图标和布局。
# page_title是浏览器标签页上的标题，page_icon是页面图标，layout="wide"是设置页面为宽布局，适用于较大的屏幕。
# """

# 2. CSS样式定义
# """
# CSS样式：自定义了聊天界面的样式，包括用户和机器人消息的背景颜色、头像的样式等。
# 通过这种方式，可以定制聊天界面的外观，使其更加美观和符合应用主题。
# """

st.markdown(
    """<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.stDeployButton {
            visibility: hidden;
        }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.block-container {
    padding: 2rem 4rem 2rem 4rem;
}

.st-emotion-cache-16txtl3 {
    padding: 3rem 1.5rem;
}
</style>
# """,
    unsafe_allow_html=True,
)

# 3. 聊天消息模板
# """
# 聊天消息模板：定义了聊天消息的HTML模板，bot_template和user_template分别表示机器人和用户的消息。
# 通过{{MSG}}占位符，动态替换成实际的消息内容。每个消息框有一个头像和内容部分，头像是通过URL加载图片。
# """
bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn.icon-icons.com/icons2/1371/PNG/512/robot02_90810.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message" style="font-size:18px;">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.shareicon.net/data/512x512/2015/09/18/103160_man_512x512.png" >
    </div>
    <div class="message" style="font-size:18px;">{{MSG}}</div>
</div>
"""

st.markdown("""
<style>
.stExpander { border:1px solid #e8edf5; border-radius:8px; }
.stCode { border-radius:8px; }
</style>
""", unsafe_allow_html=True)


import json
import re
import streamlit as st

def _extract_first_json(text: str):
    """从字符串里提取第一个 JSON 对象（最外层 {...}）"""
    if not isinstance(text, str):
        return None, text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, text
    maybe = text[start:end+1]
    try:
        obj = json.loads(maybe)
        # 剩余正文：去掉前后的多余空白
        rest = (text[:start] + text[end+1:]).strip()
        return obj, rest
    except Exception:
        return None, text

def render_data_searcher_output(raw_content):
    """
    漂亮地渲染 Data_Searcher 的输出：
    - 卡片显示数据源、时间、空间范围、分辨率
    - 列表显示输出文件
    - 存储路径 / 备注
    - 保留 Raw JSON 折叠
    """
    data, rest = (None, raw_content)
    # 1) 若是纯 dict 或者字符串里有 JSON，就解析
    if isinstance(raw_content, dict):
        data, rest = raw_content, ""
    elif isinstance(raw_content, str):
        data, rest = _extract_first_json(raw_content)

    # 2) 有结构化 JSON 就按块展示
    if isinstance(data, dict):
        data_source = data.get("data_source", "")
        temporal = data.get("temporal_coverage", "")
        spatial = data.get("spatial_coverage", "")
        resolution = data.get("spatial_resolution", "")
        files = data.get("output_files_name") or data.get("output_files") or []
        storage = data.get("storage_location", "")
        notes = data.get("notes", "")

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Data source**")
            st.write(data_source or "-")
            st.markdown("**Temporal coverage**")
            st.write(temporal or "-")
        with cols[1]:
            st.markdown("**Spatial coverage**")
            st.write(spatial or "-")
            st.markdown("**Spatial resolution**")
            st.write(resolution or "-")

        if files:
            st.markdown("**Output files**")
            for p in files:
                st.markdown(f"- `{p}`")

        if storage:
            st.markdown("**Storage location**")
            st.code(storage)

        if notes:
            st.markdown("**Notes**")
            st.write(notes)

        with st.popover("Raw JSON"):
            st.json(data)

    # 3) 仍然保留后面的自然语言说明
    if isinstance(rest, str) and rest.strip():
        st.write(rest)

import json
import textwrap
import streamlit as st

def render_kb_output(kb_content):
    """漂亮地渲染 NTL_Knowledge_Base 的输出"""
    # 1) 解析成 dict
    data = kb_content
    if isinstance(kb_content, str):
        try:
            data = json.loads(kb_content)
        except Exception:
            # 不是合法 JSON 时，至少做个格式化展示
            st.code(kb_content, language="json")
            return

    # 2) 头部信息
    task = data.get("task_name") or data.get("task_id") or "Knowledge Base Task"
    category = data.get("category", "")
    st.markdown(
        f"""
        <div style="border:1px solid #e3e7ef;border-radius:8px;padding:14px;background:#f9fbff">
          <div style="font-size:18px;font-weight:700;color:#2b4a8b;">task: {task}</div>
          <div style="margin-top:4px;color:#5f6b7a;">category: {category}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3) 描述
    desc = data.get("description")
    if desc:
        st.markdown(f"**Description**")
        st.write(desc)

    # 4) 步骤
    steps = data.get("steps", [])
    if steps:
        st.markdown("**Steps**")
        for i, step in enumerate(steps, 1):
            typ = step.get("type", "")
            name = step.get("name", "")
            note = step.get("note", "")
            st.markdown(f"- **{i}. {name or typ}**", unsafe_allow_html=True)

            # 关键输入参数（简要显示，可展开看完整）
            if "input" in step and step["input"]:
                with st.popover(note or step.get("description", f"Inputs for step {i}")):
                    st.json(step["input"])

            # 代码块专门高亮
            if typ in ("geospatial_code", "code") or "Code_description" in step:
                code = step.get("Code_description") or step.get("code") or ""
                if code:
                    st.markdown("**Code Code_description**")
                    st.code(code, language="python")

    # 5) 输出说明
    output = data.get("output")
    if output:
        st.markdown("**Output**")
        st.write(output)

    # 6) 原始 JSON（折叠）
    with st.popover("Raw JSON"):
        st.json(data)


REQUIRED_KEYS = [
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "LANGCHAIN_API_KEY",
    "PROJECT_ID",   # 新增 GEE Project ID
]

MODEL_OPTIONS = ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-4.1"]

# 初始化状态容器
st.session_state.setdefault("cfg_api_keys", {})
st.session_state.setdefault("cfg_model", MODEL_OPTIONS[0])
st.session_state.setdefault("cfg_temp", 0.0)
st.session_state.setdefault("initialized", False)

with st.sidebar:
    st.subheader("🔑 API Keys (Required)")
    st.markdown(
        "<p style='text-align: center; font-size: 10px; color: gray;'>请输入所需的 API Keys</p>",
        unsafe_allow_html=True
    )
    for k in REQUIRED_KEYS:
        default = st.session_state["cfg_api_keys"].get(k, "")
        # project_id 明文输入，其它 key 用密码框
        if k == "PROJECT_ID":
            val = st.text_input(f"{k}", value=default)
        else:
            val = st.text_input(f"{k}", value=default, type="password")
        if val:
            st.session_state["cfg_api_keys"][k] = val

    st.subheader("🧠 LLM Settings")
    st.session_state["cfg_model"] = st.selectbox(
        "Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(st.session_state["cfg_model"])
    )
    st.session_state["cfg_temp"] = st.slider("Temperature", 0.0, 1.0, st.session_state["cfg_temp"], 0.05)

    # 确认 & 重置按钮
    cols = st.columns(2)
    with cols[0]:
        if st.button("✅ Initialize", use_container_width=True):
            missing = [k for k in REQUIRED_KEYS if not st.session_state["cfg_api_keys"].get(k)]
            if missing:
                st.warning(f"Missing keys: {', '.join(missing)}")
            else:
                # 默认配置
                os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                # 用户输入的 Key
                for k, v in st.session_state["cfg_api_keys"].items():
                    os.environ[k] = v
                st.session_state["initialized"] = True
                st.success("Initialized. You can start chatting.")
                st.rerun()

    with cols[1]:
        if st.button("♻️ Reconfigure", use_container_width=True):
            st.session_state["initialized"] = False
            st.info("Reconfigure and click Initialize again.")
            st.rerun()




def build_conversation():
    from NTL_Code_Assistant import Code_Assistant_system_prompt_text, Code_tools
    from NTL_Data_Searcher import system_prompt_data_searcher, data_searcher_tools
    """用侧栏选择的模型构建/重建对话Agent"""
    llm = init_chat_model(
        f"openai:{st.session_state['cfg_model']}",
        temperature=st.session_state["cfg_temp"],
        max_retries=3,
    )
    memory1 = MemorySaver()
    Code_Assistant = create_react_agent(llm, tools=Code_tools, prompt=Code_Assistant_system_prompt_text, checkpointer=memory1, name="Code_Assistant",)

    memory2 = MemorySaver()
    Data_Searcher = create_react_agent(llm, tools=data_searcher_tools, prompt = system_prompt_data_searcher, name="Data_Searcher",checkpointer=memory2)


    # 这里用你现有的 system_prompt_text 与 tools
    graph = create_supervisor(
        model=llm,
        agents=[Data_Searcher, Code_Assistant],
        prompt=system_prompt_text,
        add_handoff_back_messages=True,
        output_mode="full_history",
        tools=tools,
        supervisor_name="NTL_Engineer",
    ).compile(checkpointer=MemorySaver(), name="NTL-GPT")
    return graph

def ensure_conversation_initialized():
    """只有用户确认后才初始化，模型变更时重建"""
    if not st.session_state["initialized"]:
        return  # 未确认，不创建
    cur_model = st.session_state.get("cur_model_name")
    if ("conversation" not in st.session_state) or (cur_model != st.session_state["cfg_model"]):
        st.session_state.conversation = build_conversation()
        st.session_state.cur_model_name = st.session_state["cfg_model"]


def handle_userinput(user_question):
    chat_history = st.session_state.chat_history
    state = {
        "messages": [{"role": "user", "content": user_question}],
    }
    RECURSION_LIMIT = 2*15 + 1

    # 更新聊天记录，加入用户问题和AI的最新回复
    st.session_state.chat_history.append(("user", user_question))
    # 显示用户问题
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)

    # 创建一个占位符，等待答案
    wait_placeholder = st.empty()  # Create a placeholder for the "please wait" message
    wait_placeholder.write(bot_template.replace("{{MSG}}", "Please Wait"), unsafe_allow_html=True)

    # 初始化一个变量来存储最终的答案
    final_answer = None
    selected_images = []  # List to hold selected images
    image_found_Act = False
    CSV_found_Act = False
    # 用于流式输出的显示区

    # 每轮唯一 ID
    st.session_state.setdefault("run_counter", 0)
    st.session_state.run_counter += 1
    run_id = st.session_state.run_counter

    # 每轮展开标志（默认展开）
    expander_flag_key = f"run_expanded_{run_id}"
    if expander_flag_key not in st.session_state:
        st.session_state[expander_flag_key] = True

    with st.expander("#### Analysis Process", expanded=True):
        events = st.session_state.conversation.stream(
            state,
            config={
                "configurable": {"thread_id": st.session_state.thread_id},
                "recursion_limit": RECURSION_LIMIT
            },
            stream_mode="values"
        )

        final_answer = None
        previous_messages_len = 0

        for i, event in enumerate(events):
            st.markdown(f"""
            <div style="border: 1px solid #ccc; border-radius: 4px; padding: 10px; margin: 12px 0; background-color: #f8f9fa;">
              <div style="display: flex; align-items: center; gap: 6px;">
                <span style="font-size: 18px; line-height: 1;">🔎</span>
                <span style="color: #4a6fa5; font-size: 18px; font-weight: 600;">Raw Event {i + 1}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            new_messages = event["messages"][previous_messages_len:]
            previous_messages_len = len(event["messages"])

            for msg in new_messages:
                if isinstance(msg, HumanMessage):
                    st.markdown(f"<div style='color:#d6336c;font-weight:bold;font-size:18px;'>🧍 Human:</div>",
                                unsafe_allow_html=True)
                    st.markdown(f"<div style='margin-left:15px;font-size:16px;'>{msg.content}</div>",
                                unsafe_allow_html=True)


                elif isinstance(msg, AIMessage):

                    agent_name = (msg.name or "AI")

                    # ✅ 对 Data_Searcher 做美化渲染

                    if agent_name.lower() == "data_searcher":

                        st.markdown(

                            f"<div style='color:#5cb85c;font-weight:bold;font-size:18px;'>🤖 {agent_name}:</div>",

                            unsafe_allow_html=True

                        )

                        # 尝试美化

                        try:

                            render_data_searcher_output(msg.content)

                        except Exception:

                            # 兜底：按原样输出

                            st.markdown(f"<div style='margin-left:15px; font-size:16px;'>{msg.content}</div>",
                                        unsafe_allow_html=True)

                    elif agent_name.lower() == "code_assistant":
                        # ✅ Code_Assistant 用 python 高亮
                        st.markdown(
                            f"<div style='color:#007bff;font-weight:bold;font-size:18px;'>🤖 {agent_name}:</div>",
                            unsafe_allow_html=True)
                        st.code(msg.content, language="python")

                    else:

                        # 其它 Agent（比如 NTL_Engineer / Code_Assistant）仍按你原来的方式

                        st.markdown(

                            f"<div style='color:#5cb85c;font-weight:bold;font-size:18px;'>🤖 {agent_name}:</div>",

                            unsafe_allow_html=True

                        )

                        st.markdown(f"<div style='margin-left:15px;font-size:16px;'>{msg.content}</div>",
                                    unsafe_allow_html=True)



                elif isinstance(msg, ToolMessage):

                    st.markdown(

                        f"<div style='color:#0275d8;font-weight:bold;font-size:18px;'>🛠 Tool ({msg.name}) Output:</div>",

                        unsafe_allow_html=True)

                    # 针对 NTL_Knowledge_Base 美化输出

                    if msg.name and "NTL_Knowledge_Base" in msg.name:

                        render_kb_output(msg.content)

                    else:

                        # 其它工具保持原样（或做简单 prettify）

                        try:

                            parsed = json.loads(msg.content) if isinstance(msg.content, str) else msg.content

                            st.json(parsed)

                        except Exception:

                            st.write(msg.content)


                elif isinstance(msg, FunctionMessage):
                    tool_call = msg.additional_kwargs.get("tool_calls", [{}])[0]
                    tool_name = tool_call.get("name", "Unknown")
                    args = tool_call.get("function", {}).get("arguments", {})
                    st.markdown(
                        f"<div style='color:#f0ad4e;font-weight:bold;font-size:18px;'>📡 Function Call to `{tool_name}` with arguments:</div>",
                        unsafe_allow_html=True)
                    st.json(json.loads(args) if isinstance(args, str) else args)

                else:
                    st.code(str(msg), language="json")

                st.markdown("<hr style='margin: 15px 0; border: 1px dashed #ccc;'>", unsafe_allow_html=True)

            final_answer = event["messages"][-1].content

            # Check if printed_content contains "geo_visualization_tool"
            if "png" in final_answer:
                image_found_Act = True
            if "csv" in final_answer:
                CSV_found_Act = True


        # After the stream processing ends, check if images need to be displayed
        if image_found_Act:
            image_dir = "C:\\NTL_Agent\\report\\image"
            if os.path.exists(image_dir):
                # Get all image files in the directory
                image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

                if image_files:
                    # Move the image display logic to the sidebar
                    with st.sidebar:
                        # Track the images already displayed in session_state to avoid duplicate display
                        if "displayed_images" not in st.session_state:
                            st.session_state.displayed_images = []

                        # Use checkboxes to select multiple images
                        for idx, image_file in enumerate(image_files):
                            # Skip images that have already been displayed
                            if image_file in st.session_state.displayed_images:
                                continue

                            image_path = os.path.join(image_dir, image_file)
                            image = Image.open(image_path)

                            col1, col2 = st.columns([7, 2])  # Use columns to display image and delete button
                            with col1:
                                st.image(image, caption=f"Image {idx + 1}: {image_file}", use_container_width=True)
                            with col2:
                                if st.button(f"Delete {image_file}", key=f"delete_{idx}_{image_file}"):
                                    os.remove(image_path)  # Delete the image file
                                    st.success(f"Image {image_file} deleted.")
                                    st.experimental_rerun()  # Rerun to refresh the page after deletion

                            # Add the image to the displayed list
                            st.session_state.displayed_images.append(image_file)

        if CSV_found_Act:
            # 定义 CSV 文件目录
            csv_dir = "C:\\NTL_Agent\\report\\csv"
            if os.path.exists(csv_dir):
                # 获取目录中所有的 CSV 文件
                csv_files = [f for f in os.listdir(csv_dir) if f.lower().endswith('.csv')]

                if csv_files:
                    # 在页面上显示找到的 CSV 文件
                    st.subheader("CSV Files Found:")
                    for csv_file in csv_files:
                        csv_path = os.path.join(csv_dir, csv_file)

                        try:
                            # 读取 CSV 文件并显示为表格
                            df = pd.read_csv(csv_path)
                            with st.expander(f"Preview of {csv_file}"):  # 使用 expander 展开显示 CSV 内容
                                st.write(df)

                            # 提供一个下载链接
                            csv_download_link = f'<a href="data:file/csv;base64,{df.to_csv(index=False).encode().decode()}" download="{csv_file}">Download {csv_file}</a>'
                            st.markdown(csv_download_link, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Failed to read {csv_file}: {e}")

        # Display the final answer after all responses and images are processed
        if final_answer:
            wait_placeholder.write(bot_template.replace("{{MSG}}", final_answer), unsafe_allow_html=True)
            st.session_state.chat_history.append(("assistant", final_answer))

            # ✅ 保存会话历史，确保数据持久化
            save_history(st.session_state)


# 8. 显示对话历史
# '''
# show_history：显示聊天历史。它遍历聊天记录并根据消息的索引判断是用户的提问还是机器人的回答。交替显示用户和机器人的消息。
# '''
def show_history():
    chat_history = st.session_state.chat_history

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message[1]), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message[1]), unsafe_allow_html=True)


# 9.主函数
# '''
# main：主函数。设置页面标题并初始化会话状态。
# 通过 Streamlit sidebar 允许用户上传文件，并对上传的文档进行处理，提取文本并存入向量存储。
# 在主容器内，展示聊天输入框和聊天历史。用户输入问题后，如果会话存在，机器人就会根据上下文进行回答。
# '''
@st.cache_data

def main():
    # 初始化会话状态
    init_session_state(st.session_state)

    # 🧷 在一切之前：确保（或跳过）初始化
    ensure_conversation_initialized()
    st.header("**NTL-GPT: A Multi-Agent for Automating Nighttime Light Remote Sensing Tasks**")

    # 如果还没初始化，禁用聊天输入并提示
    if not st.session_state["initialized"]:
        st.info("Please enter API keys, choose model & temperature in the sidebar, then click **Initialize**.")
        # 禁用聊天输入
        user_question = st.chat_input("Please Ask Something~", disabled=True)
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ✅ 添加 Thread ID 初始化
    import uuid
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    with st.sidebar:
        # Image upload section
        uploaded_image = st.file_uploader('Image Display',type=["jpg", "jpeg", "png", "gif"])

        if uploaded_image is not None:
            # Open the image with PIL
            image = Image.open(uploaded_image)
            st.image(image, caption="Image Display", use_container_width=True)

        st.write(f"**Your Thread ID:** `{st.session_state.thread_id}`")

        # 重新生成 Thread ID，清空会话
        if st.button("🔄 New Session"):
            reset_session(st.session_state)
            st.rerun()

        # 恢复指定会话
        custom_thread_id = st.text_input("Enter Existing Thread ID to Restore Session")
        if st.button("Restore Session"):
            if custom_thread_id.strip():
                if load_history(st.session_state, custom_thread_id.strip()):
                    st.success(f"Session restored to: `{custom_thread_id}`")
                    st.rerun()
                else:
                    st.warning("Session not found.")
            else:
                st.warning("Please enter a valid Thread ID.")



    with st.container():
        user_question = st.chat_input("Please Ask Something~")

    with st.container(height=550):
        show_history()  # 显示历史对话
        if user_question:
            handle_userinput(user_question)  # 处理用户输入，流式输出每个阶段

    st.markdown("<br>", unsafe_allow_html=True)  # 空行
    st.markdown(
        "<p style='text-align: center; font-size: 14px; color: gray;'>NTL-GPT can make mistakes. Check important info.</p>",
        unsafe_allow_html=True
    )




if __name__ == "__main__":
    main()

# streamlit run C:\NTL-CHAT\NTL-GPT\Streamlit.py
