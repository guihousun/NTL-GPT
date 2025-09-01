import base64
import pdfkit  # pip install pdfkit；还需安装 wkhtmltopdf
# 或者用 weasyprint: pip install weasyprint
import streamlit as st

def build_session_html():
    # 1) 取你已有的数据
    title = "NTL-GPT: Nighttime Light Analysis – Session Report"
    chat = st.session_state.get("chat_history", [])
    thread_id = st.session_state.get("thread_id", "")

    # 2) 简单 HTML（可以替换成你现在页面的样式）
    rows = []
    for who, text in chat:
        who_cls = "user" if who == "user" else "bot"
        rows.append(f"""
          <div class="msg {who_cls}">
            <div class="role">{who.capitalize()}</div>
            <div class="text">{text}</div>
          </div>
        """)

    html = f"""
    <!doctype html>
    <html><head>
      <meta charset="utf-8">
      <title>{title}</title>
      <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; 
                margin: 24px; color:#1f2937; }}
        h1 {{ font-size: 22px; margin: 0 0 12px; }}
        .meta {{ color:#6b7280; margin-bottom:16px; }}
        .msg {{ border:1px solid #e5e7eb; border-radius:10px; padding:12px; margin:10px 0; }}
        .msg.user {{ background:#f3f4f6; }}
        .msg.bot  {{ background:#eef2ff; }}
        .role {{ font-weight:600; color:#374151; margin-bottom:4px; }}
        .footer {{ text-align:center; color:#9ca3af; margin-top:24px; font-size:12px; }}
        pre, code {{ white-space:pre-wrap; word-wrap:break-word; }}
      </style>
    </head>
    <body>
      <h1>{title}</h1>
      <div class="meta">Thread ID: {thread_id}</div>
      {''.join(rows)}
      <div class="footer">NTL-GPT can make mistakes. Check important info.</div>
    </body></html>
    """
    return html


