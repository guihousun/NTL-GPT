from langchain.tools import StructuredTool
from langchain_experimental.utilities import PythonREPL
from pydantic import BaseModel, Field
from typing import Optional
import os

# Initialize Python REPL for executing the agent-generated code
python_repl = PythonREPL()

# Define the input schema for the report generation tool
class ReportGenerationInput(BaseModel):
    generated_code: str = Field(
        ...,
        description=(
            "Python code that generates the report. This code should include all necessary imports, "
            "data definitions, template rendering, and file operations to produce the report as per the user's requirements. "
            "Ensure the code is safe and follows best practices."
        )
    )
    save_path: Optional[str] = Field(
        None,
        description=(
            "The file path where the generated report should be saved. "
            "If not specified, the default path 'C:/NTL_Agent/report/report.html' will be used."
        )
    )
    convert_to_pdf: bool = Field(
        default=False,
        description="Whether to convert the generated HTML report to PDF format."
    )

# Define the report generation function
def report_generation_func(
    generated_code: str,
    save_path: Optional[str] = None,
    convert_to_pdf: bool = False
) -> str:
    """
    Executes the provided generated_code to generate a report.

    Parameters:
    - generated_code (str): Python code that generates the report.
    - save_path (Optional[str]): Path to save the report file.
    - convert_to_pdf (bool): Whether to convert the HTML report to PDF.

    Returns:
    - str: Output from the code execution, including any success or error messages.
    """
    # Set default save path
    save_path = save_path or "C:/NTL_Agent/report/report.html"

    # Prepare the code to execute, including save_path and convert_to_pdf
    full_code = f"""
save_path = r'''{save_path}'''
convert_to_pdf = {convert_to_pdf}
{generated_code}

# Check whether the report was generated successfully
import os
try:
    if os.path.exists(save_path):
        print(f"Report generated successfully at {{save_path}}")
        if convert_to_pdf:
            try:
                from weasyprint import HTML
                pdf_path = save_path.replace('.html', '.pdf')
                HTML(save_path).write_pdf(pdf_path)
                if os.path.exists(pdf_path):
                    print(f"PDF generated successfully at {{pdf_path}}")
                else:
                    print("PDF conversion failed.")
            except Exception as e:
                print(f"PDF conversion failed: {{e}}")
    else:
        print("Failed to generate report. Please check the code and try again.")
except Exception as e:
    print(f"An error occurred: {{e}}")
"""

    # Execute the code
    try:
        # Execute the code to generate the report
        result = python_repl.run(full_code)
        return result
    except Exception as e:
        return f"An error occurred while executing the code: {e}"


report_generation_tool = StructuredTool.from_function(
    report_generation_func,
    name="report_generation_tool",
    description=(
        "通过执行提供的Python代码 (`generated_code`) 使用Jinja2生成报告。 "
        "确保代码包含所有必要元素：数据定义、Jinja2模板和文件操作。 "
        "使用相对路径以兼容不同操作系统，尤其是图片路径应位于HTML文件夹内或使用相对引用。\n\n"

        "Jinja2模板应包含带有变量占位符的HTML结构（例如 `{{ title }}`）。报告生成步骤如下：\n\n"

        "1. **数据准备**: 在 `generated_code` 中定义数据字典和报告元素。\n"
        "2. **模板设计**: 创建包含必要变量的Jinja2 HTML模板。\n"
        "3. **渲染模板**: 使用 `template.render()` 将数据填充到模板中。\n"
        "4. **保存文件**: 将渲染后的HTML写入 `save_path`（默认 `C:/NTL_Agent/report/report.html`）。\n"
        "5. **转换为PDF**（可选）: 如果 `convert_to_pdf=True`，使用 `WeasyPrint` 将HTML转换为PDF。\n\n"

        "### 示例参数:\n"
        "- **generated_code**: 包含HTML结构、数据定义、模板渲染、错误处理和输出操作的完整代码。\n"
        "- **save_path**: 报告的文件路径，默认为 `C:/NTL_Agent/report/report.html`。\n"
        "- **convert_to_pdf**: 决定是否将HTML转换为PDF的标志。\n\n"

        "返回存储的报告位置或相关错误信息。"
    ),
    input_type=ReportGenerationInput,
)


