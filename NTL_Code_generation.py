from langchain.tools import StructuredTool
from langchain_experimental.utilities import PythonREPL
# 更新函数，使用解包参数
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
import ee
import geemap
from IPython import InteractiveShell
import io
from contextlib import redirect_stdout
import json
# Define the function to execute geographic data processing commands
import re
from IPython import get_ipython
import rasterio


# def get_fixed_code_from_llm(task_description: str, geo_python_code: str) -> str:
#     # Instantiate the GPT model for code correction
#     code_llm = ChatOpenAI(model="gpt-4o", temperature=0)
#
#     # Prepare the prompt for the model to fix and optimize the code
#     prompt = f"""
#     You are an expert in geographic data processing and Python programming.
#     I have the following task description and Python code:
#
#     Task description:
#     {task_description}
#
#     Python code:
#     {geo_python_code}
#
#     Please detect any errors or inefficiencies in the code, fix the issues, and return ONLY the corrected, optimized, executable version of the code.
#     Do not include any explanations or other text—just return the modified code.
#     """
#
#     # Get the response from the model
#     response = code_llm.invoke([{"role": "system", "content": "You are a helpful code assistant."},
#                            {"role": "user", "content": prompt}])
#
#     # Extract the fixed code from the response
#     fixed_code = response.content
#     return fixed_code
# Define the input schema for geographic data processing tasks




# python_repl = PythonREPL()
# ### Build Python repl ###
# # You can create the tool to pass to an agent
# # repl_tool = Tool(
# #     name="python_repl",
# #     description=(
# #         """
# #         A Python shell. Use this to execute Python commands. Input should be a valid Python command.
# #         To ensure correct path handling, all file paths should be provided as raw strings (e.g., r'C:\\path\\to\\file').
# #         Paths should also be constructed using os.path.join() for cross-platform compatibility.
# #         Additionally, verify the existence of files before attempting to access them.
# #         After executing the command, use the `print` function to output the results.
# #         """
# #
# #     ),
# #     func=python_repl.run,
# # )
# 
# class FinalCodeInput(BaseModel):
#     final_geospatial_code: str = Field(
#         ...,
#         description=(
#             "The geospatial code after Geo-CodeCoT validation."
#         )
#     )
# 
# # Define the function to execute geographic data processing commands
# def final_geospatial_code_execution(final_geospatial_code: str) -> str:
#     try:
#         result = python_repl.run(final_geospatial_code)
#     except BaseException as e:
#         return f"Failed to execute. Error: {repr(e)}"
# 
#     import json
# 
#     return json.dumps({
#         "status": "success",
#         "stdout": result,
#         "Code": final_geospatial_code
#     })
# 
# # final_geospatial_code_execution("import rasterio")
# 
#     # Create the StructuredTool for geographic data processing
# final_geospatial_code_execution_tool = StructuredTool.from_function(
#     final_geospatial_code_execution,
#     name="Geocode_tool_local",
#     description=(
#         """
#         This tool (based on a Python shell) is used to execute the final geospatial code after the Geo-CodeCoT check.
# 
#         **Usage**: 
#         In `geo_python_code`, Use raw strings (e.g., r'C:\\path\\to\\file') and `os.path.join()`
#         Write Python code that handles file existence with `os.path.exists()` and includes error handling.
#         Use: `print(f'Result: {result}')` to return values.
#         The code should be executed in modular steps, avoiding the use of functions.
#         """
#     )
#     ,
#     args_schema=FinalCodeInput,
# )
# 
# import json
# 
# class GeoCodeCOTBlockInput(BaseModel):
#     code_block: str = Field(..., description=(
#         "A minimal, self-contained Python code snippet that tests one specific logic unit "
#         "from the original geospatial code, such as data loading, CRS matching, field access, "
#         "masking, or regression modeling. Should run on small test data."
#     ))
# 
# def GeoCode_COT_Validation(code_block: str) -> str:
#     report = {
#         "status": "pass",
#         "error_type": None,
#         "error_message": None,
#         "traceback": None,
#         "stdout": None,
#         "tested_blocks": code_block
#     }
# 
#     try:
#         result = python_repl.run(code_block)
#         report["stdout"] = result
#     except Exception as e:
#         import traceback
#         report["status"] = "fail"
#         report["error_type"] = type(e).__name__
#         report["error_message"] = str(e)
#         report["traceback"] = traceback.format_exc()
# 
#     return json.dumps(report, indent=2)
# 
# 
# GeoCode_COT_Validation_tool = StructuredTool.from_function(
#     GeoCode_COT_Validation,
#     name="GeoCode_COT_Validation_tool",
#     description=(
#         "This tool performs lightweight Geo-CodeCoT validation on a single geospatial code block. "
#         "Each code block represents a minimal logical component of a full geospatial analysis task, "
#         "such as loading raster or vector data, checking coordinate reference systems (CRS), verifying "
#         "attribute fields (e.g., 'GDP', 'Year'), or computing basic spatial statistics. \n\n"
#         "The tool executes the provided code block within a sandboxed Python shell and returns a "
#         "structured result indicating whether the block passed or failed. If execution fails, the tool "
#         "will include diagnostic error messages to help identify missing field references, CRS mismatches, "
#         "NaN or -999 values, or invalid analysis outputs. \n\n"
#         "This tool is not intended to execute full workflows. Instead, it is used by the Code Assistant "
#         "to pre-validate components of the original geospatial code using test data or small input subsets. "
#         "Once all test blocks pass, the Code Assistant may proceed to full execution with the final code."
#     )
#     ,
#     args_schema=GeoCodeCOTBlockInput
# )

from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from IPython import get_ipython
import io, json, re, traceback, contextlib

class GeoCodeCOTBlockInput(BaseModel):
    code_block: str = Field(
        ...,
        description=("A minimal, self-contained Python code snippet that tests one specific logic unit "
            "from the original geospatial code, such as data loading, CRS matching, field access, "
            "masking, or regression modeling. Should run on small test data."))

def _gee_run_cell(code_block: str):
    ip = get_ipython()
    if ip is None:
        return False, "", "EnvironmentError", "IPython shell is not available.", None

    # 尝试初始化 GEE（已初始化会忽略）
    bootstrap = (
        "import ee\n"
        "project_id = 'empyrean-caster-430308-m2'\n"
        "try:\n"
        "    ee.Initialize(project=project_id)\n"
        "except Exception:\n"
        "    pass\n"
    )

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            # 先跑初始化
            res0 = ip.run_cell(bootstrap, store_history=False)
            if getattr(res0, "error_in_exec", None):
                raise res0.error_in_exec

            # 再跑用户代码
            res = ip.run_cell(code_block, store_history=False)
            if getattr(res, "error_before_exec", None):
                raise res.error_before_exec
            if getattr(res, "error_in_exec", None):
                raise res.error_in_exec

        logs = buf.getvalue()
        # 去掉 ANSI 颜色码（吸收你旧版的优点）
        logs = re.sub(r'\x1b\[[0-9;]*m', '', logs)
        return True, logs, None, None, None

    except Exception as e:
        logs = buf.getvalue()
        logs = re.sub(r'\x1b\[[0-9;]*m', '', logs)
        return False, logs, type(e).__name__, str(e), traceback.format_exc()
    finally:
        buf.close()

def GEE_GeoCode_COT_Validation(code_block: str) -> str:
    ok, logs, etype, emsg, tb = _gee_run_cell(code_block)
    report = {
        "status": "pass" if ok else "fail",
        "stdout": logs,
        "error_type": etype,
        "error_message": emsg,
        "traceback": tb,
        "code_block": code_block
    }
    return json.dumps(report, indent=2)

GeoCode_COT_Validation_tool = StructuredTool.from_function(
    GEE_GeoCode_COT_Validation,
    name="GeoCode_COT_Validation_tool",
    description=(
        "Lightweight Geo-CodeCoT validation for Python. Executes ONE minimal logic block "
        "in an IPython cell. Returns structured JSON: pass/fail, stdout (ANSI stripped), "
        "and detailed error info (type/message/traceback). NOT for full workflows."
        "IMPORTANT: GEE code block **must** start with:\n"
        "    project_id = 'empyrean-caster-430308-m2'\n"
        "    ee.Initialize(project=project_id)\n"
        "to ensure correct GEE project initialization."
    ),
    args_schema=GeoCodeCOTBlockInput
)

class FinalCodeInput(BaseModel):
    final_geospatial_code: str = Field(
        ...,
        description="The final geospatial Python code that has passed all Geo-CodeCoT mini tests."
    )

def final_geospatial_code_execution(final_geospatial_code: str) -> str:
    ok, logs, etype, emsg, tb = _gee_run_cell(final_geospatial_code)
    if ok:
        return json.dumps({"status": "success", "stdout": logs, "code": final_geospatial_code}, indent=2)
    else:
        return json.dumps({
            "status": "fail",
            "stdout": logs,
            "error_type": etype,
            "error_message": emsg,
            "traceback": tb,
            "code": final_geospatial_code
        }, indent=2)

final_geospatial_code_execution_tool = StructuredTool.from_function(
    final_geospatial_code_execution,
    name="final_geospatial_code_execution_tool",
    description=(
        "Execute the **final** geospatial workflow in a Python (IPython) context. "
        "Captures stdout (ANSI stripped). Returns structured JSON with success/failure and traceback. "
        "Use only after all mini tests pass."
        "IMPORTANT: GEE code block **must** start with:\n"
        "    project_id = 'empyrean-caster-430308-m2'\n"
        "    ee.Initialize(project=project_id)\n"
        "to ensure correct GEE project initialization."
    ),
    args_schema=FinalCodeInput
)



# class TestCodeInput(BaseModel):
#     Test_Code: str = Field(
#         ...,
#         description=(
#             ""
#         )
#     )
#
# # Define the function to execute geographic data processing commands
# def GeoCode_COT_Validation(TestCodeInput: str) -> str:
#     try:
#         result = python_repl.run(TestCodeInput)
#     except BaseException as e:
#         return f"Failed to execute. Error: {repr(e)}"
#
#     import json
#
#     return json.dumps({
#         "status": "pass",
#         "stdout": result
#     })
#
#     # Create the StructuredTool for geographic data processing
# GeoCode_COT_Validation_tool = StructuredTool.from_function(
#     GeoCode_COT_Validation,
#     name="GeoCode_COT_Validation_tool",
#     description=(
#         """
#         This tool (based on a Python shell) is check the origin geospatial code by the Geo-CodeCoT framework.
#         Example:
#         """
#     )
#     ,
#     args_schema=TestCodeInput,
# )
    # # Initialize the interactive Python shell
    # shell = get_ipython()
    # if shell is None:
    #     return json.dumps({
    #         "status": "error",
    #         "logs": "",
    #         "error": "Interactive Python shell is not available."
    #     }, indent=2)
    #
    # # Step 1: Print the input code (for logging purposes)
    # # print(f"Task Description: {task_description}")
    # # print(f"Input Python Code:\n{geo_python_code}")
    #
    # # Step 2: Execute the Python code
    # try:
    #     # Capture stdout
    #     captured_output = io.StringIO()
    #     with redirect_stdout(captured_output):
    #         result = shell.run_cell(geo_python_code)
    #
    #     # Get captured stdout content
    #     execution_logs = captured_output.getvalue()
    #
    #     # Remove ANSI escape codes from logs (if present)
    #     execution_logs = re.sub(r'\x1b\[[0-9;]*m', '', execution_logs)
    #
    #     # Check for errors in execution
    #     if hasattr(result, 'error_in_exec') and result.error_in_exec:
    #         return json.dumps({
    #             "status": "error",
    #             "logs": execution_logs,
    #             "error": str(result.error_in_exec)
    #         }, indent=2)
    #
    #     # Return success logs
    #     return json.dumps({
    #         "status": "Run",
    #         "logs": execution_logs
    #     }, indent=2)
    #
    # except Exception as e:
    #     # Handle unexpected exceptions
    #     return json.dumps({
    #         "status": "error",
    #         "logs": captured_output.getvalue() if 'captured_output' in locals() else "",
    #         "error": str(e)
    #     }, indent=2)


class GeoVisualizationInput(BaseModel):
    task_description: str = Field(
        ...,
        description=(
            "A detailed description of the visualization task, including goals, data, and expected outputs. "
        )
    )
    visualization_code: str = Field(
        ...,
        description=(
            "Python code that generates the geographic visualization. "
            "Include imports, data processing, plotting, and saving the figure, style like Nature and Science. "
            "Ensure the code is safe and follows best practices."
        )
    )


# Define the visualization function
# def geo_visualization_func(
#         task_description: str,
#         visualization_code: str
# ) -> str:
#     # print(f"Task Description: {task_description}")
#     # print(f"Visualization Code: {visualization_code}")
#     try:
#         result = python_repl.run(visualization_code)
#     except BaseException as e:
#         return f"Failed to execute. Error: {repr(e)}"
#     result_str = f"Successfully executed\nStdout: {result}"
#     return result_str




project_id = 'empyrean-caster-430308-m2'
ee.Initialize(project=project_id)
# Initialize IPython shell
shell = InteractiveShell().instance()


# Define the input schema for geographic data processing tasks
class GEE_Task(BaseModel):
    task_description: str = Field(
        ...,
        description=(
            "A detailed description of GEE processing task to be performed. "
            "Specify the operation, input file paths, parameters, and expected outputs. ")
    )
    GEE_python_code: str = Field(
        ...,
        description=(
            "GEE Python code generated based on the task description. "
            "The code should be carefully reviewed before execution.\n\n"
            "The code should initialize Earth Engine, process the geographic data, and output the results.\n"
            "It should include the necessary imports, initialization of the Earth Engine project, "
            "and any required error handling."
        )
    )


def Geocode_tool_GEE(task_description: str, GEE_python_code: str) -> str:
    # print(f"Task Description: {task_description}")

    # Initialize the interactive Python shell
    shell = get_ipython()
    if shell is None:
        return json.dumps({
            "status": "error",
            "logs": "",
            "error": "Interactive Python shell is not available."
        }, indent=2)

    # Step 1: Print the input code (for logging purposes)
    # print(f"Input Python Code:\n{GEE_python_code}")

    # Step 2: Execute the Python code
    try:
        # Capture stdout
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            result = shell.run_cell(GEE_python_code)

        # Get captured stdout content
        execution_logs = captured_output.getvalue()

        # Remove ANSI escape codes from logs (if present)
        execution_logs = re.sub(r'\x1b\[[0-9;]*m', '', execution_logs)

        # Check for errors in execution
        if hasattr(result, 'error_in_exec') and result.error_in_exec:
            return json.dumps({
                "status": "error",
                "logs": execution_logs,
                "error": str(result.error_in_exec)
            }, indent=2)

        # Return success logs
        return json.dumps({
            "status": "Run",
            "logs": execution_logs
        }, indent=2)

    except Exception as e:
        # Handle unexpected exceptions
        return json.dumps({
            "status": "error",
            "logs": captured_output.getvalue() if 'captured_output' in locals() else "",
            "error": str(e)
        }, indent=2)



# Create the StructuredTool for geographic data processing
Geocode_tool_GEE = StructuredTool.from_function(
    Geocode_tool_GEE,
    name="Geocode_tool_GEE",
    description=(
        """
        This tool (based on an IPython shell) performs normal and simple GEE processing tasks.

        **Usage**: In `task_description`, provide a clear description with file paths and parameters.

        In `GEE_python_code`,Please initialise first: "ee.Initialize(project='empyrean-caster-430308-m2')".
        Using the image collection from NTL_retrieve_tool, and including error handling.
        Use raw strings (e.g., r'C:\\path\\to\\file') and `os.path.join()` for cross-platform path handling.
        you should use: `print(f'Result: {result}')` to return values.
        The tool captures all standard output (`stdout`) from the task and returns it as `logs` in the result.
        Any errors in the execution will be returned as `error` in the result.
        """

    )
    ,
    input_type=GEE_Task,
)

# Create the StructuredTool with updated description and usage instructions
# visualization_tool = StructuredTool.from_function(
#     geo_visualization_func,
#     name="Visualization_tool",
#     description=(
#         """
#         This tool (based on a Python shell) is specially for geographic data visualization.
#         **Usage**:
#         The `task_description` outlines the related visualization task.
#         The `visualization_code` should include imports, data processing, plotting, and saving the figure, styled like Science. 
#         Default Setting:
#         vmin, vmax = np.percentile(ntl_data, (1, 99))
#         cax = ax.imshow(ntl_data, cmap='cividis', vmin=vmin, vmax=vmax)
#         figsize=(10, 10); title: fontsize=16, fontweight='bold'; x/ylabel: fontsize=15
#         Do not visualize the nodata (-999).
#         Figure must be saved in the folder path `C:/NTL_Agent/report/image` with the English name (e.g., 'NTL Image of Nanjing - June 2020.png'), 300 DPI. 
#         Use raw strings (e.g., r'C:\\path\\to\\file') and `os.path.join()`.
#         Use: `print(f'Full storage address: {save_path}')`.
#         """
# 
#     ),
#     input_type=GeoVisualizationInput,
# )
# # 任务描述和 Python 代码
# task_description = """
# Mask the NTL image for Shanghai using the shapefile and set the outer values to nodata.
# """
#
# geo_python_code = """
# import numpy as np
#
# # Function to calculate the sum of squares of positive numbers
# def sum_of_squares_positive(arr):
#     positive_numbers = arr[arr > 0]  # Filter positive numbers
#     return np.sum(positive_numbers ** 2)  # Return the sum of their squares
#
# # Example array
# data = np.array([-1, 2, 3, -4, 5, -6])
#
# # Call the function and print the result
# result = sum_of_squares_positive(data)
# print(f"The sum of squares of positive numbers is: {result}")
#
# """
# #
# result1 = Geocode_tool_local.func(task_description,geo_python_code)
# print(f'Result: {result1}')
# --- Simple smoke test for GeoCode_COT_Validation_tool ---

# import json

# 要验证的最小代码块：不做服务器往返，只做本地对象构造与打印
# test_code_ok = """
# import ee
# img = ee.Image.constant(42).rename('ntl')
# print('Created ee.Image:', img)  # 打印的是客户端对象描述，不会触发 getInfo()
# vals = [1, 2, 3]
# print('sum =', sum(vals))
# """
#
# # 调用工具（LangChain v0.2+ 推荐 .invoke）
# result_str = GeoCode_COT_Validation_tool.invoke({"code_block": test_code_ok})
#
# # 结果是 JSON 字符串，这里解析并友好打印
# report = json.loads(result_str)
# print("Status:", report["status"])
# print("Stdout:")
# print(report["stdout"])
# if report["status"] != "pass":
#     print("Error Type:", report["error_type"])
#     print("Error Message:", report["error_message"])
#     print("Traceback:", report["traceback"])
