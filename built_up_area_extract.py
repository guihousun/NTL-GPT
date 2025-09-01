import os
import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field


def read_tif(path):
    with rasterio.open(path) as src:
        image = src.read(1).astype(np.float32)
    return image

def calculate_perimeter(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
    return perimeter

def analyze_thresholds(image, num_thresholds=64):
    min_val = image.min()
    max_val = image.max()
    thresholds = np.linspace(min_val, max_val, num_thresholds)
    perimeters = []
    for t in thresholds:
        binary = np.uint8((image > t) * 255)
        perimeter = calculate_perimeter(binary)
        perimeters.append(perimeter)
    return thresholds, perimeters

def find_optimal_threshold(thresholds, perimeters):
    for i in range(1, len(perimeters)):
        if perimeters[i] > perimeters[i - 1] and all(
            perimeters[j] > perimeters[j - 1] for j in range(i, min(i + 3, len(perimeters)))
        ):
            return thresholds[i]
    return thresholds[np.argmax(perimeters)]

# ====================== 参数模型 ======================

class UrbanExtractionInput(BaseModel):
    tif_path: str = Field(..., description="输入的夜间灯光影像路径")
    output_path: str = Field(..., description="输出的建成区掩膜图像路径")

# ====================== 合并工具函数 ======================

def extract_urban_area_with_optimal_threshold(tif_path: str, output_path: str) -> str:
    # Step 1: 读取影像并计算最佳阈值
    image = read_tif(tif_path)
    thresholds, perimeters = analyze_thresholds(image)
    optimal_threshold = find_optimal_threshold(thresholds, perimeters)
    print(f"最佳阈值为：{optimal_threshold}")

    # Step 2: 生成建成区掩膜
    with rasterio.open(tif_path) as src:
        raw = src.read(1)
        profile = src.profile

    urban_mask = raw >= optimal_threshold
    profile.update(dtype=rasterio.uint8, count=1)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(urban_mask.astype(rasterio.uint8), 1)

    print(f"建成区掩膜图像已保存至：{os.path.abspath(output_path)}")
    return f"Optimal threshold = {optimal_threshold:.2f}\nMask saved to: {os.path.abspath(output_path)}"


urban_extraction_tool = StructuredTool.from_function(
    func=extract_urban_area_with_optimal_threshold,
    name="extract_urban_area_use_change_point",
    description="""
    Automatically extract built-up areas from nighttime light (NTL) imagery using a change-point detection method, and export a binary mask.
    
    **Input:**
    - `tif_path`: Path to the input NTL image (.tif, single-band, float32)
    - `output_path`: Path to save the output binary built-up area mask (.tif)
    
    **Output:**
    - A binary mask GeoTIFF (1 = built-up, 0 = non-built-up)
    - Text output indicating the optimal threshold and saved file path
    
    **Example:**
    Input:
      tif_path = "C:/NTL_Agent/Night_data/Shanghai/NTL_2020.tif"
      output_path = "C:/NTL_Agent/Night_data/Shanghai/urban_mask_2020.tif"
    
    Output:
      Optimal threshold = 38.75
      Mask saved to: C:/NTL_Agent/Night_data/Shanghai/urban_mask_2020.tif
    """,
    input_type=UrbanExtractionInput,
)

# extract_urban_area_with_optimal_threshold(
#     tif_path="C:/NTL_Agent/Night_data/上海市/Annual/NTL_上海市_VIIRS_2020.tif",
#     output_path="C:/NTL_Agent/Night_data/上海市/Annual/urban_mask_2020.tif"
# )

# 示例调用方式（实际使用中由 LangChain Agent 调用）
# result1 = detect_optimal_urban_threshold.run({"tif_path": "C:/NTL_Agent/Night_data/上海市/Annual/NTL_上海市_VIIRS_2020.tif"})
# result2 = generate_urban_mask_by_threshold.run({
#     "tif_path": "C:/NTL_Agent/Night_data/上海市/Annual/NTL_上海市_VIIRS_2020.tif",
#     "output_path": "C:/NTL_Agent/Night_data/上海市/Annual/urban_mask.tif",
#     "optimal_threshold": result1
# })

# with rasterio.open("urban_mask.tif") as src:
#     image = src.read(1)  # 读取第一个波段，假设是单波段灰度图
#     profile = src.profile
#
# # 可视化城市区域掩膜
# plt.figure(figsize=(8, 6))
# plt.imshow(image, cmap='gray')
# plt.title("Extracted Urban Area (Binary Mask)")
# plt.axis('off')  # 关闭坐标轴
# plt.show()