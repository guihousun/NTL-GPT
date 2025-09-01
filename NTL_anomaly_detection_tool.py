from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import rasterio
import os

# ===== 输入参数模型 =====
class SimpleAnomalyDetectionInput(BaseModel):
    raster_files: List[str] = Field(
        ...,
        description="时间序列 NTL 栅格文件路径列表（按时间顺序）。"
    )
    target_index: Optional[int] = Field(
        None,
        description="待检测影像在列表中的索引（0-based），不填默认最后一张。"
    )
    k_sigma: float = Field(
        3.0,
        description="异常阈值系数，表示高于均值多少个标准差算异常。"
    )
    save_path: Optional[str] = Field(
        None,
        description="输出结果保存文件夹，默认 'C:/NTL_Agent/raster_anomaly_simple'"
    )

# ===== 简单异常检测函数 =====
def simple_ntl_anomaly_detection(
    raster_files: List[str],
    target_index: Optional[int] = None,
    k_sigma: float = 3.0,
    save_path: Optional[str] = None
) -> str:
    save_path = save_path or "C:/NTL_Agent/raster_anomaly_simple"
    os.makedirs(save_path, exist_ok=True)

    # 自动选择最后一张作为检测影像
    target_index = target_index if target_index is not None else len(raster_files) - 1
    if target_index < 0 or target_index >= len(raster_files):
        return f"Error: target_index 超出范围 (0~{len(raster_files)-1})"

    # 读取所有影像
    with rasterio.open(raster_files[0]) as src:
        profile = src.profile
        height, width = src.height, src.width

    stack = np.empty((len(raster_files), height, width), dtype=np.float32)
    for i, f in enumerate(raster_files):
        with rasterio.open(f) as src:
            stack[i] = src.read(1)

    # 计算基线期（去掉目标期）
    baseline = np.delete(stack, target_index, axis=0)
    mean_img = np.nanmean(baseline, axis=0)
    std_img = np.nanstd(baseline, axis=0)

    # 目标期影像
    target_img = stack[target_index]

    # Z 分数计算
    z_score = (target_img - mean_img) / (std_img + 1e-6)
    anomaly_mask = (z_score > k_sigma).astype(np.uint8)

    # 保存结果
    anomaly_file = os.path.join(save_path, "anomaly_mask.tif")
    profile.update(dtype=rasterio.uint8, count=1)
    with rasterio.open(anomaly_file, "w", **profile) as dst:
        dst.write(anomaly_mask, 1)

    return (
        f"Anomaly detection completed.\n"
        f"Target image: {raster_files[target_index]}\n"
        f"Anomaly mask saved at: {anomaly_file}"
    )

# ===== 注册成工具 =====
simple_ntl_anomaly_detection_tool = StructuredTool.from_function(
    func=simple_ntl_anomaly_detection,
    name="NTL_anomaly_detection_tool",
    description=(
        "Simplified NTL anomaly detection using baseline mean/std from the first N-1 rasters and z-score thresholding on the latest raster.\n"
        "Outputs: baseline_mean.tif, baseline_std.tif, z_score_map.tif, anomaly_map.tif, anomaly_summary.csv.\n"
        "Handles nodata/NaN; checks CRS/grid consistency; latest raster is the target by default."
        "Parameters:\n"
        "- raster_files: List of time-series image paths (in chronological order)\n"
        "- target_index: Index of the image to be detected (0-based), default is the last one\n"
        "- k_sigma: Threshold multiplier (default is 3)\n"
        "- save_path: Output result save path"
    ),
    input_type=SimpleAnomalyDetectionInput,
)

# tool = simple_ntl_anomaly_detection_tool
# print(tool.name)

# simple_ntl_anomaly_detection_tool.run({
#     "raster_files": [
#         "C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VIIRS_2015.tif",
#         "C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VIIRS_2016.tif",
#         "C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VIIRS_2017.tif",
#         "C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VIIRS_2018.tif",
#         "C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VIIRS_2019.tif",
#         "C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VIIRS_2020.tif"
#     ]
# })
#
