import numpy as np
import geopandas as gpd
import rasterio
import rasterio.mask
from shapely.geometry import mapping
import pandas as pd
from tqdm import tqdm  # 加个进度条
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
from shapely.ops import unary_union
def calc_TNTL(ntl_array):
    """Total Nighttime Light Intensity (TNTL) 总夜间灯光强度"""
    return np.nansum(ntl_array)

def calc_LArea(ntl_array, pixel_area):
    """Lit Area (LArea) 发光区域面积"""
    lit_mask = ntl_array > 0
    return np.sum(lit_mask) * pixel_area  # pixel_area 单位：平方米或公顷

def calc_3DPLand(ntl_array):
    """3D Percentage of Landscape (3DPLand) 三维灯光景观占比"""
    max_ntl = np.nanmax(ntl_array)
    n_pixels = np.sum(~np.isnan(ntl_array))
    if max_ntl == 0 or n_pixels == 0:
        return np.nan
    return np.nansum(ntl_array) / (max_ntl * n_pixels)

def calc_3DED(ntl_array):
    """3D Edge Density (3DED) 三维边缘密度"""
    # 简化版处理：按非零区域边界近似
    from scipy import ndimage

    lit_mask = ntl_array > 0
    labeled, num_features = ndimage.label(lit_mask)

    perimeter = 0
    total_intensity = np.nansum(ntl_array)

    for region_label in range(1, num_features + 1):
        region = (labeled == region_label)
        edges = ndimage.binary_dilation(region) ^ region
        perimeter += np.sum(edges)

    if total_intensity == 0:
        return np.nan
    return perimeter / total_intensity

def calc_3DLPI(ntl_array):
    """3D Largest Patch Index (3DLPI) 三维最大斑块指数"""
    from scipy import ndimage

    lit_mask = ntl_array > 0
    labeled, num_features = ndimage.label(lit_mask)

    region_intensities = []
    for region_label in range(1, num_features + 1):
        region = (labeled == region_label)
        region_intensities.append(np.nansum(ntl_array[region]))

    if len(region_intensities) == 0:
        return np.nan
    return np.nanmax(region_intensities) / np.nansum(ntl_array)

def calc_ANTL(ntl_array):
    """Average Nighttime Light Intensity (ANTL) 平均夜间灯光强度"""
    valid_pixels = np.sum(~np.isnan(ntl_array))
    if valid_pixels == 0:
        return np.nan
    return np.nansum(ntl_array) / valid_pixels

def calc_DNTL(ntl_array):
    """Deviation of Nighttime Light Intensity (DNTL) 夜间灯光强度离散度"""
    valid_pixels = np.sum(~np.isnan(ntl_array))
    mean_ntl = calc_ANTL(ntl_array)
    if valid_pixels == 0:
        return np.nan
    deviation = np.nansum((ntl_array - mean_ntl) ** 2) / valid_pixels
    return deviation

def calc_SDNTL(ntl_array):
    """Standard Deviation of Nighttime Light (SDNTL) 夜间灯光标准差"""
    return np.nanstd(ntl_array)

def calc_MaxNTL(ntl_array):
    """Maximum Nighttime Light Intensity (MaxNTL) 最大亮度"""
    return np.nanmax(ntl_array)

def calc_MinNTL(ntl_array):
    """Minimum Nighttime Light Intensity (MinNTL) 最小亮度"""
    return np.nanmin(ntl_array)

def calc_HistNTL(ntl_array, bins=10):
    """Pixel value distribution (HistNTL) 像素亮度分布直方图"""
    return np.histogram(ntl_array[~np.isnan(ntl_array)], bins=bins)
# 之前定义的指数计算函数（可以直接用上面那版）
# calc_TNTL, calc_LArea, calc_3DPLand, calc_3DED, calc_3DLPI, calc_ANTL, calc_DNTL

def calc_indices_per_polygon(ntl_array, mask_array, pixel_area, hist_bins=10, selected_indices=None):
    """
    给定 NTL 影像数组和掩膜，计算选定的夜光景观指数
    参数：
        ntl_array: 原始夜光数据 (2D array)
        mask_array: 掩膜（True 表示保留像素）
        pixel_area: 单个像素面积（单位 m²）
        hist_bins: 直方图分箱数（默认10）
        selected_indices: 要计算的指标名称列表（默认 None 表示全部计算）
    返回：
        字典形式的指标计算结果
    """
    masked_ntl = np.where(mask_array, ntl_array, np.nan)
    index_dict = {}

    def is_selected(name):
        return (selected_indices is None) or (name in selected_indices)

    if is_selected("MaxNTL"):
        index_dict["MaxNTL"] = np.nanmax(masked_ntl)
    if is_selected("MinNTL"):
        index_dict["MinNTL"] = np.nanmin(masked_ntl)
    if is_selected("SDNTL"):
        index_dict["SDNTL"] = np.nanstd(masked_ntl)
    if is_selected("HistNTL"):
        hist_values, _ = np.histogram(masked_ntl[~np.isnan(masked_ntl)], bins=hist_bins)
        index_dict["HistNTL"] = hist_values.tolist()
    if is_selected("TNTL"):
        index_dict["TNTL"] = calc_TNTL(masked_ntl)
    if is_selected("LArea"):
        index_dict["LArea"] = calc_LArea(masked_ntl, pixel_area)
    if is_selected("3DPLand"):
        index_dict["3DPLand"] = calc_3DPLand(masked_ntl)
    if is_selected("3DED"):
        index_dict["3DED"] = calc_3DED(masked_ntl)
    if is_selected("3DLPI"):
        index_dict["3DLPI"] = calc_3DLPI(masked_ntl)
    if is_selected("ANTL"):
        index_dict["ANTL"] = calc_ANTL(masked_ntl)
    if is_selected("DNTL"):
        index_dict["DNTL"] = calc_DNTL(masked_ntl)

    return index_dict

def NTL_raster_statistics(ntl_tif_path, shapefile_path, output_csv_path, selected_indices=None):
    """
    针对每个行政区 + 整个 shapefile 边界计算夜光景观指数，并保存成 CSV
    参数：
        ntl_tif_path: 夜光影像路径
        shapefile_path: 行政区矢量文件路径
        output_csv_path: 输出 CSV 路径
        selected_indices: 要计算的指数名称列表（如 ['TNTL', 'LArea']），None 表示全部计算
    """
    with rasterio.open(ntl_tif_path) as src:
        ntl_data = src.read(1).astype(np.float32)
        ntl_data[ntl_data == src.nodata] = np.nan
        ntl_profile = src.profile
        pixel_size_x = abs(src.transform.a)
        pixel_size_y = abs(src.transform.e)
        pixel_area = pixel_size_x * pixel_size_y

        # 读取 shapefile，统一坐标系
        gdf = gpd.read_file(shapefile_path)
        gdf = gdf.to_crs(ntl_profile['crs'])

        # Step 1：全局指数计算（整张 shapefile 合并后的区域）
        global_geom = unary_union(gdf.geometry)
        mask_global, _, _ = rasterio.mask.raster_geometry_mask(
            src, [mapping(global_geom)], invert=False, all_touched=True
        )
        global_indices = calc_indices_per_polygon(
            ntl_data, ~mask_global, pixel_area, selected_indices=selected_indices
        )

        # Step 2：各个行政区逐一计算
        results = []
        for idx, row in tqdm(gdf.iterrows(), total=len(gdf)):
            geom = row.geometry
            if geom.is_empty:
                continue

            mask_local, _, _ = rasterio.mask.raster_geometry_mask(
                src, [mapping(geom)], invert=False, all_touched=True
            )
            local_indices = calc_indices_per_polygon(
                ntl_data, ~mask_local, pixel_area, selected_indices=selected_indices
            )

            result = {
                '行政区': row['name'],
                **local_indices
            }
            results.append(result)

        # Step 3：添加全局结果
        results.append({
            '行政区': '全域',  # 或 '总体', 'Total'
            **global_indices
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig', float_format="%.4f")
    print(f"✅ 全部计算完成，结果保存到：{output_csv_path}")
    return df



class NTL_raster_statistics_input(BaseModel):
    ntl_tif_path: str = Field(..., description="输入的夜间灯光影像路径")
    shapefile_path: str = Field(..., description="输入的行政区划shp路径")
    output_csv_path: str = Field(..., description="输出的指数csv路径")
    selected_indices: list[str] = Field(
        default=None,
        description="（可选）需要计算的指数名称列表，如 ['TNTL', 'LArea', '3DPLand']。若不指定则计算全部指标。"
    )

NTL_raster_statistics = StructuredTool.from_function(
    func=NTL_raster_statistics,
    name="NTL_raster_statistics",
    description=(
        "Calculates nighttime light (NTL) metrics for each region in a shapefile using a single-band NTL GeoTIFF. "
        "Supports regional and global summary. Metrics include:\n"
        "- TNTL: Total NTL intensity\n"
        "- LArea: Lit area (pixels > 0)\n"
        "- ANTL: Average NTL\n"
        "- DNTL: NTL deviation\n"
        "- SDNTL: NTL standard deviation\n"
        "- MaxNTL / MinNTL: Max/Min brightness\n"
        "- 3DPLand: 3D % of landscape\n"
        "- 3DED: 3D edge density\n"
        "- 3DLPI: 3D largest patch index\n"
        "- HistNTL: NTL histogram (default 10 bins)\n"
        "Specify selected indices via 'selected_indices'. Useful for urban/economic analysis. "
        "Input: NTL raster path, shapefile path, output CSV path."
    ),
    args_schema=NTL_raster_statistics_input,
)


# result = NTL_raster_statistics.run({
#     "ntl_tif_path": "C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VIIRS_2020.tif",
#     "shapefile_path": "C:/NTL_Agent/report/shp/Shanghai/上海.shp",
#     "output_csv_path": "shanghai_TNTL_only.csv",
#     "selected_indices": ["TNTL", "LArea", "3DPLand"]
#     })