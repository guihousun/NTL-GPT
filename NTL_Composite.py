from typing import List, Optional
import os
import numpy as np
import rasterio
from pydantic.v1 import BaseModel, Field
from langchain.tools import StructuredTool

class LocalNTLCompositeInput(BaseModel):
    file_paths: List[str] = Field(..., description="Paths to daily NTL GeoTIFFs (same grid/CRS).")
    out_tif: str = Field(..., description="Output GeoTIFF path for the mean composite.")
    enforce_same_grid: bool = Field(True, description="If True, raise if any input grid/CRS differs from the first.")
    fallback_nodata: Optional[float] = Field(None, description="If an input has no nodata, use this value. If None, fall back to the first image nodata or -1.")

def build_ntl_mean_composite_local(
    file_paths: List[str],
    out_tif: str,
    enforce_same_grid: bool = True,
    fallback_nodata: Optional[float] = None,
) -> str:
    if not file_paths:
        raise ValueError("file_paths is empty.")
    # Read first image metadata/grid
    with rasterio.open(file_paths[0]) as src0:
        profile = src0.profile.copy()
        height, width = src0.height, src0.width
        transform = src0.transform
        crs = src0.crs
        nodata0 = src0.nodata

    if nodata0 is None:
        nodata0 = -1 if fallback_nodata is None else fallback_nodata

    stack_sum = np.zeros((height, width), dtype=np.float64)
    stack_cnt = np.zeros((height, width), dtype=np.uint32)

    for i, fp in enumerate(file_paths, 1):
        with rasterio.open(fp) as ds:
            if enforce_same_grid:
                if ds.height != height or ds.width != width or ds.transform != transform or ds.crs != crs:
                    raise ValueError(f"Grid/CRS mismatch in {fp}. Reproject/resample before compositing.")
            arr = ds.read(1).astype(np.float64)
            nd = ds.nodata if ds.nodata is not None else nodata0
            valid = (arr != nd) & np.isfinite(arr)
            # accumulate
            stack_sum[valid] += arr[valid]
            stack_cnt[valid] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        mean_arr = np.where(stack_cnt > 0, stack_sum / stack_cnt, np.nan)

    os.makedirs(os.path.dirname(out_tif) or ".", exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=rasterio.float32, nodata=nodata0, compress=profile.get("compress", "lzw"))

    # replace NaN with nodata before save
    out_data = mean_arr.copy()
    out_data[~np.isfinite(out_data)] = nodata0

    with rasterio.open(out_tif, "w", **out_profile) as dst:
        dst.write(out_data.astype(np.float32), 1)

    # simple log string
    valid_ratio = float(np.mean(stack_cnt > 0))
    return f"Composite saved: {out_tif}; nodata={nodata0}; valid_pixel_ratio={valid_ratio:.4f}"

NTL_composite_local_tool = StructuredTool.from_function(
    func=build_ntl_mean_composite_local,
    name="NTL_composite_local_tool",
    description=(
        "Build a pixelwise mean NTL composite from local daily GeoTIFFs. "
        "Uses each file's nodata to mask invalid pixels, avoids divide-by-zero, "
        "and preserves grid/CRS from the first image. "
        "Inputs: file_paths (list of GeoTIFFs), out_tif (output path), "
        "enforce_same_grid (default True), fallback_nodata (optional). "
        "Example:\n"
        "file_paths=['C:/NTL/.../2020-01-01.tif','C:/NTL/.../2020-01-02.tif'], "
        "out_tif='C:/NTL/.../Mean_2020-01.tif'"
    ),
    args_schema=LocalNTLCompositeInput,
)

from typing import Optional, Literal
from pydantic.v1 import BaseModel, Field
from langchain.tools import StructuredTool

class NTLCompositeGEEInput(BaseModel):
    study_area: str = Field(..., description="Name of the study area. Example: '南京市'")
    scale_level: Literal['country', 'province', 'city', 'county'] = Field(..., description="Administrative scale level.")
    dataset_name: Literal['VNP46A2', 'VNP46A1'] = Field('VNP46A2', description="Daily VIIRS NTL dataset: 'VNP46A2' (default) or 'VNP46A1'.")
    time_range_input: str = Field(..., description="Date range in 'YYYY-MM-DD to YYYY-MM-DD' format.")
    export_path: str = Field(..., description="Local file path to save the composite image. Example: 'C:/NTL_Agent/Night_data/Shanghai/Composite.tif'")

def NTL_composite_GEE_tool(
    study_area: str,
    scale_level: str,
    dataset_name: str,
    time_range_input: str,
    export_path: str
):
    import os
    import ee
    import geemap
    from datetime import datetime, timedelta

    ee.Initialize(project='empyrean-caster-430308-m2')

    # --- Admin boundary sources ---
    national_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/World_countries")
    province_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/province")
    city_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/city")
    county_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/county")

    def get_administrative_boundaries(scale_level: str):
        directly_governed_cities = ['北京市', '天津市', '上海市', '重庆市']
        if scale_level == 'province' or (scale_level == 'city' and study_area in directly_governed_cities):
            admin_boundary = province_collection
            name_property = 'name'
        elif scale_level == 'country':
            admin_boundary = national_collection
            name_property = 'NAME'
        elif scale_level == 'city':
            admin_boundary = city_collection
            name_property = 'name'
        elif scale_level == 'county':
            admin_boundary = county_collection
            name_property = 'name'
        else:
            raise ValueError("Unknown scale level. Options: 'country', 'province', 'city', 'county'.")
        return admin_boundary, name_property

    admin_boundary, name_property = get_administrative_boundaries(scale_level)
    region = admin_boundary.filter(ee.Filter.eq(name_property, study_area))
    if region.size().getInfo() == 0:
        raise ValueError(f"No area named '{study_area}' found for scale '{scale_level}'.")

    # --- Dataset selection ---
    daily_map = {
        'VNP46A2': {'id': 'NASA/VIIRS/002/VNP46A2', 'band': 'DNB_BRDF_Corrected_NTL'},
        'VNP46A1': {'id': 'NOAA/VIIRS/001/VNP46A1', 'band': 'DNB_At_Sensor_Radiance_500m'}
    }
    if dataset_name not in daily_map:
        raise ValueError("dataset_name must be 'VNP46A2' or 'VNP46A1'.")

    col_id, band = daily_map[dataset_name]['id'], daily_map[dataset_name]['band']

    # --- Parse date range ---
    if 'to' not in time_range_input:
        raise ValueError("time_range_input must be 'YYYY-MM-DD to YYYY-MM-DD'.")
    start_str, end_str = [s.strip() for s in time_range_input.split('to')]
    start_date = datetime.strptime(start_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_str, '%Y-%m-%d') + timedelta(days=1)

    # --- Load and composite ---
    col = (
        ee.ImageCollection(col_id)
        .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        .select(band)
        .filterBounds(region.geometry())
        .map(lambda img: img.updateMask(img.neq(0)))  # 去掉 nodata=0
    )

    composite = col.mean().clip(region)

    # --- Export ---
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    geemap.ee_export_image(
        ee_object=composite,
        filename=export_path,
        scale=500,
        region=region.geometry(),
        crs='EPSG:4326',
        file_per_band=False
    )

    return f"Composite saved to {export_path}"

NTL_composite_GEE_tool = StructuredTool.from_function(
    NTL_composite_GEE_tool,
    name="NTL_composite_GEE_tool",
    description="""
        Composite daily VIIRS NTL data in Google Earth Engine over a given date range and region,
        masking nodata pixels and computing the mean brightness. Saves the composite to a local GeoTIFF.
        Example Input:
        (
            study_area='上海市',
            scale_level='city',
            dataset_name='VNP46A2',
            time_range_input='2020-01-01 to 2020-01-07',
            export_path='C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VNP46A2_DAILY_2020-01_Mean.tif'
        )
    """,
    input_type=NTLCompositeGEEInput,
)


# result = NTL_composite_local_tool.func(
#     file_paths=[
#         r'C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VNP46A2_DAILY_2020-01-01.tif',
#         r'C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VNP46A2_DAILY_2020-01-02.tif',
#         r'C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VNP46A2_DAILY_2020-01-03.tif',
#         r'C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VNP46A2_DAILY_2020-01-04.tif',
#         r'C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VNP46A2_DAILY_2020-01-05.tif',
#         r'C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VNP46A2_DAILY_2020-01-06.tif',
#         r'C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VNP46A2_DAILY_2020-01-07.tif'
#     ],
#     out_tif=r'C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VNP46A2_DAILY_2020-01_Mean.tif',
#     enforce_same_grid=True,
#     fallback_nodata=-1
# )

# result = NTL_composite_GEE_tool.func(
#     study_area='上海市',
#     scale_level='city',
#     dataset_name='VNP46A2',
#     time_range_input='2020-01-01 to 2020-01-07',
#     export_path='C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_VNP46A2_DAILY_2020-01_Mean1.tif'
# )
#
# print(result)
