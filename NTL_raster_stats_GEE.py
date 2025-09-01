import calendar
import ee
import pandas as pd
from datetime import datetime, timedelta
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

project_id = 'empyrean-caster-430308-m2'
ee.Initialize(project=project_id)

class GEE_NTL_Stats_Input(BaseModel):
    study_area: str = Field(..., description="行政区名称（中国请用中文，例如 '上海市'）")
    scale_level: str = Field(..., description="行政区级别，可选 'country', 'province', 'city', 'county'")
    time_range_input: str = Field(...,
                                  description="Time range. Annual: 'YYYY to YYYY' or 'YYYY'. Monthly: 'YYYY-MM to YYYY-MM'. Daily: 'YYYY-MM-DD to YYYY-MM-DD'.")
    dataset_name: str = Field(default=None, description="数据集名称，可选 'NPP-VIIRS-Like'（默认）, 'NPP-VIIRS', 'DMSP-OLS', 'VNP46A2', 'VNP46A1'")
    temporal_resolution: str = Field(default='annual', description="时间分辨率，可选 'annual', 'monthly', 'daily'")
    output_csv_path: str = Field(..., description="输出 CSV 文件路径")

import re
import calendar
from datetime import datetime, timedelta

def parse_time_range(time_range_input: str, temporal_resolution: str):
    tr = time_range_input.replace(' ', '')
    if 'to' in tr:
        start_str, end_str = [s.strip() for s in tr.split('to')]
    else:
        start_str = end_str = tr

    if temporal_resolution == 'annual':
        if not re.fullmatch(r'\d{4}', start_str) or not re.fullmatch(r'\d{4}', end_str):
            raise ValueError("Annual format must be 'YYYY' or 'YYYY to YYYY'.")
        start_date, end_date = f"{start_str}-01-01", f"{end_str}-12-31"
    elif temporal_resolution == 'monthly':
        if not re.fullmatch(r'\d{4}-\d{2}', start_str) or not re.fullmatch(r'\d{4}-\d{2}', end_str):
            raise ValueError("Monthly format must be 'YYYY-MM' or 'YYYY-MM to YYYY-MM'.")
        sy, sm = map(int, start_str.split('-'))
        ey, em = map(int, end_str.split('-'))
        start_date = f"{sy}-{sm:02d}-01"
        end_date = f"{ey}-{em:02d}-{calendar.monthrange(ey, em)[1]}"
    elif temporal_resolution == 'daily':
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', start_str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', end_str):
            raise ValueError("Daily format must be 'YYYY-MM-DD' or 'YYYY-MM-DD to YYYY-MM-DD'.")
        start_date, end_date = start_str, end_str
    else:
        raise ValueError("temporal_resolution must be one of: 'annual', 'monthly', 'daily'.")

    if datetime.strptime(start_date, '%Y-%m-%d') > datetime.strptime(end_date, '%Y-%m-%d'):
        raise ValueError("Start date cannot be later than end date.")

    return start_date, end_date


def GEE_NTL_statistics_per_image(
    study_area: str,
    scale_level: str,
    time_range_input: str,
    dataset_name: str = None,
    temporal_resolution: str = 'annual',
    output_csv_path: str = None
):
    # ---------------- 获取行政区 ----------------
    national_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/World_countries")
    province_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/province")
    city_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/city")
    county_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/county")

    def get_admin_boundary(level):
        directly_governed = ['北京市', '天津市', '上海市', '重庆市']
        if level == 'province' or (level == 'city' and study_area in directly_governed):
            return province_collection, 'name'
        elif level == 'country':
            return national_collection, 'NAME'
        elif level == 'city':
            return city_collection, 'name'
        elif level == 'county':
            return county_collection, 'name'
        else:
            raise ValueError("scale_level 必须为 'country', 'province', 'city', 'county'")

    admin_boundary, name_prop = get_admin_boundary(scale_level)
    region = admin_boundary.filter(ee.Filter.eq(name_prop, study_area))
    if region.size().getInfo() == 0:
        raise ValueError(f"未找到 {scale_level} 层级下的 {study_area}")
    region = region.geometry()

    # ---------------- 获取影像集合 ----------------
    start_date, end_date = parse_time_range(time_range_input, temporal_resolution)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if temporal_resolution == 'annual':
        dataset_name = dataset_name or 'NPP-VIIRS-Like'
        if dataset_name == 'NPP-VIIRS-Like':
            col_id, band = 'projects/sat-io/open-datasets/npp-viirs-ntl', 'b1'
        elif dataset_name == 'NPP-VIIRS':
            col_id, band = 'NOAA/VIIRS/DNB/ANNUAL_V21', 'average'
        elif dataset_name == 'DMSP-OLS':
            col_id, band = 'BNU/FGS/CCNL/v1', 'b1'
        else:
            raise ValueError("annual 模式下 dataset_name 必须为 'NPP-VIIRS-Like', 'NPP-VIIRS', 'DMSP-OLS'")

        images = []
        for y in range(start_dt.year, end_dt.year + 1):
            y_start = f"{y}-01-01"
            y_end = f"{y+1}-01-01"
            img = ee.ImageCollection(col_id).filterDate(y_start, y_end).select(band).filterBounds(region).mean().clip(region)
            images.append(img.set('system:time_start', ee.Date(y_start).millis()))
        NTL_collection = ee.ImageCollection(images)

    elif temporal_resolution == 'monthly':
        col_id, band = 'NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG', 'avg_rad'
        images = []
        for y in range(start_dt.year, end_dt.year + 1):
            m_start = 1 if y > start_dt.year else start_dt.month
            m_end = 12 if y < end_dt.year else end_dt.month
            for m in range(m_start, m_end + 1):
                s_day = f"{y}-{m:02d}-01"
                e_day = f"{y}-{m:02d}-{calendar.monthrange(y, m)[1]}"
                img = ee.ImageCollection(col_id).filterDate(s_day, e_day).select(band).filterBounds(region).mean().clip(region)
                images.append(img.set('system:time_start', ee.Date(s_day).millis()))
        NTL_collection = ee.ImageCollection(images)

    elif temporal_resolution == 'daily':
        dataset_name = dataset_name or 'VNP46A2'
        if dataset_name == 'VNP46A2':
            col_id, band = 'NASA/VIIRS/002/VNP46A2', 'DNB_BRDF_Corrected_NTL'
        elif dataset_name == 'VNP46A1':
            col_id, band = 'NOAA/VIIRS/001/VNP46A1', 'DNB_At_Sensor_Radiance_500m'
        else:
            raise ValueError("daily 模式下 dataset_name 必须为 'VNP46A2' 或 'VNP46A1'")
        NTL_collection = ee.ImageCollection(col_id).filterDate(start_date, (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")).select(band).filterBounds(region).map(lambda img: img.clip(region))

    else:
        raise ValueError("temporal_resolution 必须为 'annual', 'monthly', 'daily'")

    # ---------------- 计算每一幅影像的统计值 ----------------
    stats_list = []
    image_list = NTL_collection.toList(NTL_collection.size())
    n = NTL_collection.size().getInfo()

    for i in range(n):
        img = ee.Image(image_list.get(i))
        date_str = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        stat_dict = img.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.max(), sharedInputs=True).combine(ee.Reducer.sum(), sharedInputs=True),
            geometry=region,
            scale=500,
            bestEffort=True
        ).getInfo()
        stats_list.append({"date": date_str, "mean": stat_dict.get('mean'), "max": stat_dict.get('max'), "sum": stat_dict.get('sum')})

    df = pd.DataFrame(stats_list)
    if output_csv_path:
        df.to_csv(output_csv_path, index=False, encoding="utf-8-sig", float_format="%.4f")
        print(f"✅ 已保存到 {output_csv_path}")

    return df

# 注册为 Tool
GEE_NTL_statistics_per_image_tool = StructuredTool.from_function(
    func=GEE_NTL_statistics_per_image,
    name="GEE_NTL_statistics_per_image",
    description="从 GEE 获取指定区域和时间范围的逐影像 NTL 统计值（mean、max、sum），支持 annual / monthly / daily 时间分辨率和多种数据源。",
    args_schema=GEE_NTL_Stats_Input
)

df = GEE_NTL_statistics_per_image(
    study_area="上海市",
    scale_level="province",
    time_range_input="2020 to 2020",
    dataset_name="NPP-VIIRS",
    temporal_resolution="annual",
    output_csv_path="C:/NTL_Agent/Shanghai_2020_Q1_stats.csv"
)

