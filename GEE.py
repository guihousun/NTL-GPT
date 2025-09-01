import ee

# 初始化 Earth Engine（包含项目初始化）
project_id = 'empyrean-caster-430308-m2'
ee.Initialize(project=project_id)


# mandalay = (
#     ee.FeatureCollection('FAO/GAUL/2015/level2')
#     .filter(ee.Filter.eq('ADM2_NAME', 'Mandalay'))
#     .geometry()
# )

# Sagaing = (
#     ee.FeatureCollection('FAO/GAUL/2015/level2')
#     .filter(ee.Filter.eq('ADM2_NAME', 'Sagaing'))
#     .geometry()
# )

mandalay = (
    ee.Geometry.Point([95.98, 21.87])  # 经度在前，纬度在后
    .buffer(5000)  # 5 km 缓冲范围
)


# VIIRS 日度数据集（VNP46A2），波段名为 'DNB_BRDF_Corrected_NTL'
collection = ee.ImageCollection('NASA/VIIRS/002/VNP46A2')

# 工具函数：将日期区间映射为逐日的 {date, mean_ntl} 要素集合
def daily_mean_fc(region, start_str, end_exclusive_str):
    start = ee.Date(start_str)
    end_exclusive = ee.Date(end_exclusive_str)  # 右开区间
    n_days = ee.Number(end_exclusive.difference(start, 'day')).toInt()
    date_list = ee.List.sequence(0, n_days.subtract(1)).map(lambda d: start.advance(d, 'day'))

    def per_day(date):
        date = ee.Date(date)
        daily = (
            collection
            .filterDate(date, date.advance(1, 'day'))
            .filterBounds(region)
            .select('DNB_BRDF_Corrected_NTL')
            .map(lambda img: img.updateMask(img.select('DNB_BRDF_Corrected_NTL').gt(0)))
        )
        mean_img = daily.mean().clip(region)
        mean_val = mean_img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=500,
            maxPixels=1e9
        ).get('DNB_BRDF_Corrected_NTL')

        return ee.Feature(None, {'date': date.format('YYYY-MM-dd'), 'mean_ntl': mean_val})

    return ee.FeatureCollection(date_list.map(per_day))

# 1) 计算 2025-03-14 ~ 2025-03-21（含）的“逐日均值”的中值
fc_pre = daily_mean_fc(mandalay, '2025-03-14', '2025-03-22')  # end 为开区间
fc_pre_valid = fc_pre.filter(ee.Filter.notNull(['mean_ntl']))
pre_list = ee.List(fc_pre_valid.aggregate_array('mean_ntl'))
pre_median = pre_list.reduce(ee.Reducer.median())

# 2) 计算 2025-04-05 ~ 2025-04-12（含）的“逐日均值”的中值
fc_post = daily_mean_fc(mandalay, '2025-04-05', '2025-04-13')
fc_post_valid = fc_post.filter(ee.Filter.notNull(['mean_ntl']))
post_list = ee.List(fc_post_valid.aggregate_array('mean_ntl'))
post_median = post_list.reduce(ee.Reducer.median())

# 3) 计算 2025-03-29 当天的均值
fc_eq = daily_mean_fc(mandalay, '2025-03-29', '2025-03-30')
eq_mean = ee.List(fc_eq.aggregate_array('mean_ntl')).get(0)

# 拉回结果
print("Median of daily means (2025-03-14 ~ 2025-03-21):", pre_median.getInfo())
print("Median of daily means (2025-04-05 ~ 2025-04-12):", post_median.getInfo())
print("Daily mean (2025-03-29):", eq_mean.getInfo())
