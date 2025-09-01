# -*- coding: utf-8 -*-
import h5py
from pyresample import geometry
from typing import Optional
from skimage.transform import radon
from scipy.signal import find_peaks
from skimage import morphology
import numpy as np
from osgeo import gdal
import xml.etree.ElementTree as ET
import geemap
import xarray as xr
import dask.array as da
from satpy import Scene
from pyresample import create_area_def
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field
import gc
import os

def parse_calib_file(calib_path):
    """从 .calib XML 文件中提取 RGB 波段的 GAIN 和 BIAS"""
    try:
        with open(calib_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
    except UnicodeDecodeError:
        with open(calib_path, 'r', encoding='gbk') as f:
            xml_content = f.read()

    # 解析 XML 字符串
    root = ET.fromstring(xml_content)
    giu = root.find('.//GIU/VERSION')
    gain = {
        'R': float(giu.find('RADIANCE_GAIN_BAND_RED').text),
        'G': float(giu.find('RADIANCE_GAIN_BAND_GREEN').text),
        'B': float(giu.find('RADIANCE_GAIN_BAND_BLUE').text)
    }
    bias = {
        'R': float(giu.find('RADIANCE_BIAS_BAND_RED').text),
        'G': float(giu.find('RADIANCE_BIAS_BAND_GREEN').text),
        'B': float(giu.find('RADIANCE_BIAS_BAND_BLUE').text)
    }
    return gain, bias

#
def calibrate_rgb_from_calib_file(input_tif, output_tif, gray_output_tif):
    gain, bias = parse_calib_file("C:/NTL_Agent/Night_data/SDGSAT-1/Test1_calib.xml")

    # 带宽设置（根据论文）
    bandwidth = {
        'R': 294,
        'G': 106,
        'B': 102
    }

    dataset = gdal.Open(input_tif)
    if dataset is None:
        raise Exception(f"无法打开图像文件：{input_tif}")

    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    # DN 原始值
    DN_red = dataset.GetRasterBand(1).ReadAsArray()
    DN_green = dataset.GetRasterBand(2).ReadAsArray()
    DN_blue = dataset.GetRasterBand(3).ReadAsArray()

    # 辐射定标 + 单位转换
    L_red = (DN_red * gain['R'] + bias['R']) * bandwidth['R'] * 1e2
    L_green = (DN_green * gain['G'] + bias['G']) * bandwidth['G'] * 1e2
    L_blue = (DN_blue * gain['B'] + bias['B']) * bandwidth['B'] * 1e2

    calibrated = np.array([L_red, L_green, L_blue])
    gray = 0.2989 * L_red + 0.5870 * L_green + 0.1140 * L_blue
    # 输出为 GeoTIFF
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = L_red.shape
    # ✅ 只保留一次 RGB 输出
    out_ds = driver.Create(output_tif, cols, rows, 3, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    for i in range(3):
        out_ds.GetRasterBand(i + 1).WriteArray(calibrated[i])
    out_ds.FlushCache()
    del out_ds  # ✅ 释放文件

    # ✅ 然后再写灰度图
    gray_ds = driver.Create(gray_output_tif, cols, rows, 1, gdal.GDT_Float32)
    gray_ds.SetGeoTransform(geo_transform)
    gray_ds.SetProjection(projection)
    gray_ds.GetRasterBand(1).WriteArray(gray)
    gray_ds.FlushCache()
    del gray_ds  # ✅ 释放文件

    return f"✅ Calibration completed. RGB saved to: {output_tif}, Grayscale saved to: {gray_output_tif}"

# class RGBRadianceCalibInput(BaseModel):
#     input_tif: str = Field(..., description="Path to the input SDGSAT-1 RGB image (GeoTIFF format)")
#     output_tif: str = Field(..., description="Output path for the calibrated RGB GeoTIFF image")
#     gray_output_tif: str = Field(..., description="Output path for the grayscale (luminance) image")
#
#
#
#
#
# rgb_radiance_calib_tool = StructuredTool.from_function(
#     func=calibrate_rgb_from_calib_file,
#     name="calibrate_sdgsat1_rgb_and_generate_grayscale",
#     description=(
#         "Perform radiometric calibration on an SDGSAT-1 RGB image using an XML calibration file. "
#         "Outputs both a calibrated RGB GeoTIFF and a grayscale luminance image derived via perceptual weighting."
#     ),
#     args_schema=RGBRadianceCalibInput,
#     return_direct=True
# )
#
def Strip_removal(img_input, img_output, theta, threshold, method):
    gdal.AllRegister()

    img = gdal.Open(img_input)  # 读取文件 \ read file
    im_proj = (img.GetProjection())  # 读取投影 \ Read Projection
    im_Geotrans = (img.GetGeoTransform())  # 读取仿射变换 \ Read Affine Transformation

    w = img.RasterXSize  # 列数 \ Number of columns
    h = img.RasterYSize  # 行数 \ Number of rows

    band1 = img.GetRasterBand(1)  # 获取栅格图像三个波段 \ Acquire three bands of raster images
    band2 = img.GetRasterBand(2)
    band3 = img.GetRasterBand(3)

    # 将图像读取为数组 \ Reading an image as an array
    band1 = band1.ReadAsArray(0, 0, w, h)
    band2 = band2.ReadAsArray(0, 0, w, h)
    band3 = band3.ReadAsArray(0, 0, w, h)

    # 创建全为1的模板数组 \ Create a template array with all 1's
    I1 = np.ones((h, w), dtype=np.uint8)
    I2 = np.ones((h, w), dtype=np.uint8)
    I3 = np.ones((h, w), dtype=np.uint8)

    '''由于SDGSAT夜光数据在不同时期的图像背景值不同，存在差异，早期图像背景值为7，
       为了统一数据的值范围，将背景值为1的图像也处理成背景值为7的图像
       Due to differences in image background values for SDGSAT noctilucent data during different periods, the early image background value is 7.
       In order to unify the value range of the data, images with a background value of 1 are also processed into images with a background value of 7'''
    one_loc = np.where(band1 == 1)
    one_loc = np.array(one_loc)
    b = one_loc.size
    if b != 0:
        band1[band1 == 1] += 6
        band2[band2 == 1] += 6
        band3[band3 == 1] += 6
        for i in range(h):
            for j in range(w):
                if band1[i, j] == 6:
                    band1[i, j] = 0
                if band2[i, j] == 6:
                    band2[i, j] = 0
                if band3[i, j] == 6:
                    band3[i, j] = 0

    # 检测条带潜在像元，但是并不都是条带 \ Detect potential pixels in stripes, but not all stripes
    for i in range(h):
        for j in range(w):
            if band1[i, j] == 7 and band2[i, j] == 7 and band3[i, j] == 7:
                I1[i, j] = 0
                I2[i, j] = 0
                I3[i, j] = 0
            if band1[i, j] > 7:
                I1[i, j] = 0

            if band2[i, j] > 7:
                I2[i, j] = 0

            if band3[i, j] > 7:
                I3[i, j] = 0
    # 找出条带所在的位置 \ Find the location of the strip
    moban1 = RGB_Stripe_loc(I1, theta, threshold=threshold)
    moban2 = RGB_Stripe_loc(I2, theta, threshold=threshold)
    moban3 = RGB_Stripe_loc(I3, theta, threshold=threshold)

    # 去除潜在条带像元中的条带 \ Removing Stripes from Potential Striped Pixels
    m1 = moban1 * I1
    m2 = moban2 * I2
    m3 = moban3 * I3

    # 得到三个通道所有条带的位置 \ Obtain the positions of all stripes in the three channels
    m = np.logical_not(m1) * np.logical_not(m2) * np.logical_not(m3)

    # 对原始通道进行条带去除 \ Striping the original channel
    b1 = m * band1
    b2 = m * band2
    b3 = m * band3

    # 对去除的像元进行8像元差值，使用三分位数进行插值 \ Perform an 8-pixel difference on the removed pixels and interpolate using the third quantile
    if method == 'median':
        for i in range(h - 2):
            for j in range(w - 2):
                if b1[i, j] and b2[i, j] and b3[i, j] == 7:
                    b1[i, j] = np.median([b1[i - 1, j - 1], b1[i, j - 1], b1[i + 1, j - 1], b1[i - 1, j], b1[i + 1, j],
                                          b1[i + 1, j - 1], b1[i, j + 1], b1[i + 1, j + 1], b1[i - 1, j + 2],
                                          b1[i, j + 2], b1[i + 1, j + 2]])
                    b2[i, j] = np.median([b2[i - 1, j - 1], b2[i, j - 1], b2[i + 1, j - 1], b2[i - 1, j], b2[i + 1, j],
                                          b2[i + 1, j - 1], b2[i, j + 1], b2[i + 1, j + 1], b2[i - 1, j + 2],
                                          b2[i, j + 2], b2[i + 1, j + 2]])
                    b3[i, j] = np.median([b3[i - 1, j - 1], b3[i, j - 1], b3[i + 1, j - 1], b3[i - 1, j], b3[i + 1, j],
                                          b3[i + 1, j - 1], b3[i, j + 1], b3[i + 1, j + 1], b3[i - 1, j + 2],
                                          b3[i, j + 2], b3[i + 1, j + 2]])
    else:
        for i in range(h - 2):
            for j in range(w - 2):
                if b1[i, j] and b2[i, j] and b3[i, j] == 7:
                    b1[i, j] = np.percentile(
                        [b1[i - 1, j - 1], b1[i, j - 1], b1[i + 1, j - 1], b1[i - 1, j], b1[i + 1, j],
                         b1[i + 1, j - 1], b1[i, j + 1], b1[i + 1, j + 1], b1[i - 1, j + 2], b1[i, j + 2],
                         b1[i + 1, j + 2]], 75)
                    b2[i, j] = np.percentile(
                        [b2[i - 1, j - 1], b2[i, j - 1], b2[i + 1, j - 1], b2[i - 1, j], b2[i + 1, j],
                         b2[i + 1, j - 1], b2[i, j + 1], b2[i + 1, j + 1], b2[i - 1, j + 2], b2[i, j + 2],
                         b2[i + 1, j + 2]], 75)
                    b3[i, j] = np.percentile(
                        [b3[i - 1, j - 1], b3[i, j - 1], b3[i + 1, j - 1], b3[i - 1, j], b3[i + 1, j],
                         b3[i + 1, j - 1], b3[i, j + 1], b3[i + 1, j + 1], b3[i - 1, j + 2], b3[i, j + 2],
                         b3[i + 1, j + 2]], 75)

    # 处理异常像元 \ Handling abnormal pixels
    for i in range(h):
        for j in range(w):
            # 小于7视为背景值 \ Less than 7 is considered a background value
            if b1[i, j] < 7 or b2[i, j] < 7 or b3[i, j] < 7:
                b1[i, j] = b2[i, j] = b3[i, j] = 7

            # 双7为条带设为背景值 \ Double 7 is set as the background value for the stripe
            if (b1[i, j] == 7 and b2[i, j] == 7) or (b1[i, j] == 7 and b3[i, j] == 7) or (
                    b2[i, j] == 7 and b3[i, j] == 7):
                b1[i, j] = b2[i, j] = b3[i, j] = 7
    # 二值化 \ Binarization
    b11 = np.where(b1 > 7, 1, 0)
    b22 = np.where(b2 > 7, 1, 0)
    b33 = np.where(b3 > 7, 1, 0)

    # 去除连通域为8个像元的噪声 \ Remove noise with a connected domain of 8 pixels
    b1_bw = bwareaopen(b11, 8)
    b2_bw = bwareaopen(b22, 8)
    b3_bw = bwareaopen(b33, 8)

    # 将三个通道数组合为一个三维数组 \ Combine three channel numbers into a three-dimensional array
    out = np.array([b1_bw * b1, b2_bw * b2, b3_bw * b3])
    # 输出图像 \ Output Image
    write_img(img_output, im_proj, im_Geotrans, out)


def RGB_Stripe_loc(I, theta, threshold):
    a = np.rint((max(theta) - min(theta) / 0.01))  # 设置角度范围 \ Set Angle Range
    R = radon(I, theta=theta,
              preserve_range=True)  # 计算各角度方向下条带潜在像元模板的积分值 \ Calculate the integration value of the potential pixel template of the strip in each angular direction
    h, w = I.shape
    si = R.size / a
    dd = I
    moban = np.zeros((h, w))
    ij = 0

    while ij < a:
        [loc, peaks] = find_peaks(R[:, ij],
                                  threshold=threshold)  # 根据一定的阈值范围寻找积分峰值 \ Find the integration peak value according to a certain threshold range
        # 找出满足条件峰值的条带
        for iw in range(len(loc)):
            k = np.tan((theta[ij] - 90) * np.pi / 180)
            for j in range(h):
                xx = np.int(loc[iw] + j)
                yy = np.int(k * j)
                if loc[iw] + j < w and yy + 6 < h:
                    moban[yy, xx] = 1
                    moban[yy + 1, xx] = 1
                    moban[yy + 2, xx] = 1
                    moban[yy + 3, xx] = 1
                    moban[yy + 4, xx] = 1
                    moban[yy + 5, xx] = 1
                    moban[yy + 6, xx] = 1

        ij += 1

    return moban


# 输出图像 \ Output Image
def write_img(filename, im_proj, im_geotrans, im_data):
    # 判断栅格数据的数据类型 \ Determine the data type of raster data
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数 \ Interpreting array dimensions
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件 \ create a file
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数 \ Write affine transformation parameters
    dataset.SetProjection(im_proj)  # 写入投影 \ Write Projection

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据 \ Write array data
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset


def bwareaopen(image, threshold):
    # 去除小于阈值的连通区域  \ Remove connected areas that are smaller than the threshold
    filtered_image = morphology.remove_small_objects(image.astype(bool), min_size=threshold, connectivity=1).astype(
        np.uint8)

    return filtered_image


class StripRemovalInput(BaseModel):
    img_input: str = Field(..., description="输入图像文件路径（例如：'C:/NTL_Agent/Night_data/SDGSAT-1/Test1.tif'）")
    img_output: str = Field(..., description="输出图像文件路径（例如：'C:/NTL_Agent/Night_data/SDGSAT-1/Test1_strip_removal.tif'）")
    method: str = Field(..., description="插值方法，支持 'median' 或 'percentile'")

    # 以下参数提供默认值
    start_angle: Optional[int] = Field(80, description="Radon 变换起始角度，默认80")
    end_angle: Optional[int] = Field(100, description="Radon 变换结束角度，默认100")
    threshold: Optional[float] = Field(80, description="峰值检测阈值，默认80")


def run_strip_removal(
        img_input: str,
        img_output: str,
        method: str,
        start_angle: int = 80,
        end_angle: int = 100,
        threshold: float = 80
) -> str:
    theta = np.arange(start_angle, end_angle)
    Strip_removal(img_input, img_output, theta=theta, threshold=threshold, method=method)
    return f"Striping removed. Output saved to {img_output}"



class RGBRadianceCalibInput(BaseModel):
    input_tif: str = Field(..., description="Path to the input SDGSAT-1 RGB image (GeoTIFF format)")
    output_tif: str = Field(..., description="Output path for the calibrated RGB GeoTIFF image")
    gray_output_tif: str = Field(..., description="Output path for the grayscale (luminance) image")






class NTL_daily_data_preprocess_Input(BaseModel):
    study_area: str = Field(..., description="Name of the study area of interest. Example:'南京市'")
    scale_level: str = Field(..., description="Scale level, e.g.'country', 'province', 'city', 'county'.")
    time_range_input: str = Field(...,
                                  description="Time range in the format 'YYYY-MM to YYYY-MM'. Example: '2020-01 to 2020-02'")

def VNP46A2_NTL_data_preprocess(
        study_area: str,
        scale_level: str,
        time_range_input: str,
):
    import re
    import ee
    from datetime import datetime, timedelta

    # Set administrative boundary dataset based on scale level
    national_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/World_countries")
    province_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/province")
    city_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/city")
    county_collection = ee.FeatureCollection("projects/empyrean-caster-430308-m2/assets/county")

    # Select administrative boundaries
    def get_administrative_boundaries(scale_level):
        # Handle directly governed cities as province-level data in China
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
            raise ValueError("Unknown scale level. Options are 'country', 'province', 'city', or 'county'.")
        return admin_boundary, name_property

    admin_boundary, name_property = get_administrative_boundaries(scale_level)
    region = admin_boundary.filter(ee.Filter.eq(name_property, study_area))
    # Validate region
    if region.size().getInfo() == 0:
        raise ValueError(f"No area named '{study_area}' found under scale level '{scale_level}'.")
    region = region.geometry()


    # if region.isEmpty().getInfo():
    #     raise ValueError(f"No area named '{study_area}' found under scale level '{scale_level}'.")

    # Parse time range
    def parse_time_range(time_range_input):
        time_range_input = time_range_input.replace(' ', '')
        if 'to' in time_range_input:
            start_str, end_str = time_range_input.split('to')
            start_str, end_str = start_str.strip(), end_str.strip()
        else:
            # Single date input
            start_str = end_str = time_range_input.strip()

        if not re.match(r'^\d{4}-\d{2}-\d{2}$', start_str) or not re.match(r'^\d{4}-\d{2}-\d{2}$', end_str):
            raise ValueError("Invalid daily format. Use 'YYYY-MM-DD' or 'YYYY-MM-DD to YYYY-MM-DD'.")
        start_date, end_date = start_str, end_str

        if datetime.strptime(start_date, '%Y-%m-%d') > datetime.strptime(end_date, '%Y-%m-%d'):
            raise ValueError("Start date cannot be later than end date.")

        return start_date, end_date

    start_date, end_date = parse_time_range(time_range_input)

    NTL_collection = (
        ee.ImageCollection('NASA/VIIRS/002/VNP46A2')
        .filterDate(start_date, (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d'))
        .select('DNB_BRDF_Corrected_NTL')
        .filterBounds(region)
        .map(lambda image: image.clip(region))
    )

    # ========== 计算每个影像的组号 ==========

    def add_group_number(image):
        # 计算影像日期
        date = ee.Date(image.get('system:time_start'))
        # 计算组号（0-15）
        days_diff = date.difference(ee.Date(start_date), 'day')
        group_number = days_diff.mod(16).int()
        # 将组号添加到影像属性中
        return image.set('group_number', group_number)

    # 将函数应用到影像集合中
    viirs_collection = NTL_collection.map(add_group_number)
    # ========== 数据预处理：消除传感器角度影响 ==========

    # ========== 实现逐像素角度效应校正 ==========

    # **步骤1：计算年度逐像元均值影像 N**

    # 注意处理空值，使用 ee.Reducer.mean() 会自动忽略空值
    annual_mean_image = viirs_collection.mean()

    # **步骤2：按组号分组影像集合，计算每个组的逐像元均值影像 N1, N2, ..., N16**

    group_numbers = ee.List.sequence(0, 15)

    def compute_group_mean_image(group_number):
        group_number = ee.Number(group_number)
        group_collection = viirs_collection.filter(ee.Filter.eq('group_number', group_number))
        group_mean_image = group_collection.mean()
        # 将组号添加到影像属性中
        return group_mean_image.set('group_number', group_number)

    # 计算每个组的平均影像，并生成一个 ImageCollection
    group_mean_images = ee.ImageCollection(group_numbers.map(compute_group_mean_image))

    # **步骤3：计算每个组的角度效应系数影像 Ai = Ni / N**

    def compute_correction_image(image):
        group_number = image.get('group_number')
        group_mean_image = image
        # 计算校正系数影像 Ai = Ni / N
        correction_image = group_mean_image.divide(annual_mean_image).unmask(1)
        # 将组号添加到校正影像属性中
        return correction_image.set('group_number', group_number)

    # 生成校正系数影像集合
    correction_images = group_mean_images.map(compute_correction_image)

    # **步骤4：对每个组的影像集合进行校正**

    def correct_group_images(group_number):
        group_number = ee.Number(group_number)
        # 获取对应组号的校正系数影像 Ai
        correction_image = correction_images.filter(ee.Filter.eq('group_number', group_number)).first()
        # 获取对应组号的影像集合
        group_collection = viirs_collection.filter(ee.Filter.eq('group_number', group_number))
        # 对组内的每个影像进行校正
        corrected_group = group_collection.map(lambda image: image.divide(correction_image)
                                               .copyProperties(image, image.propertyNames()))
        return corrected_group

    # 对每个组进行校正，得到校正后的影像集合列表
    corrected_groups = group_numbers.map(correct_group_images)

    # 将每个 ImageCollection 展开成单个 Image 列表并合并为一个 ImageCollection
    all_corrected_images = ee.ImageCollection(corrected_groups.iterate(
        lambda img_col, acc: ee.ImageCollection(acc).merge(ee.ImageCollection(img_col)),
        ee.ImageCollection([])
    ))

    # 生成最终的 corrected_collection
    corrected_collection = all_corrected_images

    # **步骤5：按日期排序**

    corrected_collection = corrected_collection.sort('system:time_start')
    def set_filename(image):
        date = ee.Date(image.get('system:time_start')).format('yyyy_MM_dd')
        file_name = ee.String(study_area).cat('_').cat(date)
        return image.set('file_name', file_name)

    corrected_collection = corrected_collection.map(set_filename)
    # 从集合中提取所有 file_name
    filenames = corrected_collection.aggregate_array('file_name').getInfo()
    print(filenames)
    # ========= 批量导出 corrected_collection 到 Assets =========

    out_dir = r"C:\NTL_Agent\Night_data\GEE"

    print(corrected_collection.aggregate_array("system:index").getInfo())

    geemap.ee_export_image_collection(corrected_collection, out_dir=out_dir, filenames = filenames)
    print(f"The preprocessed VNP46A2 data has been saved in {out_dir}")

    return f"The preprocessed VNP46A2 data has been saved in {out_dir}."


# 读取SDR文件中的Radiance、QF1_VIIRSDNBSDR以及SDR_GEO中的Longitude_TC、Latitude_TC、QF2_VIIRSSDRGEO、
# SolarZenithAngle、QF1_SCAN_VIIRSSDRGEO、LunarZenithAngle
def read_h5(sdr_data_path, SDR_names, SDR_GEO_names):
    with h5py.File(sdr_data_path, 'r') as sdr_file:
        GROUP_DNB_SDR = dict()
        GROUP_DNB_SDR_GEO = dict()

        if len(SDR_names) != 0:
            for SDR_name in SDR_names:
                temp_subdataset = sdr_file.get(SDR_name)
                if temp_subdataset is None:
                    print("The subdataset:%s don't exist." % (SDR_name))
                    continue
                GROUP_DNB_SDR[SDR_name] = temp_subdataset[()]
                del temp_subdataset

        if len(SDR_GEO_names) != 0:
            for SDR_GEO_name in SDR_GEO_names:
                temp_subdataset = sdr_file.get(SDR_GEO_name)
                if temp_subdataset is None:
                    print("The subdataset:%s don't exist." % (SDR_GEO_name))
                    continue
                GROUP_DNB_SDR_GEO[SDR_GEO_name] = temp_subdataset[()] # temp_subdataset.value
                del temp_subdataset

    return GROUP_DNB_SDR, GROUP_DNB_SDR_GEO


# 对SDR进行质量控制，剔除受边缘噪声、阳光、月光等影响的数据，输出数据还还未进行云掩膜
def sdr_radiance_filter(SDR_GEO_path, SDR_names, SDR_GEO_names, sdr_out_dir):
    GROUP_DNB_SDR, GROUP_DNB_SDR_GEO = read_h5(SDR_GEO_path, SDR_names, SDR_GEO_names)
    sdr_output_name = os.path.basename(SDR_GEO_path).split('.')[0]

    # 1. VIIRS Fill Values
    cloud_radiance = GROUP_DNB_SDR[SDR_names[0]]
    r_fillvalue = np.array([-999.3, -999.5, -999.8, -999.9])
    radiance_mask = np.isin(cloud_radiance, r_fillvalue)
    print(f"[VIIRS Fill Values] Masked Pixels: {np.sum(radiance_mask)}, Total: {radiance_mask.size}, "
          f"Percentage: {np.sum(radiance_mask) / radiance_mask.size * 100:.2f}%")

    # 2. Edge-of-swath pixels
    edge_of_swath_mask = np.zeros_like(cloud_radiance, dtype=bool)
    edge_of_swath_mask[:, 0:230] = True
    edge_of_swath_mask[:, 3838:] = True
    print(f"[Edge-of-Swath] Masked Pixels: {np.sum(edge_of_swath_mask)}, Total: {edge_of_swath_mask.size}, "
          f"Percentage: {np.sum(edge_of_swath_mask) / edge_of_swath_mask.size * 100:.2f}%")

    # 3. QF1_VIIRSDNBSDR_flags
    qf1_viirsdnbsdr = GROUP_DNB_SDR[SDR_names[1]]
    SDR_Quality_mask = (qf1_viirsdnbsdr & 3) > 0
    Saturated_Pixel_mask = ((qf1_viirsdnbsdr & 12) >> 2) > 0
    Missing_Data_mask = ((qf1_viirsdnbsdr & 48) >> 4) > 0
    Out_of_Range_mask = ((qf1_viirsdnbsdr & 64) >> 6) > 0
    print(f"[QF1] SDR Quality Masked Pixels: {np.sum(SDR_Quality_mask)}")
    print(f"[QF1] Saturated Pixels: {np.sum(Saturated_Pixel_mask)}")
    print(f"[QF1] Missing Data Pixels: {np.sum(Missing_Data_mask)}")
    print(f"[QF1] Out of Range Pixels: {np.sum(Out_of_Range_mask)}")

    # 4. QF2_VIIRSSDRGEO_flags
    qf2_viirssdrgeo = GROUP_DNB_SDR_GEO[SDR_GEO_names[2]]
    qf2_viirssdrgeo_do0_mask = (qf2_viirssdrgeo & 1) > 0
    qf2_viirssdrgeo_do1_mask = ((qf2_viirssdrgeo & 2) >> 1) > 0
    qf2_viirssdrgeo_do2_mask = ((qf2_viirssdrgeo & 4) >> 2) > 0
    qf2_viirssdrgeo_do3_mask = ((qf2_viirssdrgeo & 8) >> 3) > 0
    print(f"[QF2] DO0 Masked Pixels: {np.sum(qf2_viirssdrgeo_do0_mask)}")
    print(f"[QF2] DO1 Masked Pixels: {np.sum(qf2_viirssdrgeo_do1_mask)}")
    print(f"[QF2] DO2 Masked Pixels: {np.sum(qf2_viirssdrgeo_do2_mask)}")
    print(f"[QF2] DO3 Masked Pixels: {np.sum(qf2_viirssdrgeo_do3_mask)}")

    # 5. QF1_SCAN_VIIRSSDRGEO
    qf1_scan_viirssdrgeo = GROUP_DNB_SDR_GEO[SDR_GEO_names[4]]
    within_south_atlantic_anomaly = ((qf2_viirssdrgeo & 16) >> 4) > 0
    print(f"[QF1_SCAN] South Atlantic Anomaly Pixels: {np.sum(within_south_atlantic_anomaly)}")

    # 6. Solar Zenith Angle
    solarZenithAngle = GROUP_DNB_SDR_GEO[SDR_GEO_names[3]]
    solarZenithAngle_mask = (solarZenithAngle < 118.5)
    print(f"[Solar Zenith Angle] Valid Pixels (<118.5°): {np.sum(solarZenithAngle_mask)}")

    # 7. Lunar Zenith Angle
    lunar_zenith = GROUP_DNB_SDR_GEO[SDR_GEO_names[5]]
    moon_illuminance_mask = (lunar_zenith <= 90)
    print(f"[Lunar Zenith Angle] Moon Illuminance Pixels (≤90°): {np.sum(moon_illuminance_mask)}")

    # 8. Combine all masks
    viirs_sdr_geo_mask = np.logical_or.reduce((
        radiance_mask,
        edge_of_swath_mask,
        solarZenithAngle_mask,
        moon_illuminance_mask,
        SDR_Quality_mask,
        Saturated_Pixel_mask,
        Missing_Data_mask,
        Out_of_Range_mask,
        qf2_viirssdrgeo_do0_mask,
        qf2_viirssdrgeo_do1_mask,
        qf2_viirssdrgeo_do2_mask,
        qf2_viirssdrgeo_do3_mask
    ))
    print(f"[Final Combined Mask] Total Masked Pixels: {np.sum(viirs_sdr_geo_mask)}, "
          f"Percentage: {np.sum(viirs_sdr_geo_mask) / viirs_sdr_geo_mask.size * 100:.2f}%")

    viirs_sdr_geo_mask_temp = np.logical_or.reduce((
        radiance_mask,
        solarZenithAngle_mask,
        moon_illuminance_mask,
        SDR_Quality_mask,
        Saturated_Pixel_mask,
        Missing_Data_mask,
        Out_of_Range_mask,
        qf2_viirssdrgeo_do0_mask,
        qf2_viirssdrgeo_do1_mask,
        qf2_viirssdrgeo_do2_mask,
        qf2_viirssdrgeo_do3_mask
    ))

    nan_count = np.sum(viirs_sdr_geo_mask_temp == True)
    nan_count_fraction = (nan_count / np.size(viirs_sdr_geo_mask_temp)) * 100
    if nan_count_fraction > 95:  # 如果数据受月光或者阳光影响太大，导致有效数据占比很小，那么这部分数据被忽略，不保存结果
        print(sdr_output_name + " ignored.")
        del viirs_sdr_geo_mask, radiance_mask, edge_of_swath_mask, solarZenithAngle_mask, moon_illuminance_mask
        del SDR_Quality_mask, Saturated_Pixel_mask, Missing_Data_mask, Out_of_Range_mask, qf2_viirssdrgeo_do0_mask
        del qf2_viirssdrgeo_do1_mask, qf2_viirssdrgeo_do2_mask, qf2_viirssdrgeo_do3_mask, viirs_sdr_geo_mask_temp
        del lunar_zenith
        gc.collect()
    else:
        del viirs_sdr_geo_mask_temp, GROUP_DNB_SDR
        del radiance_mask, solarZenithAngle_mask, moon_illuminance_mask, edge_of_swath_mask
        del SDR_Quality_mask, Saturated_Pixel_mask, Missing_Data_mask, Out_of_Range_mask, qf2_viirssdrgeo_do0_mask
        del qf2_viirssdrgeo_do1_mask, qf2_viirssdrgeo_do2_mask, qf2_viirssdrgeo_do3_mask
        del lunar_zenith
        gc.collect()

        fill_value = np.nan
        scalefactor = np.float32(pow(10, 9))
        cloud_radiance = cloud_radiance * scalefactor  # convert Watts to nanoWatts
        cloud_radiance[viirs_sdr_geo_mask] = fill_value  # set fill value for masked pixels in DNB
        # del viirs_sdr_geo_mask

        sdr_lon_data = GROUP_DNB_SDR_GEO[SDR_GEO_names[0]]
        sdr_lon_data[viirs_sdr_geo_mask] = np.nan
        sdr_lat_data = GROUP_DNB_SDR_GEO[SDR_GEO_names[1]]
        sdr_lat_data[viirs_sdr_geo_mask] = np.nan
        del viirs_sdr_geo_mask
        gc.collect()
        sdr_swath_def = geometry.SwathDefinition(
            xr.DataArray(da.from_array(sdr_lon_data, chunks=4096), dims=('y', 'x')),
            xr.DataArray(da.from_array(sdr_lat_data, chunks=4096), dims=('y', 'x'))
        )
        sdr_metadata_dict = {'name': 'dnb', 'area': sdr_swath_def}

        sdr_scn = Scene()
        sdr_scn['Radiance'] = xr.DataArray(
            da.from_array(cloud_radiance, chunks=4096),
            attrs=sdr_metadata_dict,
            dims=('y', 'x')  # https://satpy.readthedocs.io/en/latest/dev_guide/xarray_migration.html#id1
        )

        sdr_scn.load(['Radiance'])
        proj_str = '+proj=aea +lat_1=27 +lat_2=45 +lat_0=35 +lon_0=105 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs'  # aea坐标
        sdr_custom_area = create_area_def('aea', proj_str, resolution=750, units='meters', area_extent=[-2641644.056319, -3051079.954397, 2222583.910354, 2174272.289243]) # China Aea Extent
        sdr_proj_scn = sdr_scn.resample(sdr_custom_area, resampler='nearest')

        # sdr_proj_shape = sdr_proj_scn.datasets['Radiance'].shape

        sdr_out_path = sdr_out_dir + "\\" + sdr_output_name + '.tif'
        # 必须将enhancement_config设为False，不然输出的值会变的很小
        sdr_proj_scn.save_dataset('Radiance', sdr_out_path, writer='geotiff', dtype=np.float32, enhancement_config=False, fill_value=fill_value)
        print(sdr_output_name + ' processed.')

        # release memory
        sdr_proj_scn = None
        del r_fillvalue
        del fill_value, sdr_proj_scn, sdr_lon_data, sdr_lat_data, sdr_swath_def, sdr_metadata_dict
        gc.collect()

# input_dir代表存放需要处理的sdr文件的文件夹路径
def batch_pro(sdr_input_dir, SDR_out_dir):
    file_list = os.listdir(sdr_input_dir)
    h5_file_list = []
    # 防止出现非h5文件，所以对读出来的文件过滤一下
    for temp_file in file_list:
        if temp_file.endswith('.h5'):
            h5_file_list.append(sdr_input_dir + "\\" + temp_file)

    # 用于在h5文件中提取相应数据的关键字
    SDR_names = ["/All_Data/VIIRS-DNB-SDR_All/Radiance", "/All_Data/VIIRS-DNB-SDR_All/QF1_VIIRSDNBSDR"]
    SDR_GEO_names = ["/All_Data/VIIRS-DNB-GEO_All/Longitude_TC", "/All_Data/VIIRS-DNB-GEO_All/Latitude_TC",
                     "/All_Data/VIIRS-DNB-GEO_All/QF2_VIIRSSDRGEO", "/All_Data/VIIRS-DNB-GEO_All/SolarZenithAngle",
                     '/All_Data/VIIRS-DNB-GEO_All/QF1_SCAN_VIIRSSDRGEO', '/All_Data/VIIRS-DNB-GEO_All/LunarZenithAngle']

    for h5_file in h5_file_list:
        sdr_radiance_filter(h5_file, SDR_names, SDR_GEO_names, SDR_out_dir)


class NOAA20SDRPreprocessInput(BaseModel):
    sdr_input_dir: str = Field(..., description="输入的SDR影像文件夹路径（包含.h5文件）")
    sdr_output_dir: str = Field(..., description="输出的影像文件夹路径（将保存GeoTIFF结果）")

def preprocess_noaa20_sdr_data(sdr_input_dir: str, sdr_output_dir: str) -> str:
    batch_pro(sdr_input_dir, sdr_output_dir)
    return f"NOAA-20 SDR数据预处理完成，结果保存在：{sdr_output_dir}"

noaa20_sdr_preprocess_tool = StructuredTool.from_function(
    func=preprocess_noaa20_sdr_data,
    name="noaa20_VIIRS_preprocess",
    description="对 NOAA-20 VIIRS DNB SDR 影像进行质量控制和预处理，输出剔除阳光、月光、边缘噪声等影响的 GeoTIFF 影像",
    args_schema=NOAA20SDRPreprocessInput,
    return_direct=True
)
# Update the nightlight_download_tool
VNP46A2_angular_correction_tool = StructuredTool.from_function(
    VNP46A2_NTL_data_preprocess,
    name="VNP46A2_angular_correction_tool",
    description=(
        """
        Perform angular effect correction on NASA VNP46A2 daily NTL data from Google Earth Engine, 
        using 16-group mean normalization to remove sensor zenith angle effects, and output the corrected 
        image collection with pixel-wise mean values over the specified date range.

        Parameters:
        - study_area (str): Name of the target region. For China, use Chinese names (e.g., 江苏省, 南京市).
        - scale_level (str): Administrative level ('country', 'province', 'city', 'county').
        - time_range_input (str): Date range in 'YYYY-MM-DD to YYYY-MM-DD' format.

        Output:
        - Exported corrected NTL images to local folder or GEE Assets, with per-pixel angular correction applied.

        Example Input:
        (
            study_area='南京市',
            scale_level='city',
            time_range_input='2020-01-01 to 2020-02-01',
        )
        """
    ),
    input_type=NTL_daily_data_preprocess_Input,
)



SDGSAT1_strip_removal_tool = StructuredTool.from_function(
    func=run_strip_removal,
    name="SDGSAT-1_strip_removal_tool",
    description=(
        "Remove striping noise from SDGSAT-1 GLI RGB imagery. "
        "This tool applies a destriping algorithm to correct periodic or systematic stripe artifacts "
        "commonly observed in SDGSAT-1 raw images. "
        "It should be used as the first step in the preprocessing workflow before radiometric calibration."
    ),
    args_schema=StripRemovalInput
)


SDGSAT1_radiometric_calibration_tool = StructuredTool.from_function(
    func=calibrate_rgb_from_calib_file,
    name="SDGSAT1_radiometric_calibration_tool",
    description=(
        "Perform radiometric calibration on a destriped SDGSAT-1 GLI RGB image. "
        "This tool converts raw digital number (DN) values to top-of-atmosphere (TOA) radiance using sensor-specific calibration coefficients. "
        "It outputs a calibrated RGB GeoTIFF and a grayscale luminance image derived through perceptual weighting "
        "of the R, G, B channels. "
        "This tool assumes that the input image has already been destriped."
        "\n\n"
        "Example input:\n"
        "input_tif = 'C:/NTL_Agent/Night_data/SDGSAT-1/Test1_strip_removal.tif'\n"
        "output_tif = 'C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_rgb.tif'\n"
        "gray_output_tif = 'C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_gray.tif'"
    ),
    args_schema= RGBRadianceCalibInput,
)

class CrossSensorCalibrationInput(BaseModel):
    dmsp_folder: str = Field(..., description="Path to folder containing DMSP-OLS annual images (2000–2013)")
    viirs_folder: str = Field(..., description="Path to folder containing VIIRS-like annual images (2013–2018)")
    aux_data_path: str = Field(..., description="Path to GeoTIFF file containing auxiliary variables for 2013")
    output_folder: str = Field(..., description="Folder to save calibrated output images and trained model")

def dmsp_preprocess_tool(
    dmsp_folder: str,
    viirs_folder: str,
    aux_data_path: str,
    output_folder: str
) -> str:
    """
    Framework for cross-sensor calibration: DMSP to VIIRS-like brightness
    """
    # TODO: implement model training and calibration here
    return f"Calibration workflow initialized. Output will be saved to {output_folder}"

from langchain.tools import StructuredTool

cross_sensor_calibration_dmsp_viirs_tool = StructuredTool.from_function(
    func=dmsp_preprocess_tool,
    name="dmsp_preprocess_tool",
    description=(
        """
        This tool performs cross-sensor calibration by training a machine learning model (e.g., Random Forest) on overlapping year data (e.g., 2013)
        between DMSP-OLS and VIIRS-like NTL datasets. The model incorporates auxiliary data (e.g., population, GDP, electricity, roads),
        and is then applied to historical DMSP-OLS images (2000–2012) to generate calibrated VIIRS-like brightness rasters.

        ### Example Input:
        calibrate_dmsp_to_viirs(
            dmsp_folder='C:/NTL_GPT/DMSP/',
            viirs_folder='C:/NTL_GPT/VIIRS/',
            aux_data_path='C:/NTL_GPT/aux_vars_2013.tif',
            output_folder='C:/NTL_GPT/Calibrated_NTL/'
        )
        """
    ),
    input_type="CrossSensorCalibrationInput"  # You can define this dataclass separately
)

# if __name__ == "__main__":

    # 示例调用
    # strip_removal_tool.run({
    #     "img_input": "C:/NTL_Agent/Night_data/SDGSAT-1/SDG_rgb.tif",
    #     "img_output": "C:/NTL_Agent/Night_data/SDGSAT-1/Test1_strip_removal.tif",
    #     "method": "median"
    # })
    #
    # rgb_radiance_calib_tool.run({
    #     "input_tif": "C:/NTL_Agent/Night_data/SDGSAT-1/Test1_strip_removal.tif",
    #     "calib_file": "C:/NTL_Agent/Night_data/SDGSAT-1/Test1_calib.xml",
    #     "output_tif": "C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_rgb.tif",
    #     "gray_output_tif": "C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_gray.tif"
    # })
    # result = NTL_daily_data_preprocess_tool.run({
    #     "study_area": '南京市',
    #     "scale_level": 'city',
    #     "time_range_input": '2020-01-01 to 2020-02-01'
    # })

# input_sdr_dir = r"C:\课题组\断电\9d_10d_IndiaData\input"  # sdr的存储文件夹
# output_sdr_dir = r"C:\课题组\断电\9d_10d_IndiaData\output"  # 输出文件夹；输出是剔除了边缘噪声、阳光、月光等影响的Radiance数据，格式为geotiff
# noaa20_sdr_preprocess_tool.run({
#     "sdr_input_dir": input_sdr_dir,
#     "sdr_output_dir": output_sdr_dir
# })