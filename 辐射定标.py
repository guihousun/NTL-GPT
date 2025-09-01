# -*- coding: utf-8 -*-
import numpy as np
from osgeo import gdal
import xml.etree.ElementTree as ET

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


def calibrate_rgb_from_calib_file(input_tif, calib_file, output_tif, gray_output_tif):
    gain, bias = parse_calib_file(calib_file)

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


calibrate_rgb_from_calib_file(
    input_tif= "C:/NTL_Agent/Night_data/SDGSAT-1/Test1_strip_removal.tif",
    calib_file= "C:/NTL_Agent/Night_data/SDGSAT-1/Test1_calib.xml",
    output_tif= "C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_rgb.tif",
    gray_output_tif= "C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_gray.tif")