# -*- coding: utf-8 -*-
from typing import Optional
from skimage.transform import radon
from scipy.signal import find_peaks
from skimage import morphology
import numpy as np
from osgeo import gdal
from pydantic.v1 import BaseModel, Field

class RGBRadianceCalibInput(BaseModel):
    input_tif: str = Field(..., description="Path to the input SDGSAT-1 RGB image (GeoTIFF format)")
    calib_file: str = Field(..., description="Path to the corresponding calibration XML file")
    output_tif: str = Field(..., description="Output path for the calibrated RGB GeoTIFF image")
    gray_output_tif: str = Field(..., description="Output path for the grayscale (luminance) image")


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

result = run_strip_removal(
    img_input="C:/NTL_Agent/Night_data/SDGSAT-1/SDG_rgb.tif",
    img_output="C:/NTL_Agent/Night_data/SDGSAT-1/Test1_strip_removal.tif",
    method="median"
)
print(result)
