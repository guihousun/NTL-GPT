from osgeo import gdal
import numpy as np
from pydantic.v1 import BaseModel, Field
from langchain.tools import StructuredTool

# 指数计算函数
def compute_rbli(b, g):
    return b / (g + 1e-6)

def compute_rrli(r, g):
    return r / (g + 1e-6)

def compute_ndibg(b, g):
    return (b - g) / (b + g + 1e-6)

def compute_ndigr(g, r):
    return (g - r) / (g + r + 1e-6)

# 保存影像函数
def save_index_tif(array, reference_tif, output_tif_path, description="Index"):
    ds = gdal.Open(reference_tif)
    geo_transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    rows, cols = array.shape

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif_path, cols, rows, 1, gdal.GDT_Float32)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetDescription(description)
    out_band.SetNoDataValue(-9999)
    out_ds.FlushCache()
    del out_ds
    print(f"✅ {description} image saved to: {output_tif_path}")

# 输入模型
class IndexInput(BaseModel):
    radiance_tif: str = Field(..., description="Path to the RGB radiance GeoTIFF")
    output_tif: str = Field(..., description="Path to save output GeoTIFF")
    index_type: str = Field(..., description="Index type: 'RBLI', 'RRLI', 'NDIBG', 'NDIGR'")

# 主函数
def compute_index_from_rgb_tif(
    radiance_tif: str,
    output_tif: str,
    index_type: str
) -> str:
    ds = gdal.Open(radiance_tif)
    band_r = ds.GetRasterBand(1).ReadAsArray()
    band_g = ds.GetRasterBand(2).ReadAsArray()
    band_b = ds.GetRasterBand(3).ReadAsArray()

    index_type = index_type.upper()
    if index_type == "RBLI":
        index_array = compute_rbli(band_b, band_g)
        description = "RBLI (Blue / Green)"
    elif index_type == "RRLI":
        index_array = compute_rrli(band_r, band_g)
        description = "RRLI (Red / Green)"
    elif index_type == "NDIBG":
        index_array = compute_ndibg(band_b, band_g)
        description = "NDIBG (Blue - Green / Blue + Green)"
    elif index_type == "NDIGR":
        index_array = compute_ndigr(band_g, band_r)
        description = "NDIGR (Green - Red / Green + Red)"
    else:
        raise ValueError("Unsupported index type. Choose from: RBLI, RRLI, NDIBG, NDIGR.")

    save_index_tif(index_array, radiance_tif, output_tif, description)
    return f"✅ {index_type} computation completed. Output saved to: {output_tif}"

# LangChain工具封装
SDGSAT1_index_tool = StructuredTool.from_function(
    func=compute_index_from_rgb_tif,
    name="SDGSAT1_compute_index",
    description=(
        "Compute a pixelwise index (RBLI, RRLI, NDIBG, or NDIGR) from an SDGSAT-1 RGB radiance GeoTIFF. "
        "The result is saved as a single-band GeoTIFF with the same spatial reference as the input. "
        "Example input:"
        """
        radiance_tif="C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_rgb.tif",
        output_tif="C:/NTL_Agent/Night_data/SDGSAT-1/Test1_RRLI.tif",
        index_type="RRLI"
        """
    ),
    args_schema=IndexInput,
    return_direct=True
)


# from osgeo import gdal
# import numpy as np
# from pydantic.v1 import BaseModel, Field
#
# # 指数计算函数
# def compute_rbli(b, g):
#     return b / (g + 1e-6)
#
# def compute_rrli(r, g):
#     return r / (g + 1e-6)
#
# def compute_ndibg(b, g):
#     return (b - g) / (b + g + 1e-6)
#
# def compute_ndigr(g, r):
#     return (g - r) / (g + r + 1e-6)
#
# # 保存影像函数
# def save_index_tif(array, reference_tif, output_tif_path, description="Index"):
#     ds = gdal.Open(reference_tif)
#     geo_transform = ds.GetGeoTransform()
#     projection = ds.GetProjection()
#     rows, cols = array.shape
#
#     driver = gdal.GetDriverByName('GTiff')
#     out_ds = driver.Create(output_tif_path, cols, rows, 1, gdal.GDT_Float32)
#     out_ds.SetGeoTransform(geo_transform)
#     out_ds.SetProjection(projection)
#     out_band = out_ds.GetRasterBand(1)
#     out_band.WriteArray(array)
#     out_band.SetDescription(description)
#     out_band.SetNoDataValue(-9999)
#     out_ds.FlushCache()
#     del out_ds
#     print(f"✅ {description} image saved to: {output_tif_path}")
#
# # 输入模型
# class IndexInput(BaseModel):
#     radiance_tif: str = Field(..., description="Path to the RGB radiance GeoTIFF")
#     output_tif: str = Field(..., description="Path to save output GeoTIFF")
#     index_type: str = Field(..., description="Index type: 'RBLI', 'RRLI', 'NDIBG', 'NDIGR'")
#
# # 主函数
# def compute_index_from_rgb_tif(radiance_tif: str, output_tif: str, index_type: str) -> str:
#     ds = gdal.Open(radiance_tif)
#     band_r = ds.GetRasterBand(1).ReadAsArray()
#     band_g = ds.GetRasterBand(2).ReadAsArray()
#     band_b = ds.GetRasterBand(3).ReadAsArray()
#
#     index_type = index_type.upper()
#     if index_type == "RBLI":
#         index_array = compute_rbli(band_b, band_g)
#         description = "RBLI (Blue / Green)"
#     elif index_type == "RRLI":
#         index_array = compute_rrli(band_r, band_g)
#         description = "RRLI (Red / Green)"
#     elif index_type == "NDIBG":
#         index_array = compute_ndibg(band_b, band_g)
#         description = "NDIBG (Blue - Green / Blue + Green)"
#     elif index_type == "NDIGR":
#         index_array = compute_ndigr(band_g, band_r)
#         description = "NDIGR (Green - Red / Green + Red)"
#     else:
#         raise ValueError("Unsupported index type. Choose from: RBLI, RRLI, NDIBG, NDIGR.")
#
#     save_index_tif(index_array, radiance_tif, output_tif, description)
#     return f"✅ {index_type} computation completed. Output saved to: {output_tif}"
#
# # 示例调用
# # compute_index_from_rgb_tif(
# #     radiance_tif="C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_rgb.tif",
# #     output_tif="C:/NTL_Agent/Night_data/SDGSAT-1/Test1_RRLI.tif",
# #     index_type="RRLI"
# # )
