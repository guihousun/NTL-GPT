from osgeo import gdal, osr
import numpy as np
from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

def save_classification_tif(class_array, reference_tif, output_tif_path):
    """
    使用参考影像的地理信息，将分类结果数组保存为 GeoTIFF 文件。
    class_array: numpy 数组，分类值（0=未照明, 1=WLED, 2=RLED, 3=Other）
    reference_tif: 用于获取地理参考信息的 GeoTIFF 文件路径（如 RRLI）
    output_tif_path: 输出路径
    """
    ref_ds = gdal.Open(reference_tif)
    if ref_ds is None:
        raise ValueError(f"无法打开参考影像：{reference_tif}")

    geo_transform = ref_ds.GetGeoTransform()
    projection = ref_ds.GetProjection()
    rows, cols = class_array.shape

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_tif_path, cols, rows, 1, gdal.GDT_Byte)
    out_ds.SetGeoTransform(geo_transform)
    out_ds.SetProjection(projection)

    band = out_ds.GetRasterBand(1)
    band.WriteArray(class_array)
    band.SetDescription('Light Classification (0=Dark, 1=WLED, 2=RLED, 3=Other)')
    band.SetNoDataValue(0)  # 0 表示未照明区域

    out_ds.FlushCache()
    del out_ds

    print(f"✅ 灯光分类结果已保存为：{output_tif_path}")

class LightIndexClassificationInput(BaseModel):
    rrli_tif: str = Field(..., description="Path to the RRLI GeoTIFF file (Red/Green Ratio)")
    rbli_tif: str = Field(..., description="Path to the RBLI GeoTIFF file (Blue/Green Ratio)")
    output_tif: str = Field(..., description="Path to save the classified light types (GeoTIFF format)")

def classify_light_types_from_rrli_rbli(rrli_tif: str, rbli_tif: str, output_tif: str) -> str:
    """
    Classify light source types (WLED, RLED, Other) based on RRLI and RBLI GeoTIFF images.

    Args:
        rrli_tif (str): Path to the RRLI GeoTIFF file (Red/Green Ratio).
        rbli_tif (str): Path to the RBLI GeoTIFF file (Blue/Green Ratio).
        output_tif (str): Path to save the classified light types (GeoTIFF format).

    Returns:
        str: Confirmation message with output path.
    """
    ds_rrli = gdal.Open(rrli_tif)
    ds_rbli = gdal.Open(rbli_tif)
    rrli = ds_rrli.GetRasterBand(1).ReadAsArray()
    rbli = ds_rbli.GetRasterBand(1).ReadAsArray()

    light_class = np.full(rrli.shape, 0, dtype=np.uint8)  # 初始化为未照明区域

    raw_class = np.full(rrli.shape, 3, dtype=np.uint8)  # 默认是 Other (3)
    raw_class[rrli > 9] = 2  # RLED
    raw_class[(rrli <= 9) & (rbli > 0.57)] = 1  # WLED

    lit_mask = (rrli > 0) | (rbli > 0)  # 照亮区域掩膜
    light_class[lit_mask] = raw_class[lit_mask]

    save_classification_tif(light_class, reference_tif=rrli_tif, output_tif_path=output_tif)

    return f"✅ Light type classification (from indices) completed. Output saved to: {output_tif}"

light_index_classification_tool = StructuredTool.from_function(
    classify_light_types_from_rrli_rbli,
    name="classify_light_types_from_rrli_rbli",
    description="Classify light source types (WLED, RLED, Other) based on precomputed RRLI and RBLI index images (GeoTIFF). "
                "Output is a GeoTIFF with pixel-level classification.",
    args_schema=LightIndexClassificationInput,
)


# light_type_classification_tool.run({
#     "radiance_tif": "C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_rgb.tif",
#     "output_tif": "C:/NTL_Agent/Night_data/SDGSAT-1/light_type_class.tif"
# # })

# from osgeo import gdal
# import numpy as np
# from pydantic.v1 import BaseModel, Field
#
# def compute_rbli(band_b, band_g):
#     rbli = band_b / (band_g + 1e-6)  # 防止除零
#     return rbli
#
# def save_rbli_tif(rbli_array, reference_tif, output_tif_path):
#     """
#     保存 RBLI 指数为 GeoTIFF 文件。
#     """
#     ds = gdal.Open(reference_tif)
#     geo_transform = ds.GetGeoTransform()
#     projection = ds.GetProjection()
#     rows, cols = rbli_array.shape
#
#     driver = gdal.GetDriverByName('GTiff')
#     out_ds = driver.Create(output_tif_path, cols, rows, 1, gdal.GDT_Float32)
#     out_ds.SetGeoTransform(geo_transform)
#     out_ds.SetProjection(projection)
#     out_ds.GetRasterBand(1).WriteArray(rbli_array)
#     out_ds.GetRasterBand(1).SetDescription('RBLI (Blue / Green)')
#     out_ds.GetRasterBand(1).SetNoDataValue(-9999)
#
#     out_ds.FlushCache()
#     del out_ds
#
#     print(f"✅ RBLI image saved to: {output_tif_path}")
#
# class RBLIInput(BaseModel):
#     radiance_tif: str = Field(..., description="Path to the radiometrically calibrated RGB image (GeoTIFF format)")
#     output_tif: str = Field(..., description="Path to save the RBLI result (GeoTIFF format)")
#
# def compute_rbli_from_rgb_tif(radiance_tif: str, output_tif: str) -> str:
#     ds = gdal.Open(radiance_tif)
#     band_g = ds.GetRasterBand(2).ReadAsArray()
#     band_b = ds.GetRasterBand(3).ReadAsArray()
#
#     rbli = compute_rbli(band_b, band_g)
#     save_rbli_tif(rbli, reference_tif=radiance_tif, output_tif_path=output_tif)
#
#     return f"✅ RBLI computation completed. Output saved to: {output_tif}"
#
# compute_rbli_from_rgb_tif("C:/NTL_Agent/Night_data/SDGSAT-1/Test1_radiance_rgb.tif","C:/NTL_Agent/Night_data/SDGSAT-1/Test1_RBLI.tif")