from pydantic import BaseModel

class BVNTLInput(BaseModel):
    ntl_tif: str
    volume_tif: str
    output_tif: str

def compute_bvntl_index(ntl_tif: str, volume_tif: str, output_tif: str) -> str:
    import rasterio
    import numpy as np

    with rasterio.open(ntl_tif) as ntl_src:
        ntl = ntl_src.read(1)
        profile = ntl_src.profile

    with rasterio.open(volume_tif) as vol_src:
        volume = vol_src.read(1)

    bvntl = np.where(volume > 0, ntl / volume, 0)
    profile.update(dtype='float32')

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(bvntl.astype(np.float32), 1)

    return f"BV-NTL image saved to: {output_tif}"



from langchain.tools import StructuredTool

bvntl_index_tool = StructuredTool.from_function(
    func=compute_bvntl_index,
    name="bvntl_index_tool",
    description="""
    Compute the Building Volume Adjusted Nighttime Light (BV-NTL) index by dividing a nightlight radiance raster 
    by a building volume raster. This index can help improve correlation with population or urban density.

    **Example:**
    Input:
        ntl_tif = "C:/NTL_GPT/BVNTL/NTL_2020_06.tif"
        volume_tif = "C:/NTL_GPT/BVNTL/building_volume_2020.tif"
        output_tif = "C:/NTL_GPT/BVNTL/BVNTL_2020_06.tif"

    Output:
        "BV-NTL image saved to: C:/NTL_GPT/BVNTL/BVNTL_2020_06.tif"
    """,
    input_type="BVNTLInput"  # 请根据你的环境定义 dataclass 或 Pydantic 模型
)

class VNCIInput(BaseModel):
    ndvi_tif: str
    ntl_tif: str
    output_tif: str


def compute_vnci_index(ndvi_tif: str, ntl_tif: str, output_tif: str) -> str:
    import rasterio
    import numpy as np

    # Define triangle in feature space
    A = np.array([0.1, 0])
    B = np.array([0.8, 0])
    C = np.array([0.3, 60])

    def point_dist(p, a, b):
        ap = np.array(p) - np.array(a)
        ab = np.array(b) - np.array(a)
        return np.abs(np.cross(ab, ap)) / np.linalg.norm(ab)

    with rasterio.open(ndvi_tif) as ndvi_src:
        ndvi = ndvi_src.read(1).astype(np.float32) / 10000  # Scale MODIS NDVI
        profile = ndvi_src.profile

    with rasterio.open(ntl_tif) as ntl_src:
        ntl = ntl_src.read(1).astype(np.float32)

    rows, cols = ndvi.shape
    vnci = np.zeros_like(ndvi, dtype=np.float32)

    for i in range(rows):
        for j in range(cols):
            x = ndvi[i, j]
            y = ntl[i, j]
            if x <= 0 or y < 0:
                vnci[i, j] = 0
                continue
            d = point_dist([x, y], A, B)
            d_max = point_dist(C, A, B)
            vnci[i, j] = d / d_max if d_max > 0 else 0

    profile.update(dtype='float32')

    with rasterio.open(output_tif, 'w', **profile) as dst:
        dst.write(vnci, 1)

    return f"VNCI image saved to: {output_tif}"


from langchain.tools import StructuredTool

vnci_index_tool = StructuredTool.from_function(
    func=compute_vnci_index,
    name="vnci_index_tool",
    description="""
    Compute Vegetation Nighttime Condition Index (VNCI) using NDVI and nighttime light (NTL) imagery 
    by constructing a triangular feature space and computing normalized perpendicular distances.

    **Example:**
    Input:
        ndvi_tif = "C:/NTL_GPT/VNCI/ndvi_2021.tif"
        ntl_tif = "C:/NTL_GPT/VNCI/ntl_2021.tif"
        output_tif = "C:/NTL_GPT/VNCI/vnci_2021.tif"

    Output:
        "VNCI image saved to: C:/NTL_GPT/VNCI/vnci_2021.tif"
    """,
    input_type="VNCIInput"
)
