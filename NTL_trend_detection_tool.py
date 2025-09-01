from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import numpy as np
import rasterio
from pymannkendall import original_test, sens_slope
import os
from typing import List, Optional


class RasterTrendDetectionInput(BaseModel):
    raster_files: List[str] = Field(
        ...,
        description=(
            "List of file paths to raster images in time series (e.g., for each month over multiple years). "
            "Files should be single-band rasters with consistent resolution and extent. "
            "Example: ['C:/NTL_Agent/Night_data_上海/上海市_NightLights_2020-01.tif', 'C:/NTL_Agent/Night_data_上海/上海市_NightLights_2020-02.tif', ...]"
        )
    )
    save_path: Optional[str] = Field(
        None,
        description=(
            "Path to save the resulting trend raster files. If not provided, saves to 'C:/NTL_Agent/Night_data_上海/trend_result'."
        )
    )


def NTL_trend_analysis(raster_files: List[str], save_path: Optional[str] = None) -> str:
    """
    Perform trend detection on a stack of raster files representing time-series data,
    outputting both a trend slope map (indicating rate of change) and a significance map
    (indicating trend significance based on Mann-Kendall test).

    Parameters:
    - raster_files (List[str]): Paths to time-series raster files (must be spatially aligned).
    - save_path (Optional[str]): Folder where output trend raster files will be saved.

    Returns:
    - str: Success message with output file paths.
    """
    save_path = save_path or "C:/NTL_Agent/raster_trend"
    os.makedirs(save_path, exist_ok=True)
    slope_file = os.path.join(save_path, "slope_map.tif")
    significance_file = os.path.join(save_path, "significance_map.tif")

    # Load raster stack into a 3D numpy array: (time, rows, cols)
    with rasterio.open(raster_files[0]) as src:
        profile = src.profile
        profile.update(count=1, dtype='float32', nodata=np.nan)
        data_stack = np.array([rasterio.open(f).read(1) for f in raster_files])

    # Prepare arrays to store results
    rows, cols = data_stack.shape[1], data_stack.shape[2]
    slope_array = np.full((rows, cols), np.nan, dtype=np.float32)
    significance_array = np.zeros((rows, cols), dtype=np.int8)  # -1, 0, 1

    # Loop through each pixel
    for i in range(rows):
        for j in range(cols):
            pixel_series = data_stack[:, i, j]
            if np.all(~np.isnan(pixel_series)):  # Only process if no missing data
                # Calculate Sen's slope
                slope_result = sens_slope(pixel_series)
                slope_array[i, j] = slope_result.slope

                # Perform Mann-Kendall trend test
                trend_result = original_test(pixel_series)
                if trend_result.p < 0.05:
                    significance_array[i, j] = 1 if trend_result.trend == 'increasing' else -1
                else:
                    significance_array[i, j] = 0
            else:
                # Missing data at this pixel, keep default values (NaN slope, 0 significance)
                pass

    # Save slope raster
    with rasterio.open(slope_file, 'w', **profile) as dst:
        dst.write(slope_array, 1)

    # Save significance raster
    with rasterio.open(significance_file, 'w', **profile) as dst:
        dst.write(significance_array, 1)

    return (
        f"Raster trend analysis completed.\n"
        f"Slope map saved at: {slope_file}\n"
        f"Significance map saved at: {significance_file}"
    )


# Create the StructuredTool for raster trend detection
NTL_trend_analysis_tool = StructuredTool.from_function(
    NTL_trend_analysis,
    name="NTL_trend_analysis",
    description=(
        """
        This tool analyzes nighttime light (NTL) raster data trends over time. It generates:
        1. **Slope Map**: Each pixel represents Sen's slope, indicating the NTL brightness change rate.
        2. **Significance Map**: Each pixel indicates trend significance (-1: significant decrease, 0: no trend, 1: significant increase).

        ### Notes:
        - Only for nighttime light (NTL) data.
        - Input files must be single-band rasters with consistent resolution and extent.
        - Files should follow a time-series order (e.g., monthly/yearly).

        ### Example:
        - raster_files: ['C:/NTL_Agent/Night_data_上海/上海市_NightLights_2020-01.tif', ...]
        - save_path: 'C:/NTL_Agent/Night_data_上海/trend_result'

        Outputs: 'slope_map.tif' & 'significance_map.tif'.
        """
    )
    ,
    input_type=RasterTrendDetectionInput,
)
