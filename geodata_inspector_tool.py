# geodata_inspector_tool.py
from __future__ import annotations
import json
import os
from typing import List, Optional, Dict, Any

import numpy as np
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from pydantic.v1 import BaseModel, Field
from langchain.tools import StructuredTool


import os
from collections import defaultdict

def simple_key(path: str) -> str:
    """去掉文件名里的所有数字，作为分组键"""
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    no_digits = ''.join(ch for ch in stem if not ch.isdigit())
    # 规整一下分隔符
    no_digits = no_digits.replace('-', '_').replace('.', '_')
    while '__' in no_digits:
        no_digits = no_digits.replace('__', '_')
    return no_digits.strip('_')

def dedupe_by_name_simple(paths, keep="first"):
    """
    基于“去掉数字后的文件名”分组去重。
    keep: 'first' | 'last'
    """
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        groups[simple_key(p)].append((i, p))

    kept = []
    dropped = []
    for key, items in groups.items():
        items_sorted = sorted(items, key=lambda x: x[0])  # 按原顺序
        keep_item = items_sorted[0] if keep == "first" else items_sorted[-1]
        keep_idx, keep_path = keep_item
        kept.append(keep_path)
        for idx, path in items_sorted:
            if path != keep_path:
                dropped.append({"group": key, "path": path})

    # 保持原始顺序
    kept = [p for _, p in sorted([(paths.index(p), p) for p in kept], key=lambda x: x[0])]
    return kept, dropped

class GeoDataInspectorInput(BaseModel):
    raster_paths: Optional[List[str]] = Field(
        default=None,
        description="Absolute paths to raster files (e.g., GeoTIFF)."
    )
    vector_paths: Optional[List[str]] = Field(
        default=None,
        description="Absolute paths to vector files (e.g., .shp, .gpkg, .geojson)."
    )
    sample_pixels: int = Field(
        default=0,
        description="If >0, compute stats on a uniform subsample of roughly this many pixels per raster (for speed). 0 = full image."
    )


def _raster_basic_stats(arr: np.ndarray) -> Dict[str, Any]:
    # arr is a masked array (nodata masked)
    data = arr.compressed()
    if data.size == 0:
        return {
            "count_valid": 0,
            "min": None,  "max": None,
            "mean": None, "std": None
        }
    return {
        "count_valid": int(data.size),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
        "std": float(np.nanstd(data)),
    }


def _raster_report(path: str, sample_pixels: int = 0) -> Dict[str, Any]:
    report: Dict[str, Any] = {"path": path}
    if not os.path.isabs(path):
        report["warning"] = "Path is not absolute."

    with rasterio.open(path) as ds:
        report.update({
            "driver": ds.driver,
            "crs": str(ds.crs) if ds.crs else None,
            "width": ds.width,
            "height": ds.height,
            "count_bands": ds.count,
            "dtype": ds.dtypes[0] if ds.count > 0 else None,
            "resolution": (abs(ds.transform.a), abs(ds.transform.e)),
            "nodata": ds.nodata,
            "bounds": {
                "left": ds.bounds.left, "bottom": ds.bounds.bottom,
                "right": ds.bounds.right, "top": ds.bounds.top
            },
        })

        # Work on band 1 for summary (extend easily to all bands if you wish)
        if ds.count >= 1:
            # Sampling strategy
            if sample_pixels and sample_pixels > 0:
                # Build a coarse grid to approximate requested sample size
                step_x = max(1, int(np.sqrt((ds.width * ds.height) / sample_pixels)))
                window = Window(col_off=0, row_off=0, width=ds.width, height=ds.height)
                band = ds.read(1, window=window)[::step_x, ::step_x]
            else:
                band = ds.read(1)

            nd = ds.nodata
            if nd is None:
                # Try to infer obvious nodata (e.g., negatives for strictly non-negative NTL)
                # You can refine this rule per dataset; here we don't infer, we only mask NaN/inf
                mask = np.isfinite(band)
            else:
                mask = (band != nd) & np.isfinite(band)

            marr = np.ma.array(band, mask=~mask)

            report["band1_stats"] = _raster_basic_stats(marr)
            # Heuristic hints for NTL ranges (optional)
            hints = []
            if "mean" in report["band1_stats"] and report["band1_stats"]["mean"] is not None:
                mn, mx = report["band1_stats"]["min"], report["band1_stats"]["max"]
                if mn is not None and mn < -1e-6:
                    hints.append("Contains negative values; verify nodata and sensor units.")
                if mx is not None and mx > 1e6:
                    hints.append("Very large max; check radiance units or scale.")
            report["hints"] = hints

    return report


def _vector_report(path: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {"path": path}
    if not os.path.isabs(path):
        report["warning"] = "Path is not absolute."

    gdf = gpd.read_file(path)
    geom_types = sorted(list(gdf.geom_type.unique()))
    report.update({
        "crs": str(gdf.crs) if gdf.crs else None,
        "feature_count": int(len(gdf)),
        "geometry_types": geom_types,
        "bounds": {
            "minx": float(gdf.total_bounds[0]),
            "miny": float(gdf.total_bounds[1]),
            "maxx": float(gdf.total_bounds[2]),
            "maxy": float(gdf.total_bounds[3]),
        },
        "fields": {c: str(gdf[c].dtype) for c in gdf.columns if c != gdf.geometry.name},
        "sample_records": gdf.drop(columns=gdf.geometry.name).head(1).to_dict(orient="records")
    })
    return report


def _bbox_intersect(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return not (a["right"] <= b["minx"] or a["left"] >= b["maxx"] or
                a["top"] <= b["miny"] or a["bottom"] >= b["maxy"])


def inspect_geospatial_assets(
    raster_paths: Optional[List[str]] = None,
    vector_paths: Optional[List[str]] = None,
    sample_pixels: int = 0
) -> str:
    """
    Inspect rasters/vectors, collect spatial metadata & basic stats, and run simple alignment checks.
    Returns a JSON string; optionally writes a JSON report to disk.
    """

    report = {
        "raster_reports": [],
        "vector_reports": [],
        "cross_checks": []
    }

    # 在读取栅格前调用
    if raster_paths:
        raster_paths, dedupe_dropped = dedupe_by_name_simple(raster_paths, keep="first")
        report["dedupe_raster"] = {
            "policy": "by_name_no_digits_keep_first",
            "dropped": dedupe_dropped
        }

    vector_paths = vector_paths or []


    # Per-raster reports
    for rp in raster_paths:
        try:
            report["raster_reports"].append(_raster_report(rp, sample_pixels=sample_pixels))
        except Exception as e:
            report["raster_reports"].append({"path": rp, "error": str(e)})

    # Per-vector reports
    for vp in vector_paths:
        try:
            report["vector_reports"].append(_vector_report(vp))
        except Exception as e:
            report["vector_reports"].append({"path": vp, "error": str(e)})

    # Cross checks (first raster vs each vector)
    if report["raster_reports"] and report["vector_reports"]:
        r0 = report["raster_reports"][0]
        r_crs = r0.get("crs")
        r_bounds = r0.get("bounds")
        for vrep in report["vector_reports"]:
            v_crs = vrep.get("crs")
            v_bounds = vrep.get("bounds")
            cc = {
                "raster_path": r0.get("path"),
                "vector_path": vrep.get("path"),
                "crs_match": (r_crs == v_crs) if (r_crs and v_crs) else False,
                "bbox_intersection": _bbox_intersect(r_bounds, v_bounds) if (r_bounds and v_bounds) else None,
                "advice": []
            }
            if not cc["crs_match"]:
                cc["advice"].append("CRS mismatch: reproject vector to raster CRS before analysis.")
            if cc["bbox_intersection"] is False:
                cc["advice"].append("No spatial overlap: verify ROI or clip/align inputs.")
            report["cross_checks"].append(cc)

    # Save JSON if requested
    out_json = json.dumps(report, indent=2, ensure_ascii=False)

    return out_json


# ---- LangChain tool wrapper ----
geodata_inspector_tool = StructuredTool.from_function(
    func=inspect_geospatial_assets,
    name="geodata_inspector_tool",
    description=(
        "Inspect raster/vector spatial metadata and validate alignment. "
        "Reports CRS, resolution, size, dtype, nodata, stats (min/mean/max/std), "
        "vector fields/types, geometry types, feature counts, and bbox. "
        "If both raster(s) and vector(s) are provided, checks CRS match and bbox overlap. "
        "Performs simple name-based deduplication to avoid repeated yearly tiles."
    ),
    args_schema=GeoDataInspectorInput,
)



# result_json = geodata_inspector_tool.run({
#     "raster_paths": [
#         r"C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_NPP-VIIRS-Like_2019.tif",
#         r"C:/NTL_Agent/Night_data/Shanghai/NTL_上海市_NPP-VIIRS-Like_2020.tif"
#     ],
#     "vector_paths": [
#         r"C:/NTL_Agent/report/shp/Shanghai_districts/上海市.shp"
#     ],
#     "sample_pixels": 0,  # 每幅图抽样约1万像素计算统计，0表示全图
# })
#
# print(result_json)
