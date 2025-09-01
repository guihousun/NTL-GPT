import pandas as pd
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
import geopandas as gpd
import requests
import json
import os
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from typing import List, Optional
from geopy.geocoders import Nominatim
import numpy as np

# Ensure the API key is set in the environment variables
# os.environ["amap_api_key"] = "your_amap_api_key"
os.environ["amap_api_key"]= "2195d3797b651780acb2b2573179bced"
# Set up URLs for API requests
DISTRICT_URL = 'https://restapi.amap.com/v3/config/district?keywords={city}&key={api_key}'
GEO_JSON_URL = 'https://geo.datav.aliyun.com/areas/bound/{city_code}_full.json'


# --- Updated GetAdministrativeDivisionInput ---
class GetAdministrativeDivisionInput(BaseModel):
    city: str = Field(..., description="Name of the city or administrative division to retrieve data for.")
    save_path: str = Field(
        None,
        description="Path to save the Shapefile. Default is 'C:/NTL_Agent/report/shp/shape_files' if not specified."
    )

def get_administrative_division_data(city: str, save_path: str = None) -> str:
    """
    Function to fetch administrative division data for a specified city, generate a Shapefile, and save it to the provided path.

    Parameters:
    - city (str): Name of the administrative division (e.g., province, city, or county).
    - save_path (str): Optional path to save the Shapefile. Defaults to 'C:/NTL_Agent/report/shp/shape_files'.

    Returns:
    - str: Message indicating the success or failure of the operation and the save location.
    """
    # Get API key from environment variable
    api_key = os.environ.get("amap_api_key")
    if not api_key:
        return "API key is not set. Please set 'amap_api_key' in environment variables."

    # Define default path if not provided
    save_path = save_path or "C:/NTL_Agent/report/shp/shape_files"
    os.makedirs(save_path, exist_ok=True)

    # Step 1: Fetch the district code for the specified city
    district_url = DISTRICT_URL.format(city=city, api_key=api_key)
    print(f"Requesting district code from URL: {district_url}")
    response = requests.get(district_url)
    print(f"Response status code: {response.status_code}")
    if response.status_code != 200:
        return "Failed to fetch district code from Amap API."

    district_data = response.json()
    # Print the district_data for debugging
    print(f"District data: {district_data}")
    try:
        city_code = district_data["districts"][0]["adcode"]
        print(f"City code for {city}: {city_code}")
    except (IndexError, KeyError):
        return "Could not find administrative division code for the specified city."

    # Step 2: Download GeoJSON data
    geo_json_url = GEO_JSON_URL.format(city_code=city_code)
    print(f"GeoJSON URL: {geo_json_url}")
    geojson_file_path = os.path.join(save_path, f"{city}.json")

    if os.path.exists(geojson_file_path):
        print("Reading from local files...")
        with open(geojson_file_path, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    else:
        print("Downloading GeoJSON data from Aliyun...")
        geojson_response = requests.get(geo_json_url)
        print(f"GeoJSON response status code: {geojson_response.status_code}")
        if geojson_response.status_code != 200:
            return "Failed to download GeoJSON data."

        geojson_data = geojson_response.json()
        with open(geojson_file_path, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=4, ensure_ascii=False)

    # Print GeoJSON data keys for debugging
    # print(f"GeoJSON data keys: {geojson_data.keys()}")

    # GCJ-02 to WGS-84 conversion functions
    from math import sin, cos, sqrt, radians

    def gcj02_to_wgs84(lng, lat):
        """
        Convert GCJ-02 coordinates to WGS-84.
        """
        def out_of_china(lng, lat):
            return not (73.66 < lng < 135.05 and 3.86 < lat < 53.55)

        def transformlat(lng, lat):
            ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * sqrt(abs(lng))
            ret += (20.0 * sin(6.0 * lng * 3.1415926) + 20.0 * sin(2.0 * lng * 3.1415926)) * 2.0 / 3.0
            ret += (20.0 * sin(lat * 3.1415926) + 40.0 * sin(lat / 3.0 * 3.1415926)) * 2.0 / 3.0
            ret += (160.0 * sin(lat / 12.0 * 3.1415926) + 320 * sin(lat * 3.1415926 / 30.0)) * 2.0 / 3.0
            return ret

        def transformlng(lng, lat):
            ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * sqrt(abs(lng))
            ret += (20.0 * sin(6.0 * lng * 3.1415926) + 20.0 * sin(2.0 * lng * 3.1415926)) * 2.0 / 3.0
            ret += (20.0 * sin(lng * 3.1415926) + 40.0 * sin(lng / 3.0 * 3.1415926)) * 2.0 / 3.0
            ret += (150.0 * sin(lng / 12.0 * 3.1415926) + 300.0 * sin(lng / 30.0 * 3.1415926)) * 2.0 / 3.0
            return ret

        if out_of_china(lng, lat):
            return lng, lat
        else:
            dlat = transformlat(lng - 105.0, lat - 35.0)
            dlng = transformlng(lng - 105.0, lat - 35.0)
            radlat = lat / 180.0 * 3.1415926
            magic = sin(radlat)
            magic = 1 - 0.00669342162296594323 * magic * magic
            sqrtmagic = sqrt(magic)
            dlat = (dlat * 180.0) / ((6378245.0 * (1 - 0.00669342162296594323)) / (magic * sqrtmagic) * 3.1415926)
            dlng = (dlng * 180.0) / (6378245.0 / sqrtmagic * cos(radlat) * 3.1415926)
            mglat = lat + dlat
            mglng = lng + dlng
            return lng * 2 - mglng, lat * 2 - mglat

    # Function to convert geometry from GCJ-02 to WGS-84
    def gcj02_to_wgs84_geom(geom):
        if geom.is_empty:
            return geom
        geom_type = geom.geom_type
        if geom_type == 'Point':
            x, y = geom.x, geom.y
            x, y = gcj02_to_wgs84(x, y)
            return Point(x, y)
        elif geom_type == 'LineString':
            coords = [gcj02_to_wgs84(x, y) for x, y in geom.coords]
            return LineString(coords)
        elif geom_type == 'Polygon':
            exterior_coords = [gcj02_to_wgs84(x, y) for x, y in geom.exterior.coords]
            interiors = []
            for interior in geom.interiors:
                interior_coords = [gcj02_to_wgs84(x, y) for x, y in interior.coords]
                interiors.append(interior_coords)
            return Polygon(shell=exterior_coords, holes=interiors)
        elif geom_type == 'MultiPolygon':
            polygons = [gcj02_to_wgs84_geom(part) for part in geom.geoms]
            return MultiPolygon(polygons)
        elif geom_type == 'MultiLineString':
            lines = [gcj02_to_wgs84_geom(part) for part in geom.geoms]
            return MultiLineString(lines)
        elif geom_type == 'MultiPoint':
            points = [gcj02_to_wgs84_geom(part) for part in geom.geoms]
            return MultiPoint(points)
        else:
            return geom  # For other geometry types, return as is

    # Step 3: Convert GeoJSON to GeoDataFrame
    try:
        # Create GeoDataFrame from GeoJSON features
        gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])

        # Apply the transformation to the geometry column
        print("Converting coordinates from GCJ-02 to WGS-84...")
        gdf['geometry'] = gdf['geometry'].apply(gcj02_to_wgs84_geom)

        # Set CRS to WGS84
        gdf.set_crs(epsg=4326, inplace=True)
        print("CRS set to EPSG:4326 (WGS84).")

        # Shorten column names to 10 characters or less for Shapefile compatibility
        gdf.columns = [col[:10] for col in gdf.columns]

        # Save GeoDataFrame to Shapefile
        shp_file_path = os.path.join(save_path, f"{city}.shp")
        gdf.to_file(shp_file_path, driver="ESRI Shapefile", encoding="utf-8")
        return f"{city} Shapefile generated successfully. Saved at: {shp_file_path}"
    except Exception as e:
        return f"Failed to generate Shapefile: {e}"

# Update the StructuredTool for administrative division data retrieval
get_administrative_division_tool = StructuredTool.from_function(
    get_administrative_division_data,
    name="get_administrative_division_Amap_tool",
    description=(
        "Fetches administrative division data (province, city, county) for a specified location using the Amap API. "
        "Converts the data to a Shapefile format in WGS-84 coordinate system and saves it locally. "
        "If no save path is provided, the Shapefile will be saved to the default path 'C:/NTL_Agent/report/shp/shape_files'. "
        "Note: For administrative division boundaries outside China, please use 'get_administrative_division_osm_tool'."
        "### Example Usage:\n"
        "- city: '北京'\n"
        "- save_path: 'C:/NTL_Agent/report/shp/shape_files'\n\n"
    ),
    input_type=GetAdministrativeDivisionInput,
)


import os
import geopandas as gpd
import osmnx as ox

def get_administrative_division_osm(place_name: str, save_path: str = None) -> str:
    """
    Fetch administrative boundary data for a given place from OSM using geocode_to_gdf
    and save as Shapefile.

    Parameters:
    - place_name (str): Name of the administrative area (e.g., 'Myanmar', 'Yangon', 'Mandalay').
    - save_path (str): Path to save the Shapefile. Defaults to 'C:/NTL_Agent/report/shp/shape_files'.

    Returns:
    - str: Message indicating the success or failure of the operation and the save location.
    """
    save_path = save_path or "C:/NTL_Agent/report/shp/shape_files"
    os.makedirs(save_path, exist_ok=True)

    try:
        print(f"Fetching OSM boundary for '{place_name}' ...")
        gdf = ox.geocode_to_gdf(place_name, which_result=1)
        gdf = gdf.to_crs(epsg=4326)

        # 截短字段名以适配 Shapefile 格式
        gdf.columns = [col[:10] for col in gdf.columns]

        # 保存为 Shapefile
        shp_file_path = os.path.join(save_path, f"{place_name.replace(' ', '_')}.shp")
        gdf.to_file(shp_file_path, driver="ESRI Shapefile", encoding="utf-8")

        return f"{place_name} administrative boundary Shapefile generated successfully. Saved at: {shp_file_path}"

    except Exception as e:
        return f"Failed to fetch administrative division data for {place_name}: {e}"


# 输入模型
class GetAdministrativeDivisionOSMInput(BaseModel):
    place_name: str = Field(..., description="Name of the city or administrative division to retrieve data for (e.g., 'Myanmar').")
    save_path: str = Field(
        None,
        description="Path to save the Shapefile. Default is 'C:/NTL_Agent/report/shp/shape_files' if not specified."
    )


# 转换为 StructuredTool
get_administrative_division_osm_tool = StructuredTool.from_function(
    get_administrative_division_osm,
    name="get_administrative_division_osm_tool",
    description=(
        "Fetches administrative division boundaries (country, province, city) from OpenStreetMap "
        "using OSMnx's geocode_to_gdf function, which queries Nominatim for the specified place "
        "and retrieves its administrative boundary geometry. Saves the result as an ESRI Shapefile "
        "in WGS-84 coordinate system. If no save path is provided, the file will be saved to the "
        "default directory 'C:/NTL_Agent/report/shp/shape_files'.\n\n"
        "Note: For administrative division boundaries within China, please use 'get_administrative_division_Amap_tool'."
        "### Example Usage:\n"
        "- place_name: 'Myanmar'\n"
        "- save_path: 'C:/NTL_Agent/report/shp/shape_files'\n\n"
    ),
    input_type=GetAdministrativeDivisionOSMInput,
)

# --- Updated ReverseGeocodeInput ---
class ReverseGeocodeInput(BaseModel):
    latitudes: List[float] = Field(..., description="List of latitudes for the locations to reverse geocode.")
    longitudes: List[float] = Field(..., description="List of longitudes for the locations to reverse geocode.")
    save_path: Optional[str] = Field(
        None,
        description="Path to save the CSV file. If not specified, saves to 'C:/NTL_Agent/report/csv/geocode_results.csv'"
    )
    region: str = Field("China", description="the region in 'China' or 'other_country'")

def reverse_geocode(
    latitudes: List[float],
    longitudes: List[float],
    save_path: Optional[str] = None,
    region: str = "China"
) -> str:
    """
    Performs reverse geocoding for a list of latitude and longitude pairs.

    Parameters:
    - latitudes (List[float]): List of latitudes.
    - longitudes (List[float]): List of longitudes.
    - save_path (Optional[str]): Path to save the CSV file. Defaults to 'C:/NTL_Agent/report/geocode_results/geocode_results.csv'.

    Returns:
    - str: Message indicating the success of the operation and the save location.
    """
    addresses = []
    save_path = save_path or "C:/NTL_Agent/report/csv/reverse_geocode_results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    amap_api_key = os.environ.get("amap_api_key")

    if region == "China" :
        # Use Amap API for reverse geocoding
        for latitude, longitude in zip(latitudes, longitudes):
            url = f"https://restapi.amap.com/v3/geocode/regeo?location={longitude},{latitude}&key={amap_api_key}&extensions=base"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "1":
                    address = data.get("regeocode", {}).get("formatted_address", "Address not found")
                else:
                    address = f"Amap API error: {data.get('info')}"
            else:
                address = f"Failed to request Amap API. HTTP status code: {response.status_code}"
            addresses.append(address)
    else:
        # Use Nominatim for reverse geocoding
        geolocator = Nominatim(user_agent="your_app_name")
        for latitude, longitude in zip(latitudes, longitudes):
            location = geolocator.reverse((latitude, longitude))
            address = location.address if location else "Address not found"
            addresses.append(address)
    # Save results to CSV file
    df = pd.DataFrame({"Latitude": latitudes, "Longitude": longitudes, "Address": addresses})
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    # return f"Reverse geocoding completed. Results saved at: {save_path}"
    return f"Reverse geocoding completed. Results saved at: {save_path}\n\nTop rows:\n{df.head().to_string()}"

# Update the StructuredTool for reverse geocoding
reverse_geocode_tool = StructuredTool.from_function(
    reverse_geocode,
    name="reverse_geocode_tool",
    description=(
        "This tool performs reverse geocoding for a list of latitudes and longitudes, returning the corresponding full addresses. "
        "### Input Example:\n"
        "- latitudes: [40.748817, 34.052235]\n"
        "- longitudes: [-73.985428, -118.243683]\n"
        "- save_path: 'C:/NTL_Agent/report/csv/reverse_geocode_results.csv'\n\n"
        "- region: 'China' or 'other_country'"
        "### Output:\n"
        "Returns the addresses corresponding to the provided latitudes and longitudes, and saves the results to a CSV file."
    ),
    input_type=ReverseGeocodeInput,
)

# --- Updated POISearchInput ---
class POISearchInput(BaseModel):
    latitude: float = Field(..., description="Central point latitude.")
    longitude: float = Field(..., description="Central point longitude.")
    radius: int = Field(500, description="Search radius in meters. Default is 500 meters.")
    types: Optional[str] = Field(None, description="POI category codes, refer to Amap POI category code table.")
    save_path: Optional[str] = Field(
        None,
        description="Path to save the CSV file. If not specified, saves to 'C:/poi_results/poi_results.csv'"
    )

def search_poi_nearby(
    latitude: float,
    longitude: float,
    radius: int = 500,
    types: Optional[str] = None,
    save_path: Optional[str] = None
) -> str:
    """
    Searches for Points of Interest (POIs) around a given coordinate and saves the results to a CSV file.

    Parameters:
    - latitude (float): Latitude of the central point.
    - longitude (float): Longitude of the central point.
    - radius (int): Search radius in meters. Default is 500.
    - types (Optional[str]): POI category codes. Refer to Amap POI category code table.
    - save_path (Optional[str]): Path to save the CSV file. Defaults to 'C:/poi_results/poi_results.csv'.

    Returns:
    - str: Message indicating the success of the operation and the save location.
    """
    save_path = save_path or "C:/poi_results/poi_results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    amap_api_key = os.environ.get("amap_api_key")
    if not amap_api_key:
        return "API key is not set. Please set 'amap_api_key' in environment variables."

    url = "https://restapi.amap.com/v5/place/around"
    params = {
        "key": amap_api_key,
        "location": f"{longitude},{latitude}",
        "radius": radius,
        "types": types,
        "output": "json"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "1":
            pois = data.get("pois", [])
            # Save POI results to CSV file
            df = pd.DataFrame(pois)
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            # return f"POI search completed. Results saved at: {save_path}"
            # 将前 5 行数据转换为字符串并包含在返回信息中
            return f"POI search completed. Results saved at: {save_path}\n\nTop rows:\n{df.head().to_string()}"
        else:
            return f"Amap API error: {data.get('info')}"
    else:
        return f"Failed to request Amap API. HTTP status code: {response.status_code}"

# Update the StructuredTool for POI search
poi_search_tool = StructuredTool.from_function(
    search_poi_nearby,
    name="poi_search_tool",
    description=(
        "This tool retrieves Points of Interest (POIs) within a specified radius around a given coordinate and saves the results to a CSV file. "
        "It uses Amap's POI search API. The API key is read from the environment variable 'amap_api_key'. "
        "With the retrieved latitude and longitude of the target pixel, this tool can obtain nearby POI information, "
        "helping to determine the main types of facilities in the area. This facilitates further analysis and interpretation "
        "of the context within each grid cell.\n\n"
        "### Input Example:\n"
        "- latitude: 39.984154\n"
        "- longitude: 116.307490\n"
        "- radius: 500\n"
        "- types: '050000' (Restaurant services)\n"
        "- save_path: 'C:/NTL_Agent/report/csv/poi_results.csv'\n\n"
        "### Output:\n"
        "Returns a list of POIs containing name, address, type, and other information, saving the results to the specified CSV file. "
        "This information supports further analysis by providing insight into the types of nearby facilities around the selected grid cell."
    ),
    input_type=POISearchInput,
)


# --- Updated GeocodeInput ---
class GeocodeInput(BaseModel):
    address: str = Field(..., description="The address to geocode.If in China,")
    save_path: Optional[str] = Field(
        None,
        description="Path to save the CSV file. If not specified, saves to 'C:/NTL_Agent/report/csv/geocode_results.csv'"
    )

def geocode_address(
    address: str,
    save_path: Optional[str] = None
) -> str:
    """
    Geocodes an address using the Amap API, returning latitude and longitude.

    Parameters:
    - address (str): Address to geocode.
    - save_path (Optional[str]): Path to save the CSV file. Defaults to 'C:/NTL_Agent/report/geocode_results/geocode_results.csv'.

    Returns:
    - str: Message indicating the success of the operation and the save location.
    """
    save_path = save_path or "C:/NTL_Agent/report/csv/geocode_results.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    amap_api_key = os.environ.get("amap_api_key")
    if not amap_api_key:
        return "API key is not set. Please set 'amap_api_key' in environment variables."

    url = "https://restapi.amap.com/v3/geocode/geo"
    params = {
        "key": amap_api_key,
        "address": address,
        "output": "json"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "1":
            geocode_info = data.get("geocodes", [])[0]
            location = geocode_info.get("location", "").split(",")
            latitude = float(location[1])
            longitude = float(location[0])
            # Save result to CSV
            df = pd.DataFrame([{"Address": address, "Latitude": latitude, "Longitude": longitude}])
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            # 打印出前 5 行数据
            # print(df.head())  # 默认显示前 5 行
            # 将前 5 行数据转换为字符串并包含在返回信息中
            return f"Geocoding completed. Results saved at: {save_path}\n\nTop rows:\n{df.head().to_string()}"
        else:
            return f"Amap API error: {data.get('info')}"
    else:
        return f"Failed to request Amap API. HTTP status code: {response.status_code}"

# Update the StructuredTool for geocoding
geocode_tool = StructuredTool.from_function(
    geocode_address,
    name="geocode_tool",
    description=(
        "This tool geocodes a given address, retrieving latitude and longitude using the Amap API, "
        "and saves the result to a CSV file. The API key is read from the environment variable 'amap_api_key'. "
        "With the obtained geographic coordinates, further analysis can be conducted, such as examining nighttime light "
        "intensity values at the specified latitude and longitude for the given indicator.\n\n"
        "Address can only in China and must be Chinese Name"
        "### Input Example:\n"
        "- address: '上海市静安区南京西路'\n"
        "- save_path: 'C:/NTL_Agent/report/csv/geocode_results.csv'\n\n"
        "### Output:\n"
        "Returns latitude and longitude for the specified address, saving the result to the specified CSV file. "
    ),
    input_type=GeocodeInput,
)



# # Sample usage
# if __name__ == "__main__":
#     # Ensure the API key is set
#     # os.environ["amap_api_key"] = "your_actual_amap_api_key"
#
#     # Example usage of get_administrative_division_tool
#     result = get_administrative_division_tool.func(
#         city="上海市",
#         save_path="./test"
#     )
#     print(result)

#     # Example usage of reverse_geocode_tool
#     reverse_geocode_result = reverse_geocode_tool.func(
#         latitudes=[31.2304],
#         longitudes=[121.4737],
#         save_path="C:/NTL_Agent/report/geocode_results/geocode_results.csv"
#     )
#     print(reverse_geocode_result)
#
#     # Example usage of poi_search_tool
#     poi_search_result = poi_search_tool.func(
#         latitude=31.2397,
#         longitude=121.4903,
#         types='050000',  # Restaurant services
#         save_path="C:/NTL_Agent/report/poi_results.csv"
#     )
#     print(poi_search_result)

# Sample usage of geocode_tool
# if __name__ == "__main__":
#     # Example usage of geocode_tool
#     geocode_result = geocode_tool.func(
#         address="上海市东方明珠",
#         save_path="C:/NTL_Agent/report/geocode_results/geocode_results.csv"
#     )
#     print(geocode_result)

# get_administrative_division_tool.func(city="Myanmar")
# result = get_administrative_division_osm("Myanmar", admin_level=2)
# print(result)
# 省级: get_administrative_division_osm("Yangon Region, Myanmar", admin_level=4)
# 市级: get_administrative_division_osm("Yangon, Myanmar", admin_level=8)

# get_administrative_division_osm_tool.func(place_name= "Shanghai",  save_path="C:/NTL_Agent/report/shp/china")