import rasterio
from rasterio.plot import show
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry.point import Point


def check_multi_poly(gdf):
    return 'MULTI' in gdf.geometry.geom_type.unique()

def remove_inner_polys(polygons):

    # create a new column to store the area of each polygon
    polygons['area'] = polygons.geometry.area

    # sort the polygons by area in descending order
    polygons = polygons.sort_values('area', ascending=False)

    # create a new column to store the union of all polygons larger than the current one
    polygons['union'] = polygons.iloc[1:].geometry.unary_union.cumulative_difference()

    # select only the polygons that are not completely contained within the union of larger polygons
    polygons = polygons[polygons.geometry.apply(lambda x: not x.within(polygons['union'].iloc[-1]))]

    # drop the 'area' and 'union' columns
    polygons = polygons.drop(['area', 'union'], axis=1)

    return polygons

def generate_aoi(raster_path:str, river_areas:gpd.GeoDataFrame, buffer_m=500, 
                 tolerance_m = 250, display=False, remove_inner_poly=False):
    """
    Given a vector containing the full set of river_areas and a target raster, this function generates a simplified
    area of interest around the section of the river within the raster.
    :param raster_path: a string containing the path to the target raster
    :param river_areas: A geopandas dataframe containing the river areas
    :param buffer_deg: The buffer distance in degrees
    :return: A geopandas dataframe containing the simplified area of interest
    """

    # Get the bounds of the raster
    src = rasterio.open(raster_path)
    raster = src.read()
    bounds = src.bounds

    #make sure the river areas are in the same crs as the raster
    river_areas = river_areas.to_crs(src.crs)

    # Clip the river areas to the bounds of the raster
    river_window = river_areas.clip(bounds)
    
    #check if river_window is empty, if so return an empty geodataframe, and raise a warning
    if river_window.empty:
        print('Warning: river window is empty')
        return gpd.GeoDataFrame()
    
     #convert to the closest UTM zone
    river_window_utm = river_window.to_crs(river_window.estimate_utm_crs())


    # Merge all the geometries in the river window
    merged_geometry = unary_union(river_window_utm.geometry)

    # Buffer & simplify the merged geometry to simplify it
    simplified_geometry = merged_geometry.buffer(buffer_m).simplify(tolerance_m)

    # Convert the simplified geometry back to a GeoDataFrame
    simplified_river_window = gpd.GeoDataFrame(geometry=[simplified_geometry], crs=river_window_utm.crs)

    #convert back to the original crs
    simplified_river_window = simplified_river_window.to_crs(src.crs)

    #clip to the bounds
    simplified_river_window = simplified_river_window.clip(bounds)

    if remove_inner_poly: # and check_multi_poly(simplified_river_window):
        remove_inner_polys(simplified_river_window)

    # if display:
    if display:
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        show(raster, ax=ax, transform=src.transform)
        river_window.geometry.boundary.plot(ax=ax, zorder =1)
        simplified_river_window.boundary.plot(ax=ax, zorder=2, color='red')
        #add a scale bar
        #grab the top right corner of the raster and convert it to a Shapely Point
        points = gpd.GeoSeries([Point(src.xy(src.height, src.width)), Point(src.xy(0,0))], crs=src.crs)
        points = points.to_crs(points.estimate_utm_crs())
        distance_meters = points[0].distance(points[1])
        # print(distance_meters)

        scalebar = ScaleBar(distance_meters, location='lower right')
        ax.add_artist(scalebar)
        fig.show()

    return simplified_river_window
