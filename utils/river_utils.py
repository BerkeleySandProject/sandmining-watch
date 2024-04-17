import rasterio
from rasterio.plot import show
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.geometry.point import Point
from shapely.geometry import shape, Polygon
from rasterio.features import geometry_mask
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

kms_per_radian = 6371.0088

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

def square_at_point(point, size_m, crs):
    """
    Given a point, generate a square polygon of size size_m around that point
    :param point: A shapely Point object
    :param size_m: The size of the square in meters
    :param crs: The crs of the point
    :return: A shapely Polygon object
    """
    #convert the point to UTM
    utm_crs = point.estimate_utm_crs()
    point_utm = point.to_crs(utm_crs)

    #get the x and y coordinates
    x = point_utm.x
    y = point_utm.y

    #calculate the half width and height
    half_width = size_m / 2
    half_height = size_m / 2

    #create the corners of the square
    top_left = Point(x - half_width, y + half_height)
    top_right = Point(x + half_width, y + half_height)
    bottom_left = Point(x - half_width, y - half_height)
    bottom_right = Point(x + half_width, y - half_height)

    #create the square
    square = gpd.GeoDataFrame(geometry=[Polygon([top_left, top_right, bottom_right, bottom_left])], crs=utm_crs)

    #convert back to the original crs
    square = square.to_crs(crs)

    return square

def generate_aoi(raster_path:str, river_areas:gpd.GeoDataFrame, buffer_m=500, 
                 tolerance_m = 250, display=False, remove_inner_poly=False, squares_to_draw=[960, 2560]):
    """
    Given a vector containing the full set of river_areas and a target raster, this function generates a simplified
    area of interest around the section of the river within the raster.
    :param raster_path: a string containing the path to the target raster
    :param river_areas: A geopandas dataframe containing the river areas
    :param buffer_m: The buffer distance in meters
    :param tolerance_m: The tolerance distance in meters
    :param display: A boolean indicating whether or not to display the simplified area of interest
    :param remove_inner_poly: A boolean indicating whether or not to remove inner polygons (doesn't work well)
    :param gdf_to_draw: A list of geopandas dataframes to draw on top of the plot
    :return: A geopandas dataframe containing the simplified area of interest
    """

    # Get the bounds of the raster
    src = rasterio.open(raster_path)
    
    raster = src.read()
    bounds = src.bounds
    #create a polygons from the bounds
    bounds_poly = gpd.GeoDataFrame(geometry=[Polygon([(bounds.left, bounds.top), (bounds.right, bounds.top), (bounds.right, bounds.bottom), (bounds.left, bounds.bottom)])], crs=src.crs)
    centroid = bounds_poly.geometry.centroid
    # print(bounds_poly.geometry.centroid.x)

    #create a list of squares to draw
    gdf_to_draw = []

    #sort the squares to draw in ascending order
    squares_to_draw.sort()

    for square in squares_to_draw:
        gdf_to_draw.append(square_at_point(centroid, square, src.crs))

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
    simplified_geometry = merged_geometry.simplify(tolerance_m).buffer(buffer_m).simplify(tolerance_m)
    # simplified_geometry = merged_geometry.buffer(buffer_m).simplify(tolerance_m)

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
        # Add 2 points separated by a degree in the longitude direction, which can be represented in meters
        a, b = Point(centroid.x[0], centroid.y[0]), Point(centroid.x[0] + 1, centroid.y[0])

        points = gpd.GeoSeries([a, b], crs=src.crs)
        points = points.to_crs(points.estimate_utm_crs())
        distance_meters = points[0].distance(points[1])

        scalebar = ScaleBar(distance_meters, location='lower right')
        # scalebar = ScaleBar(1, location='lower right')
        ax.add_artist(scalebar)

        #iterate over polys_to_draw and draw them
        for i, poly in enumerate(gdf_to_draw):
            if i == 0:
                c = 'yellow'
            else:
                c = 'white'
            poly.geometry.boundary.plot(ax=ax, color=c, linestyle=':' , zorder=3)


        fig.show()

    return simplified_river_window


def get_rivers_above(rivers_gdf: gpd.GeoDataFrame, boundary_gdf: gpd.GeoDataFrame = None, subset_above_length_km=50.0):
    """
    Given a geopandas dataframe containing rivers and a geopandas dataframe containing a boundary, this function
    returns a geopandas dataframe containing only the rivers that are within the boundary and are above a certain
    length
    :param rivers_gdf: A geopandas dataframe containing rivers
    :param boundary_pdf: A geopandas dataframe containing a boundary
    :param subset_above_length_km: A float containing the minimum length of rivers to keep
    :return: A geopandas dataframe containing the rivers that are within the boundary and are above a certain length
    """

    # Clip the rivers to the boundary

    if boundary_gdf is not None:
        rivers_gdf = rivers_gdf.clip(boundary_gdf)

    # Make sure the rivers are in a UTM crs so it's in meters
    rivers_subset = rivers_gdf.to_crs(rivers_subset.estimate_utm_crs())

    # Filter the rivers to only those above a certain length
    rivers_subset['river_length_m'] = rivers_subset.geometry.length
    #sort rivers by length in descending order
    rivers_subset = rivers_subset.sort_values(by='river_length_m', ascending=False)
    rivers_above_km = rivers_subset[rivers_subset.river_length_m > subset_above_length_km*1000.].copy().to_crs(rivers_gdf.crs) #go back to WGS84

    return rivers_above_km


def cluster_observations(gdf: gpd.GeoDataFrame, min_cluster_size=2, max_cluster_size=5,maximum_cluster_radius_km=None):
    """
    Given a geopandas dataframe containing observations, this function clusters the observations using HDBSCAN
    
    :param gdf: A geopandas dataframe containing observations
    :param min_cluster_size: The minimum number of points in a cluster
    :param max_cluster_size: The maximum number of points in a cluster
    :param maximum_cluster_radius_km: The maximum radius of a cluster in kilometers -> this is the epsilon parameter in HDBSCAN
    :return: A geopandas dataframe containing the observations and a column containing the cluster id
    """

    from sklearn.cluster import HDBSCAN  #requires scikit-learn 1.3.2+

    #drop all rows with NaNs in geometry
    gdf = gdf.dropna(subset=['geometry'])
    
    coords = np.array([gdf.geometry.y, gdf.geometry.x]).T

    if maximum_cluster_radius_km is not None:
        epsilon = maximum_cluster_radius_km / kms_per_radian
    else:
        epsilon = 0.

    hdb = HDBSCAN(min_cluster_size=min_cluster_size, 
                  max_cluster_size=max_cluster_size,
                  metric="haversine",
                  cluster_selection_epsilon=epsilon,
                  ).fit(np.radians(coords))

    count = 0

    gdf['cluster_id'] = -99
    for i, cluster in enumerate(hdb.labels_):
        gdf['cluster_id'].iloc[i] = cluster
        if cluster == -1:
            count += 1

    print('Number of clusters: {} / Number of unassigned points : {}'.format(len(set(hdb.labels_)), count))

    return gdf

def visualize_clusters(gdf: gpd.GeoDataFrame, rivers_gdf:gpd.GeoDataFrame = None, background_gdfs=[]):

    #Keep only the rivers from the rivers_gdf that are aligned with the observations in gdf
    gdf_sample_buffer = gdf.copy()
    gdf_sample_buffer['geometry'] = gdf.buffer(0.05)


    fig, ax = plt.subplots(figsize=[15, 15])
    for i, background in enumerate(background_gdfs):
        cm = 'tab20'
        background.plot(ax=ax, cmap=cm, alpha=0.3)

    # Perform the spatial join with the buffered points
    if rivers_gdf is not None:
        rivers_sample = gpd.sjoin(rivers_gdf, gdf_sample_buffer, how="inner")
        rivers_sample.plot(ax=ax, linewidth=2.5, color='blue', facecolor='blue' )

    gdf.plot(ax=ax, column='cluster_id', legend=False, cmap='Paired', markersize=200, edgecolor='black', marker='o', alpha=0.6)


    # Annotate the cluster_id over every point in gdf_sample
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.cluster_id):
        ax.annotate(label, xy=(x, y), xytext=(-3, -3), textcoords="offset points")


    plt.show()

