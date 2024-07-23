import os, sys

sys.path.insert(0, os.path.abspath('..'))
os.environ['USE_PYGEOS'] = '0'

# %load_ext autoreload
# %autoreload 2

from dotenv import load_dotenv
load_dotenv()

import subprocess
import json
    
import geemap, ee
import pandas as pd
from shapely.geometry import shape
from shapely.geometry import Polygon
import uuid
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from babelgrid import Babel

from google.cloud import storage
from google.oauth2 import service_account
import json

import gc
gc.enable()

from rasterio.plot import show
import rasterio
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
import glob
import concurrent.futures
import shutil
from google.api_core import retry
# from IPython.display import clear_output


# ee.Authenticate()
project_id = 'gee-sand'
    
# path to conda env with GDAL installed
# ENVBIN = sys.exec_prefix
ENVBIN_GDAL = f"{os.environ['HOME']}/.conda/envs/rv-21"


GLOBAL_BATCH_SIZE = 35
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
GLOBAL_CHUNK_SIZE = 35 ## inGB
GLOBAL_TMP_DIR = "/data/sand_mining/inference/tmp"

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("start_date", help="start date of period in dd-mm-yyyy format")
parser.add_argument("end_date", help="end date of period in dd-mm-yyyy format")
parser.add_argument("-c", "--collection", help="Name of Sentinel Collection. Currently only supports S2-HARMONIZED", default = "S2_HARMONIZED")
parser.add_argument('-r', '--rivernames', nargs='+', default=[], help="provide river names to process")
parser.add_argument('-o', '--overwrite', default = False)
# parser.add_argument('-a', '--all_rivers', help="load all river names",
#                     action="store_true")
args = parser.parse_args()


def optimal_smooth_fishnet(fishnet_grid, 
                          max_memory, padding = 0.05):
    """
    Given a Fishnet grid, combine cells to generate a new polygon (square or rectangle), such that 
    polygon occupies a max of the given memory
    
    """
    # print("Merging!", datetime.now())
    
    #Create some variables
    cells = []
    polygons = []
    
    add_geom = True
    last_polygon= None
    cell_add = True
    
    geoms = fishnet_grid.geometry.values
    
    fish_minx, fish_miny, fish_maxx, fish_maxy =  fishnet_grid.total_bounds #Don't need this, delete

    #### Starting with the left most cell, grow a polygon. 
    ### 
    for geom_id, geom in enumerate(geoms):
        
        geom_centroid = geom.centroid
        geom_minx, geom_miny, geom_maxx, geom_maxy = geom.bounds
        
        ##This snippet below is my current hack to prevent overlaps. 
        ### Overlaps can still occur in portions where the river meanders a lot. 
        
        if (last_polygon is not None) & (geom_centroid.within(last_polygon)):
            cell_add = False
        else:
            cell_add = True

        if len(cells) > 0:

            ### keep checking the area of the grown polygon if the current list of cells is not empty
            tmp = gpd.GeoDataFrame(geometry = cells)
            minx, miny, maxx, maxy = tmp.total_bounds

            
            RES = 8.983152841195215e-05
            l = np.round((maxx - minx)/RES)
#             w = np.round((fish_maxy - fish_miny)/RES)
            w = np.round(maxy - miny)/RES
            est_memory =  l * w * 8 * 12/1e9

            
            if (est_memory > max_memory):
#                 polygon = Polygon([(minx, fish_miny), (minx, fish_maxy), (maxx, fish_maxy), (maxx, fish_miny)])
                polygon = Polygon([(minx - padding, miny - padding), 
                                   (minx - padding, maxy + padding), 
                                   (maxx + padding, maxy + padding), 
                                   (maxx + padding, miny - padding)])
                polygons.append(polygon)
                last_polygon  = polygon
                new_minx = maxx
                new_miny = maxy
                add_geom = False
                cells = []

        if (add_geom) & (cell_add):
            cells.append(geom)

        if len(cells) == 0:
            add_geom = True
            
    polygon = Polygon([(minx - padding, miny - padding), 
                   (minx - padding, maxy + padding), 
                   (maxx + padding, maxy + padding), 
                   (maxx + padding, miny - padding)])
    polygons.append(polygon)
    return polygons


def combine_chunks(folderpath, rivername, destination):
    basefiles = os.listdir(folderpath)
    with rasterio.open(f"{folderpath}/{basefiles[0]}") as src:
        N_BANDS = src.count

    bandstr = " ".join([f"-b {i+1}" for i in range(N_BANDS)]) + f" -mask {N_BANDS + 1}"

    #### Build Virtual Raster
    try:
        subprocess.run(f"gdalbuildvrt -addalpha mosaic.vrt {folderpath}/*.tif", 
                       cwd = f"{ENVBIN_GDAL}/bin", 
                       shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Building VRT Failed! : {e}")
    
    cog_create = f"""gdal_translate {bandstr} \
    -co TILED=YES \
    -co BIGTIFF=YES \
    -co COMPRESS=DEFLATE \
    -co NUM_THREADS=ALL_CPUS \
    --config GDAL_CACHEMAX 4096 \
    mosaic.vrt {folderpath}/tmp.tif"""
    try:
        print("Masking...")
        subprocess.run(cog_create,
                   cwd = f"{ENVBIN_GDAL}/bin",
                   shell=True)
    except subprocess.CalledProcessError as e:
        print(e)
        
    
#     #Build overviews
    cog_cmd = f"""gdaladdo -r nearest --config BIGTIFF_OVERVIEW YES --config GDAL_CACHEMAX 4096 --config COMPRESS_OVERVIEW DEFLATE --config BIGTIFF YES --config GDAL_NUM_THREADS ALL_CPUS {folderpath}/tmp.tif 2 4 8 16 32"""
    try:
        print("Building overviews")
        subprocess.run(cog_cmd,
                   cwd = f"{ENVBIN_GDAL}/bin",
                   shell=True)
    except subprocess.CalledProcessError as e:
        print(e)
        
    
    #### create final cog
    cog_cmd = f"""gdal_translate \
    {folderpath}/tmp.tif {destination} \
    -co BIGTIFF=YES \
    -co TILED=YES \
    -co COMPRESS=DEFLATE \
    -co COPY_SRC_OVERVIEWS=YES \
    -co BLOCKXSIZE=512 \
    -co BLOCKYSIZE=512 \
    -co NUM_THREADS=ALL_CPUS \
    --config GDAL_CACHEMAX 4096"""

    try:
        print("Saving COG")
        subprocess.run(cog_cmd,
                   cwd = f"{ENVBIN_GDAL}/bin",
                   shell=True)
    except subprocess.CalledProcessError as e:
        print(e)

    shutil.rmtree(folderpath)


    
class S2ImageExtractor:
    def __init__(self, tiles, start_date, end_date, bands, collection = 'S2_HARMONIZED', 
                imgformat = 'GEOTIFF', rivername = 'test'):
        self.tiles = tiles
        self.start_date = start_date
        self.end_date = end_date
        self.bands = bands
        self.imgformat = imgformat
        self.collection = collection
        self.rivername = rivername
        
        ee.Initialize(
                        url="https://earthengine-highvolume.googleapis.com",
                        project=project_id,
                    )
        self.s2composite = ee.ImageCollection(f"COPERNICUS/{collection}")\
                            .filterDate(start_date, end_date)\
                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)).median()
        self.batch_size = GLOBAL_BATCH_SIZE

    @retry.Retry(timeout=240)
    def get_tile(self, aoi):
    
        s2_img = self.s2composite.clipToBoundsAndScale(
          geometry=aoi, 
         scale = 10)
        pixels = ee.data.computePixels(
            {
                "bandIds": self.bands,
                "expression": s2_img,
                "fileFormat": self.imgformat,
                #'grid': {'crsCode': tile.crs} this was causing weird issues
            }
        )
        return pixels

   
    def get_chips(self, tiles_to_process = []):
        chips = []
        if len(tiles_to_process) == 0:
            local_tiles = self.tiles
        else:
            local_tiles = self.tiles[self.tiles['id'].isin(tiles_to_process)]
            
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(0, len(local_tiles)+1, self.batch_size):
                batch_tiles = local_tiles[i : i + self.batch_size]
        
                futures = [executor.submit(self.get_tile, geemap.geopandas_to_ee(batch_tiles[batch_tiles['id'] == tile_id][['geometry']]))
                               for tile_id in batch_tiles['id'].unique()]
        
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    pixels = result
                    chips.append(pixels)
                    # tile_data.append(tile)
        
        return chips

    def chunk_and_merge_tiles(self):
        global dump

        df_grids = self.tiles.copy()
        df_grids['leftmost_x'] = df_grids['geometry'].apply(lambda geom: geom.exterior.xy[0][0])
        df_grids = df_grids.sort_values('leftmost_x')
        
        grid_chunks = optimal_smooth_fishnet(df_grids, GLOBAL_CHUNK_SIZE)
        if len(grid_chunks) == 1:

            chips = self.get_chips()
            openchips = [MemoryFile(c).open() for c in chips]
            outimg, out_trans = merge(openchips)

            pth = f"/data/sand_mining/inference/inputs/{self.collection}/{self.start_date}_{self.end_date}"
            if not os.path.exists(pth):
                os.makedirs(pth)
                
            with rasterio.open(pth + "/" + "tmp.tif", 'w', 
                       driver='GTiff', 
                       height=outimg.shape[1], 
                       width=outimg.shape[2], 
                       count=outimg.shape[0],
                       dtype=outimg.dtype, 
                       crs="EPSG:4326", 
                       transform=out_trans) as dest:
                dest.write(outimg)

            basepth = f"{pth}/tmp.tif"
            destpth = f"{pth}/{self.rivername}_{self.start_date}_{self.end_date}.tif"

            ##Build overviews
            cog_cmd = f"""gdaladdo -r nearest --config BIGTIFF_OVERVIEW YES --config GDAL_CACHEMAX 4096 \
                        --config COMPRESS_OVERVIEW DEFLATE --config BIGTIFF YES --config GDAL_NUM_THREADS \
                        ALL_CPUS {pth}/tmp.tif 2 4 8 16 32"""
            try:
                print("Building overviews")
                subprocess.run(cog_cmd,
                           cwd = f"{ENVBIN_GDAL}/bin",
                           shell=True)
            except subprocess.CalledProcessError as e:
                print(e)
                
            #### create final cog
            cog_cmd = f"""gdal_translate \
            {basepth} {destpth} \
            -co BIGTIFF=YES \
            -co TILED=YES \
            -co COMPRESS=DEFLATE \
            -co COPY_SRC_OVERVIEWS=YES \
            -co BLOCKXSIZE=512 \
            -co BLOCKYSIZE=512 \
            -co NUM_THREADS=ALL_CPUS \
            --config GDAL_CACHEMAX 4096"""
            
            try:
                print("Saving COG")
                subprocess.run(cog_cmd,
                           cwd = f"{ENVBIN_GDAL}/bin",
                           shell=True)
            except subprocess.CalledProcessError as e:
                print(e)

            os.remove(basepth)
        else:
            df_grid_chunks = gpd.GeoDataFrame(geometry = grid_chunks, crs = "EPSG:4326")
            df_grid_chunks = df_grid_chunks.reset_index()
            df_grid_chunks = df_grid_chunks.rename(columns = {'index':'chunk_id'})

            df_chunks = gpd.sjoin(df_grids, df_grid_chunks, predicate = 'within')
            
            pth = f"{GLOBAL_TMP_DIR}/{self.rivername}_{self.start_date}_{self.end_date}"
            print(pth)
            if not os.path.exists(pth):
                os.makedirs(pth)
                print("TMP PATH CREATED", pth)
                
            for chunk_id in df_chunks['chunk_id'].unique():
                print(f"Processing Chunk {chunk_id}")
                chunk_data = df_chunks[df_chunks['chunk_id'] == chunk_id]
                ids = chunk_data['id'].unique()
                chips = self.get_chips(ids)
                
                openchips = [MemoryFile(c).open() for c in chips]
                outimg, out_trans = merge(openchips)
                # dump.extend([outimg, out_trans])
                
                with rasterio.open(pth + "/" + f"{self.rivername}_{self.start_date}_{self.end_date}_{chunk_id}.tif", 'w', 
                       driver='GTiff', 
                       height=outimg.shape[1], 
                       width=outimg.shape[2], 
                       count=outimg.shape[0],
                       dtype=outimg.dtype, 
                       crs="EPSG:4326", 
                       transform=out_trans) as dest:
                    dest.write(outimg)

            base = "/data/sand_mining/inference/inputs"
            destpth = f"""{base}/{self.collection}/{self.start_date}_{self.end_date}/{self.rivername}_{self.start_date}_{self.end_date}.tif"""
            combine_chunks(pth, self.rivername, destpth)


outdir = f"/data/sand_mining/inference/inputs/{args.collection}/{args.start_date}_{args.end_date}"
if not os.path.exists(outdir):
    os.makedirs(outdir)

# print(args.overwrite)

if len(args.rivernames) > 0:
    for river in args.rivernames:
        assert os.path.exists(f"/data/sand_mining/rivers/river_grids/wris/{river}.geojson"), "River grid file not found!!"
        base = "/data/sand_mining/inference/inputs"
        destpth = f"""{base}/{args.collection}/{args.start_date}_{args.end_date}/{river}_{args.start_date}_{args.end_date}.tif"""
        if (args.overwrite) | (not os.path.exists(destpth)):
            # rivername = river.split(".")[0]
            print(river, datetime.now())
            tiles = gpd.read_file(f"/data/sand_mining/rivers/river_grids/wris/{river}.geojson")
            tiles = tiles.drop(columns = ['index_right', 'index'])
            
            gc.collect()
            test = S2ImageExtractor(tiles, args.start_date, args.end_date, collection = args.collection, bands = S2_BANDS, rivername=river)
            test.chunk_and_merge_tiles()
        else:
            print(river, "Done", datetime.now())
else:
    sheet_id = "1Ov1M_zsb5jYo_dtIjUXco1wvNgasmdoZKtJ877psHL4"
    sheet_name = "Ganga"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
            
    df_rivers = pd.read_csv(url)
    # df_rivers['osm_id'] = df_rivers['osm_id'].astype('float')
    
    
    df_rivers['rivname_clean'] = df_rivers['river'].replace({
        'Ghaghara':'Ghaghra', 
     'Chhoti Sarju':'Choti-Sarju', 
      'Gopat':'Gopad', 
      'Hindon':'Hindan', 
       'Parwati':'Parbati', 
     'Sahibi / Sabi Nadi':'Sahibi-Sabi', 
     'Sindh':'Sind', 
      'Sone':'Son', 
      'Gomti':'Gomati', 
       'Bagmati':'Baghmati', 
      'North Koel':'North-Koel', 
     'Burhi Gandak':'Burhi-Gandak'
                                                                                                                                  
    })
    
    river_names = np.sort(df_rivers['rivname_clean'].unique())

    for river in river_names:
        base = "/data/sand_mining/inference/inputs"
        destpth = f"""{base}/{args.collection}/{args.start_date}_{args.end_date}/{river}_{args.start_date}_{args.end_date}.tif"""
        if (args.overwrite) | (not os.path.exists(destpth)):
            print(river, datetime.now())
            tiles = gpd.read_file(f"/data/sand_mining/rivers/river_grids/wris/{river}.geojson")
            tiles = tiles.drop(columns = ['index_right', 'index'])
            
            gc.collect()
            test = S2ImageExtractor(tiles, args.start_date, args.end_date, collection = args.collection, bands = S2_BANDS, rivername=river)
            test.chunk_and_merge_tiles()
        else:
            print(river, "Done", datetime.now())