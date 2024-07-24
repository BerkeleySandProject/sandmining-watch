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
import importlib

from utils.rastervision_pipeline import create_s2_image_source, create_scene_s2, scene_to_inference_ds
from ml.learner import BinarySegmentationPredictor
from models.model_factory import model_factory
from project_config import CLASS_CONFIG
from ml.eval_utils import save_predictions

import subprocess
import json
import matplotlib.pyplot as plt

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

parser.add_argument("start_date", help="start date of period in yyyy-mm-dd format")
parser.add_argument("end_date", help="end date of period in yyyy-mm-dd format")
parser.add_argument("-c", "--collection", help="Name of Sentinel Collection. Currently only supports S2-HARMONIZED", default = "S2_HARMONIZED")
parser.add_argument('-r', '--rivernames', nargs='+', default=[], help="provide river names to process")
parser.add_argument('-o', '--overwrite', default = False, action='store_true')
parser.add_argument('-k', '--config', default = "satmae_large_methoda_lora_inf_config")
# parser.add_argument('-a', '--all_rivers', help="load all river names",
#                     action="store_true")
args = parser.parse_args()



#### Configuration
# from experiment_configs.configs import *
config_module = importlib.import_module("experiment_configs.configs")

config = getattr(config_module, args.config)

PREDICTION_ID = config.wandb_id.split("/")[-1]
if args.collection == "S2_HARMONIZED":
    N_CHANNELS = 10
elif args.collection == "S2_L1C":
    N_CHANNELS = 11
else:
    print("Define collection name correctly!!!")

# Load model
from ml.model_stats import count_number_of_weights
model = model_factory(
    config,
    n_channels= N_CHANNELS ,
)

predictor = BinarySegmentationPredictor(
    config, 
    model,         
    path_to_weights = config.encoder_weights_path,
    path_to_lora =  config.lora_weights_path
)

# crop_sz = int(config.tile_size // 5) #20% of the tiles at the edges are discarded

all_params, trainable_params = count_number_of_weights(predictor.model)
print(f"trainable params: {trainable_params/1e6}M || all params: {all_params/1e6}M || trainable%: {100 * trainable_params / all_params:.2f}")


def make_prediction(river_name, timestep):
    BASE_PATH = f"/data/sand_mining/inference/outputs/{timestep}/{river_name}"
    RIVER_URI = f"/data/sand_mining/rivers/river_polygons/wris/{river_name}.geojson"
    uri_to_s2 = f"/data/sand_mining/inference/inputs/S2_HARMONIZED/{timestep}/{river_name}_{timestep}.tif"

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    assert os.path.exists(RIVER_URI), "River shapefile not found!"
    assert os.path.exists(uri_to_s2), "Input image not found!"

    print("Generating Predictions", datetime.now())
    ## Set up rastervision predictions
    r_source = create_scene_s2(config, uri_to_s2, label_uri = None, scene_id = 0, rivers_uri = RIVER_URI)
    r_inference = scene_to_inference_ds(config, r_source, full_image=False, stride=int(config.tile_size/2))
    crs_transformer = r_inference.scene.raster_source.crs_transformer
    prediction = predictor.predict_site(r_inference, crop_sz=config.crop_sz)


    print("Saving..", datetime.now())
    #Save predictions
    save_predictions(prediction, 
         path = BASE_PATH, 
         class_config = CLASS_CONFIG, 
         crs_transformer = crs_transformer, 
         threshold=0.5)


    ## Convert to COGS
    for f in [
                "scores",
                "mask"
                ]:
        
        
        ##Build overviews (not necessary to build a valid COG, but recommended for files over 512 H X 512W
        os.rename(f"{BASE_PATH}/{f}.tif", 
                  f"{BASE_PATH}/tmp.tif")
        
        cog_cmd = f"""gdaladdo -r nearest --config BIGTIFF_OVERVIEW YES --config GDAL_CACHEMAX 4096 \
        --config COMPRESS_OVERVIEW DEFLATE --config BIGTIFF YES --config GDAL_NUM_THREADS ALL_CPUS \
        {BASE_PATH}/tmp.tif 2 4 8 16 32"""
        try:
            print("Building overviews")
            subprocess.run(cog_cmd,
                       cwd = f"{ENVBIN_GDAL}/bin",
                       shell=True)
        except subprocess.CalledProcessError as e:
            print(e)
            
        
        #### create final cog 
        cog_cmd = f"""gdal_translate \
        {BASE_PATH}/tmp.tif \
        {BASE_PATH}/{f}.tif \
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

        ##remove tmp files
        os.remove(f"{BASE_PATH}/tmp.tif")
        gc.collect()



if len(args.rivernames) > 0:
    for river in args.rivernames:
        scorefile = f"/data/sand_mining/inference/outputs/{args.start_date}_{args.end_date}/{river}/scores.tif"
        if not os.path.exists(scorefile):
            print(river, datetime.now())
            make_prediction(river, f"{args.start_date}_{args.end_date}")
        elif args.overwrite:
            for f in ['scores.tif', 'pixel_hits.npy', 'mask.tif']:
                os.remove(f"/data/sand_mining/inference/outputs/{args.start_date}_{args.end_date}/{river}/{f}")
            
            print(river, datetime.now())
            make_prediction(river, f"{args.start_date}_{args.end_date}")
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
     'Burhi Gandak':'Burhi-Gandak', 
     'Kali Sindh':'Kali-Sindh'
                                                             
    })
    river_names = np.sort(df_rivers['rivname_clean'].unique())

    for river in river_names:
        scorefile = f"/data/sand_mining/inference/outputs/{args.start_date}_{args.end_date}/{river}/scores.tif"
        if (args.overwrite) | (not os.path.exists(scorefile)):
            print(river, datetime.now())
            make_prediction(river, f"{args.start_date}_{args.end_date}")
        else:
            print(river, "Done", datetime.now())