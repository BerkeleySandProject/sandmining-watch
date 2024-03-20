# Sand Mining Watch 
[Website](http://www.globalpolicy.science/sand-mining-watch) | [NeurIPS-CCAI presentation](https://neurips.cc/virtual/2023/76963)


Deep learning methodology to detect sand mines [work in progress].

As the major ingredient of concrete and asphalt, sand is vital to economic growth and will play a key role in aiding the transition to a low carbon society. However, excessive and unregulated sand mining in the Global South has high socio-economic and environmental costs, and amplifies the effects of climate change. Sand mines are characterized by informality and high temporal variability, and data on the location and extent of these mines tends to be sparse. We provide a custom sand-mine detection tool by fine-tuning foundation models for earth observation, which leverage self supervised learning - a cost-effective and powerful approach in sparse data regimes. These tools allow for real-time monitoring of sand mining activity and can enable more effective policy and regulation.


![panel1](https://github.com/BerkeleySandProject/sandmining-watch/assets/2422530/50def1fa-52b9-4d8d-8c9d-d655d03ecaef)

## Datasets
We have acquired data (latitude, longitude, timestamp) on sand mining activities across 21 different river basins across India, through a partnership with Veditum India Foundation. Currently, these data cover 39 distinct mining sites; we expect to expand this to over 100 sites over the course
of our study. We extract image patches (ranging in size from 2.5 sq.km to 582 sq.km) from freely available Sentinel-2 multi-spectral and Sentinel-1 synthetic aperture radar imagery around visually recognizable sand mining footprints at each site3. A majority of Indian rivers are characterized by high average flood discharges and large temporal variability, leading to huge intra-annual variation in sand deposition rates and mining footprints. We consider these changes to be strong natural label augmentations (figure above, inset 1). This allows us to obtain multiple labels (of arbitrary size) for each location that represent the seasonal lifecycle of sand mines. While sub-meter resolution imagery (figure above, inset 2) captures more precise information on mining activity, we believe that 10m imagery will prove to be an effective feature set since it captures broad patterns of importance (i.e. scarring, pitting and flooding) at high temporal & spectral resolution.


## System Design
![System-Diagram](https://github.com/BerkeleySandProject/sandmining-watch/assets/2422530/a72e09c8-1c81-49e3-858a-dfb8c9375a85)

The system diagram is shown above. It consists of a data generation stage (upper half) and a data modeling stage (lower half).


#### Install conda enviroment
```
conda env create -f environment.yml
```

#### Structure of the repository

`label/` contains the labeling pipeline:
- `observation_selector.ipynb` exports Sentinel-1/2 data from Google Earth Engine to Google Cloud Platform (GCP) Storage
- `create_labelbox_dataset.ipynb` populates Labelbox dataset with pointers (URLs) to GCP
- `export_annotations.ipynb` exports annotations from Labelbox as GeoJSONs to GCP
- `aoi_generator.ipynb` populates GCP with coordinates of river boundaries with buffer


`train_eval/train_eval.ipynb` trains and evaluates models.

`inference/inference.ipynb` runs predictions on dataset without annotations.

`project_config.py` holds configuration that is valid for the entire project.

Objects of the class `SupervisedTrainingConfig` (defined in `experiment_configs/schemas.py`) hold configuration for a single training run.
