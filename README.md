# sandmining-watch
Deep learning methodology to detect sand mines. 

As the major ingredient of concrete and asphalt, sand is vital to economic growth2 and will play a key role in aiding the transition to a low carbon society. However, excessive and unregulated sand mining in the Global South has high socio-economic and environmental costs, and amplifies the effects of climate change. Sand mines are characterized by informality and high temporal variability, and data on the location and extent of these mines tends to be sparse. We provide a custom sand-mine detection tool by fine-tuning foundation models for earth observation, which leverage self supervised learning - a cost-effective and powerful approach in sparse data regimes. These tools allow for real-time monitoring of sand mining activity and can enable more effective policy and regulation.




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
