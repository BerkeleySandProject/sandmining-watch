# sandmining-watch
Deep learning methodology to detect sand mines

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
