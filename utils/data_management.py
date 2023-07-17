import datetime

def get_date_from_key(key):
    # Example: From "Sone_Rohtas_84-21_24-91_2022-03-01_rgb.tif" this function returns a the datetime 2022-03-01 00:00:00
    year, month, day = key.split("_")[-2].split("-")
    # We'd prefer to set a datetime.date instead of a datetime.datetime,
    # but Labelbox doesn't accept a date. Therefore we set a timestamp.
    return datetime.datetime(int(year), int(month), int(day))

def get_annotation_path(key):
    splitted = key.split("_")
    observation = "_".join(splitted[:4])
    image = "_".join(splitted[:5])
    path = f"labels/{observation}/annotations/{image}_annotations.geojson"
    return path
