import json
from google.cloud import storage

from project_config import BUCKET_NAME as DEFAULT_BUCKET_NAME

def get_public_url(path, bucket_name=DEFAULT_BUCKET_NAME):
    return f"https://storage.googleapis.com/{bucket_name}/{path}"

def list_subfolders(client: storage.Client, folder_name=None, bucket_name=DEFAULT_BUCKET_NAME):
    bucket = client.bucket(bucket_name)

    if folder_name is None:
        prefix = ''
    else:
        prefix = f"{folder_name}/"
    
    blobs = bucket.list_blobs(prefix=prefix, delimiter='/')
    # blobs is a google.api_core.page_iterator.HTTPIterator. list() makes it consume its values.
    list(blobs)

    subfolders = []
    for prefix in blobs.prefixes:
        # Remove the folder path from the prefix
        if folder_name is None:
            subfolder = prefix
        else:
            subfolder = prefix.replace(folder_name + '/', '')
        subfolders.append(subfolder[:-1])  # [:-1] removes the '/' at the end of the string

    return subfolders

def list_files_in_folder(client: storage.Client, folder_name, bucket_name=DEFAULT_BUCKET_NAME):
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_name)
    return list(blobs)

def list_files_in_bucket_with_prefix(client: storage.Client, prefix, bucket_name=DEFAULT_BUCKET_NAME):
    bucket = client.get_bucket(bucket_name)
    return bucket.list_blobs(prefix=prefix)

def list_files_in_bucket_with_suffix(client: storage.Client, suffix, bucket_name=DEFAULT_BUCKET_NAME):
    bucket = client.get_bucket(bucket_name)
    files_with_suffix = []
    for blob in bucket.list_blobs():
        if blob.name.endswith(suffix):
            files_with_suffix.append(blob.name)

    return files_with_suffix

def upload_json(client: storage.Client, obj, destination_path, bucket_name=DEFAULT_BUCKET_NAME):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_path)
    blob.upload_from_string(json.dumps(obj), content_type='application/json')    

def upload_raw(client: storage.Client, obj, destination_path, bucket_name=DEFAULT_BUCKET_NAME):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_path)
    # blob.upload_from_string(obj, content_type='application/json')  
    blob.upload_from_file(obj, content_type='application/json')

def upload_file(client: storage.Client, source_path, destination_path, bucket_name=DEFAULT_BUCKET_NAME):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_path)
    with open(source_path, 'rb') as f:
        blob.upload_from_file(f)
