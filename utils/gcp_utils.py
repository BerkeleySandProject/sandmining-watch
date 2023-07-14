from google.cloud import storage

DEFAULT_BUCKET_NAME = "sand_mining"

def get_public_url(path, bucket_name=DEFAULT_BUCKET_NAME):
    return f"https://storage.googleapis.com/{bucket_name}/{path}"

def list_subfolders(client: storage.Client, folder_name, bucket_name=DEFAULT_BUCKET_NAME):
    bucket = client.bucket(bucket_name)

    prefix = folder_name + '/'
    blobs = bucket.list_blobs(prefix=prefix, delimiter='/')
    # blobs is a google.api_core.page_iterator.HTTPIterator. list() makes it consume its values.
    list(blobs)

    subfolders = []
    for prefix in blobs.prefixes:
        # Remove the folder path from the prefix
        subfolder = prefix.replace(folder_name + '/', '')
        subfolders.append(subfolder[:-1])  # [:-1] removes the '/' at the end of the string

    return subfolders
