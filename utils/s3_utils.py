import boto3

def list_buckets():
    s3 = boto3.client('s3')
    response = s3.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    return buckets


# write method to access a public S3 bucket and print all the folders inside it, pass in the name of the bucket
def list_folders(s3, bucket_name):

    response = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')
    folders = [prefix['Prefix'] for prefix in response['CommonPrefixes']]
    return folders


# define get_presigned_url function -> when a bucket is not public
def get_presigned_url(s3, bucket_name, key, expiration=3600):
    url = s3.generate_presigned_url(
        ClientMethod='get_object',
        Params={
            'Bucket': bucket_name,
            'Key': key
        },
        ExpiresIn=expiration
    )
    return url

def get_public_url(s3_location, s3_bucket, key):

    url = "https://s3-%s.amazonaws.com/%s/%s" % (s3_location, s3_bucket, key)
    return url

# function, that given a folder, returns a list of all the files in that folder
def list_files(s3, bucket_name, folder):
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder)
    files = [file['Key'] for file in response['Contents']]
    return files[1:]
