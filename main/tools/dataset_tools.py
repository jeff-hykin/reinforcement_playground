import gdown
import os
temp_folder_path = f"{os.environ.get('PROJECTR_FOLDER')}/settings/.cache/common_format_datasets"

def get_path_to_mnist(quiet=False):
    """
    returns the mnist folder, downloading the mnist data if needed
    """
    url = 'https://drive.google.com/uc?id=1im9_agM-hHl8dKQU5uWh054rScjPINtz' # jeff hykin's google drive copy of mnist
    output = f'{temp_folder_path}/mnist.zip'
    gdown.cached_download(url, output, postprocess=gdown.extractall, quiet=quiet)
    
    return temp_folder_path+"/mnist"