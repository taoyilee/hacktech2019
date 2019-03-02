import os
from google.cloud import storage

if __name__ == "__main__":
    configuration_file = "gs://xecg_data/config.ini"
    if os.path.isfile(configuration_file):
        print(f"Using configuration file {configuration_file}")
    else:
        raise FileNotFoundError(f"configuration file {configuration_file} dose not exist")
