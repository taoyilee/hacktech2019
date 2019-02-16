from core.util import list_files, download_dataset
import configparser as cp

if __name__ == "__main__":
    config = cp.ConfigParser()
    config.read("config.ini.template")
    base_url = config["DEFAULT"].get("base_url")
    dataset_files = list_files(base_url)
    download_dataset(dataset_files=dataset_files, base_url=base_url, output_dir=config["DEFAULT"].get("dataset_dir"))
