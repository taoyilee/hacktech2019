import configparser as cp
import os


def list_all_hea():
    config = cp.ConfigParser()
    config.read("config.ini.template")
    hea_dir = os.path.join(config["DEFAULT"].get("dataset_dir"), config["DEFAULT"].get("hea_dir"))
    for _, _, files in os.walk(hea_dir):
        return [os.path.join(hea_dir, hea_file) for hea_file in files if hea_file[-4:] == ".hea"]
