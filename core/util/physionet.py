import requests
import os

from lxml import html


def list_files(base_url):
    print(f"Listing files from {base_url}")
    r = requests.get(base_url)
    raw_html = html.fromstring(r.text)
    dataset_files = raw_html.xpath('//pre/a/@href')
    print(f"{len(dataset_files)} items found for xpath //pre/a/@href in this html")
    return dataset_files


def download_dataset(dataset_files, base_url, output_dir="./data"):
    os.makedirs(output_dir, exist_ok=True)
    for d in dataset_files:
        file_name, file_ext = os.path.splitext(d)
        if file_ext in [".hea", ".dat", ".ari", ".atr", ".sta", ".stb", ".stc", ".cnt"]:
            print(f"Downloading {d}")
            url = f"{base_url}/{d}"
            r = requests.get(url)
            sub_dir = os.path.join(output_dir, file_ext.split(".")[1])
            os.makedirs(sub_dir, exist_ok=True)
            open(os.path.join(sub_dir, f"{d}"), "wb").write(r.content)
