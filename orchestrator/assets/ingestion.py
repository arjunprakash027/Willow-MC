import os
import requests
from dagster import asset, MaterializeResult
import shutil
import time
from typing import Tuple

def download_data(url: str, file_name: str) -> Tuple[str, int, str]:
    raw_dir = "data/raw"
    target_path = os.path.join(raw_dir, file_name)

    os.makedirs(raw_dir, exist_ok=True)

    with requests.get(url, timeout=10, stream=True) as response:
        response.raise_for_status()

        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return target_path, os.path.getsize(target_path), url
    
@asset(group_name="bronze", compute_kind="python")
def raw_data_t20(context):
    """
    Downloads the latest Men's T20 ball-by-ball JSON data from Cricsheet.
    """

    url = "https://cricsheet.org/downloads/t20s_male_json.zip"
    target_path, file_size, url = download_data(url, "t20s_male_json.zip")
    
    return MaterializeResult(
        metadata={
            "file_path": target_path,
            "file_size": file_size,
            "source_url": url,
            "download_timestamp": time.time(),
        }
    )

@asset(group_name="bronze", compute_kind="python")
def raw_data_odi(context):
    """
    Downloads the latest Men's ODI ball-by-ball JSON data from Cricsheet.
    """

    url = "https://cricsheet.org/downloads/odis_json.zip"
    target_path, file_size, url = download_data(url, "odis_json.zip")
    
    return MaterializeResult(
        metadata={
            "file_path": target_path,
            "file_size": file_size,
            "source_url": url,
            "download_timestamp": time.time(),
        }
    )
    