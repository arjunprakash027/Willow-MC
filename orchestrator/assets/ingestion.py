import os
import requests
from dagster import asset, MaterializeResult
import shutil
import time

@asset(group_name="bronze", compute_kind="python")
def raw_data(context):
    """
    Downloads the latest Men's T20 ball-by-ball JSON data from Cricsheet.
    """

    url = "https://cricsheet.org/downloads/t20s_male_json.zip"
    raw_dir = "data/raw"
    target_path = os.path.join(raw_dir, "t20s_male_json.zip")

    os.makedirs(raw_dir, exist_ok=True)

    with requests.get(url, timeout=10, stream=True) as response:
        response.raise_for_status()

        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return MaterializeResult(
        metadata={
            "file_path": target_path,
            "file_size": os.path.getsize(target_path),
            "source_url": url,
            "download_timestamp": time.time(),
        }
    )
    