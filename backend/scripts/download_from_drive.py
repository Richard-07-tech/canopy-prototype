"""Downloader for Earth Engine export outputs from Google Drive into data/raw/sentinel2/.

Reads metadata/s2_drive_export_tasks.json, queries task status via Earth Engine,
and downloads completed GeoTIFF files from Google Drive into data/raw/sentinel2/YYYY/YYYY-MM-DD/
to match the pipeline's expected local folder layout.
"""
from __future__ import annotations

import io
import json
import logging
import sys
import datetime as dt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_config, init_earth_engine, PROJECT_ROOT
import ee.oauth
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drive_downloader")


def get_drive_service():
    creds_path = Path.home() / ".config/earthengine/credentials"
    if not creds_path.exists():
        raise RuntimeError(f"EE credentials not found at {creds_path}")
    data = json.loads(creds_path.read_text())
    creds = Credentials(
        token=data.get("access_token"),
        refresh_token=data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=ee.oauth.CLIENT_ID,
        client_secret=ee.oauth.CLIENT_SECRET,
        scopes=data.get("scopes")
    )
    req = Request()
    creds.refresh(req)
    return build('drive', 'v3', credentials=creds)


def check_tasks_and_download(folder_name: str = "forest_agb_pipeline_exports") -> dict[str, str]:
    tasks_file = PROJECT_ROOT / "metadata/s2_drive_export_tasks.json"
    if not tasks_file.exists():
        logger.error("No export tasks recorded at %s", tasks_file)
        return {}

    tasks = json.loads(tasks_file.read_text())
    import ee
    init_earth_engine(load_config(PROJECT_ROOT / "config/config.yaml"))

    statuses = {}
    completed_tasks = []
    print("\nChecking status of submitted Earth Engine export tasks:")
    print("-" * 65)

    for item in tasks:
        tid = item["task_id"]
        status_list = ee.data.getTaskStatus(tid)
        st = status_list[0] if status_list else {}
        state = st.get("state", "UNKNOWN")
        statuses[tid] = state
        print(f"Task ID: {tid} | Day: {item['day']} | Band: {item['band']} | State: {state}")
        if state == "COMPLETED":
            completed_tasks.append(item)

    print("-" * 65)
    print(f"Total Tasks: {len(tasks)} | Completed: {len(completed_tasks)} | In Progress: {len(tasks) - len(completed_tasks)}")

    if not completed_tasks:
        logger.info("No tasks have reached COMPLETED state yet. Re-run this script once exports finish.")
        return statuses

    logger.info("Connecting to Google Drive to download %d completed files...", len(completed_tasks))
    service = get_drive_service()

    # Find folder ID
    q_folder = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    res_f = service.files().list(q=q_folder, fields="files(id, name)").execute()
    folders = res_f.get("files", [])
    if not folders:
        logger.error("Drive folder '%s' not found.", folder_name)
        return statuses
    folder_id = folders[0]["id"]

    # List files in folder
    res_files = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name, size)"
    ).execute()
    drive_files = {f["name"]: f["id"] for f in res_files.get("files", [])}

    raw_s2_root = PROJECT_ROOT / "data/raw/sentinel2"

    for item in completed_tasks:
        filename = item["filename"]
        if filename not in drive_files:
            logger.warning("File %s not yet visible in Drive folder %s", filename, folder_name)
            continue

        file_id = drive_files[filename]
        day = item["day"]
        band = item["band"]
        year = day[:4]
        target_dir = raw_s2_root / year / day
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{band}.tif"

        if target_file.exists() and target_file.stat().st_size > 0:
            logger.info("Reusing existing local file: %s", target_file)
            continue

        logger.info("Downloading %s from Drive -> %s...", filename, target_file)
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(str(target_file), 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        logger.info("Downloaded %s successfully (%d bytes).", target_file.name, target_file.stat().st_size)

    return statuses


if __name__ == "__main__":
    check_tasks_and_download()
