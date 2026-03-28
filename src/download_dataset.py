import os
import re

import pandas as pd
import requests
from tqdm import tqdm


def download_file(url, path):
    try:
        response = requests.get(url, stream=True, timeout=15)
        if response.status_code == 200:
            with open(path, "wb") as file_obj:
                for chunk in response.iter_content(1024):
                    file_obj.write(chunk)
            return True

        print(f"Failed ({response.status_code}): {url}")
        return False
    except Exception as exc:
        print(f"Error downloading {url}: {exc}")
        return False


def extract_folder_id(url):
    """
    Extract the hidden folder ID from the broken URL.
    Example:
    https://.../hq_data/hi/967179/825727_audio.wav -> 967179
    """

    match = re.search(r"/hi/(\d+)/", url)
    return match.group(1) if match else None


def download_dataset(excel_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    _, extension = os.path.splitext(excel_path)
    extension = extension.lower()

    if extension == ".csv":
        df = pd.read_csv(excel_path)
    else:
        df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()

    print("Columns:", df.columns.tolist())

    records = []
    success, fail = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        recording_id = int(row["recording_id"])

        raw_audio_url = row["rec_url_gcp"]
        folder_id = extract_folder_id(raw_audio_url)

        if folder_id is None:
            print(f"Could not extract folder ID for {recording_id}")
            fail += 1
            continue

        audio_url = f"https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_audio.wav"
        transcript_url = f"https://storage.googleapis.com/upload_goai/{folder_id}/{recording_id}_transcription.json"

        audio_path = os.path.join(save_dir, f"{recording_id}.wav")
        json_path = os.path.join(save_dir, f"{recording_id}.json")

        audio_ok = True
        json_ok = True

        if not os.path.exists(audio_path):
            audio_ok = download_file(audio_url, audio_path)

        if not os.path.exists(json_path):
            json_ok = download_file(transcript_url, json_path)

        if audio_ok and json_ok:
            success += 1
            records.append(
                {
                    "audio_path": audio_path,
                    "json_path": json_path,
                }
            )
        else:
            fail += 1

    print(f"\nSuccess: {success}")
    print(f"Failed: {fail}")
    return records
