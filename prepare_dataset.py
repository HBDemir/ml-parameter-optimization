"""
prepare_dataset.py
──────────────────
One-time setup script. Run this once on Colab to generate the Q-factor CSV
and persist it to Google Drive. After it finishes, copy the file ID from the
Drive link and set GDRIVE_FILE_ID in your notebooks.

Usage (Colab):
    !python prepare_dataset.py

For GPU acceleration (recommended — use a GPU runtime in Colab):
    Runs automatically if a GPU is detected.
"""

import os
import sys
import zipfile
import shutil
import subprocess
from pathlib import Path

import gdown

# ── Config ────────────────────────────────────────────────────────────────────
DRIVE_URLS = {
    '06082025': 'https://drive.google.com/file/d/1BN4DzT8Pl-FMxYl0oJ0sPYHPEQ4pjf7X/view?usp=sharing',
    '26082025': 'https://drive.google.com/file/d/1kh9HPNOR-mPc1F9yKjheTS2jZrak4Vek/view?usp=sharing',
}
DATA_DIR  = Path('data/dataset_combined')
CSV_PATH  = 'qfactors_1.0um.csv'
DRIVE_CSV = '/content/drive/MyDrive/AI Material Processing Data/qfactors_1.0um.csv'
# ─────────────────────────────────────────────────────────────────────────────


def mount_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        return True
    except Exception:
        return False


def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, url in DRIVE_URLS.items():
        zmap_dir = DATA_DIR / dataset / 'zmap'
        if zmap_dir.exists() and any(zmap_dir.iterdir()):
            print(f'  {dataset}: already extracted, skipping.')
            continue
        zip_path = Path(f'{dataset}.zip')
        if not zip_path.exists():
            print(f'  Downloading {dataset}.zip ...')
            gdown.download(url=url, output=str(zip_path), fuzzy=True, quiet=True)
        print(f'  Extracting {dataset}.zip ...')
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)


def detect_gpu():
    result = subprocess.run(['nvidia-smi'], capture_output=True)
    return result.returncode == 0


def run_extraction(use_gpu):
    env = os.environ.copy()
    env['TQDM_DISABLE'] = '1'
    if use_gpu:
        env['USE_GPU'] = '1'
    subprocess.run([sys.executable, 'src/extract_qfactors.py'], check=True, env=env)


def save_to_drive():
    os.makedirs(os.path.dirname(DRIVE_CSV), exist_ok=True)
    shutil.copy(CSV_PATH, DRIVE_CSV)


if __name__ == '__main__':
    print('=== Dataset Preparation ===\n')

    print('Step 1/4  Mounting Google Drive ...')
    on_drive = mount_drive()
    print(f'  {"Mounted." if on_drive else "Not on Colab — skipping Drive mount."}\n')

    print('Step 2/4  Downloading & extracting raw data ...')
    download_and_extract()
    print()

    print('Step 3/4  Computing Q-factors ...')
    gpu = detect_gpu()
    print(f'  GPU {"detected" if gpu else "not available"} — running on {"GPU" if gpu else "CPU"}.')
    run_extraction(use_gpu=gpu)
    print()

    print('Step 4/4  Saving CSV to Google Drive ...')
    if on_drive:
        save_to_drive()
        print(f'  Saved to: {DRIVE_CSV}')
        print()
        print('  Next step:')
        print('  1. Open Google Drive → AI Material Processing Data')
        print('  2. Right-click qfactors_1.0um.csv → Share → Copy link')
        print('  3. Extract the file ID from the link')
        print('  4. Paste it into GDRIVE_FILE_ID in your notebooks')
    else:
        print(f'  Skipped (not on Colab). CSV is at: {Path(CSV_PATH).resolve()}')

    print('\nDone.')
