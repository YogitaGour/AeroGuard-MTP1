# download_homeobjects.py
import zipfile
import urllib.request
from pathlib import Path

url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip"
zip_path = Path("homeobjects-3K.zip")
extract_dir = Path(".")

# Download the zip file
print(f"Downloading from {url}...")
urllib.request.urlretrieve(url, zip_path)

# Extract the zip file
print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Clean up the zip file
zip_path.unlink()
print("✅ Dataset downloaded and extracted successfully!")