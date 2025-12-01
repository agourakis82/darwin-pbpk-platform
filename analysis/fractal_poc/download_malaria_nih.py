#!/usr/bin/env python3
"""
Download NIH Malaria Dataset
============================

Downloads the NIH Malaria Cell Images dataset.
This is a real pathological dataset with:
- Parasitized (infected) cells
- Uninfected (normal) cells

Source: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
Mirror: https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import logging
import ssl

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"

# NIH Malaria dataset URL
MALARIA_URL = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
# Alternative mirror
MALARIA_URL_ALT = "https://ceb.nlm.nih.gov/repositories/malaria-datasets/"


def download_with_progress(url: str, dest: Path) -> bool:
    """Download file with progress indicator."""
    try:
        # Create SSL context that doesn't verify (some NIH servers have cert issues)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        logger.info(f"Downloading from {url}")
        logger.info("This may take several minutes (dataset is ~350MB)...")
        
        with urllib.request.urlopen(url, context=ctx) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192
            
            with open(dest, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\rProgress: {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end='')
        
        print()  # New line after progress
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def main():
    DATA_DIR.mkdir(exist_ok=True)
    malaria_dir = DATA_DIR / "malaria_cells"
    
    if malaria_dir.exists() and len(list(malaria_dir.rglob("*.png"))) > 100:
        logger.info("Malaria dataset already downloaded")
        n_para = len(list((malaria_dir / "Parasitized").glob("*.png")))
        n_uninf = len(list((malaria_dir / "Uninfected").glob("*.png")))
        logger.info(f"  Parasitized: {n_para} images")
        logger.info(f"  Uninfected: {n_uninf} images")
        return True
    
    zip_path = DATA_DIR / "cell_images.zip"
    
    # Try download
    if not zip_path.exists():
        success = download_with_progress(MALARIA_URL, zip_path)
        if not success:
            logger.info("\nTrying alternative download method...")
            # Try with requests if available
            try:
                import requests
                response = requests.get(MALARIA_URL, stream=True, verify=False)
                total = int(response.headers.get('content-length', 0))
                with open(zip_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            print(f"\rProgress: {(downloaded/total)*100:.1f}%", end='')
                print()
                success = True
            except:
                pass
        
        if not success:
            logger.error("""
================================================================================
Automatic download failed. Please download manually:

1. Go to: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
2. Download 'cell_images.zip' (~350MB)
3. Extract to: {malaria_dir}

The dataset contains:
- Parasitized/ folder: Malaria-infected cells
- Uninfected/ folder: Normal cells
================================================================================
            """)
            return False
    
    # Extract
    if zip_path.exists():
        logger.info("Extracting dataset...")
        malaria_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)
        
        # The zip extracts to cell_images/, we want malaria_cells/
        extracted = DATA_DIR / "cell_images"
        if extracted.exists():
            extracted.rename(malaria_dir)
        
        zip_path.unlink()  # Remove zip
        
        n_para = len(list((malaria_dir / "Parasitized").glob("*.png")))
        n_uninf = len(list((malaria_dir / "Uninfected").glob("*.png")))
        logger.info(f"Extracted successfully!")
        logger.info(f"  Parasitized: {n_para} images")
        logger.info(f"  Uninfected: {n_uninf} images")
        return True
    
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

