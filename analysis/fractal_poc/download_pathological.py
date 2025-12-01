#!/usr/bin/env python3
"""
Download Pathological Blood Cell Datasets
==========================================

Downloads public datasets for comparison:
1. BCCD - Normal blood cells (already downloaded)
2. Blood Cell Cancer (ALL) - Kaggle - Leukemia cells
3. ALL-IDB - Acute Lymphoblastic Leukemia

Note: Some datasets require Kaggle authentication or manual download.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"


def download_all_idb():
    """
    Download ALL-IDB dataset (Acute Lymphoblastic Leukemia).
    
    This is a smaller dataset that's publicly available.
    Contains both healthy and ALL cells.
    """
    all_idb_dir = DATA_DIR / "ALL_IDB"
    all_idb_dir.mkdir(exist_ok=True)
    
    # ALL-IDB1 - contains images with cells marked
    # Unfortunately requires registration at: https://homes.di.unimi.it/scotti/all/
    
    logger.info("""
    ================================================================================
    ALL-IDB Dataset requires manual download:
    
    1. Go to: https://homes.di.unimi.it/scotti/all/
    2. Request access (usually approved quickly)
    3. Download ALL_IDB1.zip and ALL_IDB2.zip
    4. Extract to: {all_idb_dir}
    ================================================================================
    """)
    
    return False


def download_kaggle_all():
    """
    Download Blood Cell Cancer (ALL) dataset from Kaggle.
    
    Dataset: https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class
    
    Contains 4 classes:
    - Benign (normal)
    - Early Pre-B ALL
    - Pre-B ALL  
    - Pro-B ALL
    """
    kaggle_dir = DATA_DIR / "blood_cell_cancer_ALL"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Check if kaggle is installed
    try:
        import subprocess
        result = subprocess.run(['kaggle', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise ImportError("Kaggle CLI not configured")
    except (ImportError, FileNotFoundError):
        logger.info("""
    ================================================================================
    Kaggle CLI required for automatic download.
    
    Option 1 - Install and configure Kaggle:
        pip install kaggle
        # Get API token from: https://www.kaggle.com/settings
        # Place kaggle.json in ~/.kaggle/
        
    Option 2 - Manual download:
        1. Go to: https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class
        2. Download the dataset
        3. Extract to: {kaggle_dir}
    ================================================================================
        """)
        return False
    
    # Download using Kaggle CLI
    try:
        logger.info("Downloading Blood Cell Cancer (ALL) dataset from Kaggle...")
        os.chdir(kaggle_dir)
        os.system("kaggle datasets download -d mohammadamireshraghi/blood-cell-cancer-all-4class")
        
        # Extract
        zip_file = kaggle_dir / "blood-cell-cancer-all-4class.zip"
        if zip_file.exists():
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(kaggle_dir)
            zip_file.unlink()
            logger.info("Dataset downloaded and extracted successfully!")
            return True
    except Exception as e:
        logger.error(f"Failed to download: {e}")
        return False
    
    return False


def create_synthetic_pathological():
    """
    Create synthetic 'pathological' dataset by modifying normal images.
    
    This is a TEMPORARY solution for testing the pipeline.
    In real analysis, we need actual pathological samples.
    
    Modifications to simulate pathological conditions:
    - Increased clustering (cells closer together)
    - Size variation (larger/smaller cells)
    - Shape distortion
    """
    from PIL import Image, ImageFilter
    import numpy as np
    
    synth_dir = DATA_DIR / "synthetic_pathological"
    synth_dir.mkdir(exist_ok=True)
    
    bccd_dir = DATA_DIR / "BCCD_Dataset-master" / "BCCD" / "JPEGImages"
    if not bccd_dir.exists():
        logger.error("BCCD dataset not found. Run: python run_poc.py --download")
        return False
    
    images = sorted(bccd_dir.glob("*.jpg"))[:20]
    logger.info(f"Creating synthetic pathological images from {len(images)} normal images...")
    
    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        
        # Apply transformations to simulate pathological conditions
        # 1. Increase contrast (abnormal staining)
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        
        # 2. Slight blur (different focus due to cell morphology)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # 3. Scale to simulate cell size changes
        w, h = img.size
        scale = np.random.uniform(0.9, 1.1)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Crop/pad back to original size
        if new_w > w:
            left = (new_w - w) // 2
            img = img.crop((left, 0, left + w, h))
        
        # Save
        output_path = synth_dir / f"synth_patho_{i:03d}.jpg"
        img.save(output_path, quality=95)
    
    logger.info(f"Created {len(images)} synthetic pathological images in {synth_dir}")
    return True


def download_malaria_sample():
    """
    Download a sample of the NIH Malaria dataset.

    Full dataset: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
    This is a large dataset (27,558 cell images).

    We'll download a sample for testing.
    """
    malaria_dir = DATA_DIR / "malaria_cells"
    malaria_dir.mkdir(exist_ok=True)

    # The NIH malaria dataset is available via TensorFlow datasets
    # Let's try to download it
    try:
        import tensorflow_datasets as tfds
        logger.info("Downloading NIH Malaria dataset via TensorFlow Datasets...")

        ds = tfds.load('malaria', split='train[:100]', as_supervised=True)

        # Save images
        parasitized_dir = malaria_dir / "Parasitized"
        uninfected_dir = malaria_dir / "Uninfected"
        parasitized_dir.mkdir(exist_ok=True)
        uninfected_dir.mkdir(exist_ok=True)

        for i, (image, label) in enumerate(ds):
            img = Image.fromarray(image.numpy())
            if label.numpy() == 1:
                img.save(parasitized_dir / f"parasitized_{i:04d}.png")
            else:
                img.save(uninfected_dir / f"uninfected_{i:04d}.png")

        logger.info(f"Downloaded malaria sample to {malaria_dir}")
        return True

    except ImportError:
        logger.info("""
    ================================================================================
    TensorFlow Datasets not installed.

    To download NIH Malaria dataset:
        pip install tensorflow-datasets

    Or manual download from:
        https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
    ================================================================================
        """)
        return False


def check_available_datasets():
    """Check which datasets are available."""
    datasets = {
        "BCCD (Normal)": DATA_DIR / "BCCD_Dataset-master" / "BCCD" / "JPEGImages",
        "Blood Cell Cancer (ALL)": DATA_DIR / "blood_cell_cancer_ALL",
        "ALL-IDB": DATA_DIR / "ALL_IDB",
        "Malaria (NIH)": DATA_DIR / "malaria_cells",
        "Synthetic Pathological": DATA_DIR / "synthetic_pathological"
    }
    
    print("\n" + "="*60)
    print("AVAILABLE DATASETS")
    print("="*60)
    
    for name, path in datasets.items():
        if path.exists():
            n_images = len(list(path.rglob("*.jpg"))) + len(list(path.rglob("*.png")))
            print(f"✅ {name}: {n_images} images")
        else:
            print(f"❌ {name}: Not found")
    
    print("="*60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kaggle', action='store_true', help='Download Kaggle ALL dataset')
    parser.add_argument('--synthetic', action='store_true', help='Create synthetic pathological')
    parser.add_argument('--check', action='store_true', help='Check available datasets')
    parser.add_argument('--all', action='store_true', help='Try all methods')
    
    args = parser.parse_args()
    
    DATA_DIR.mkdir(exist_ok=True)
    
    if args.check or not any([args.kaggle, args.synthetic, args.all]):
        check_available_datasets()
    
    if args.kaggle or args.all:
        download_kaggle_all()
    
    if args.synthetic or args.all:
        create_synthetic_pathological()
    
    if args.all:
        download_all_idb()
        check_available_datasets()

