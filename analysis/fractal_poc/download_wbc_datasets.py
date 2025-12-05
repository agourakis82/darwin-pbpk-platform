#!/usr/bin/env python3
"""
Download White Blood Cell (WBC/Leukocyte) Image Datasets
=========================================================

Organizes leukocyte image datasets by:
- Subpopulation: Neutrophils, Lymphocytes (T/B/NK), Monocytes, Eosinophils, Basophils
- Condition: Normal, Leukemia, Sepsis, Leukopenia

This mirrors the approach used for RBC analysis but with WBC-specific datasets.

Created: 2025-12-01
Author: Darwin PBPK Platform
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path
import logging
from typing import Dict, List, Optional
import urllib.request
import ssl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
WBC_DATA_DIR = DATA_DIR / "leukocytes"

# ============================================================================
# DATASET DEFINITIONS
# ============================================================================

WBC_DATASETS = {
    "bccd_normal": {
        "name": "BCCD - Normal Blood Cells",
        "url": "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip",
        "description": "Normal blood cells including WBCs (already downloaded for RBC analysis)",
        "target": "normal",
        "subpopulations": ["all"]
    },
    
    "leukemia_ALL_kaggle": {
        "name": "Acute Lymphoblastic Leukemia (ALL)",
        "kaggle_id": "mehradaria/leukemia",
        "description": "3,256 images from 89 patients with ALL subtypes",
        "target": "leukemia",
        "subpopulations": ["lymphocytes"],
        "classes": ["Benign", "Malignant_Early_Pre-B", "Malignant_Pre-B", "Malignant_Pro-B"],
        "url_manual": "https://www.kaggle.com/datasets/mehradaria/leukemia"
    },
    
    "wbc_classification": {
        "name": "Blood Cell Classification",
        "kaggle_id": "paultimothymooney/blood-cells",
        "description": "Classified WBCs: Eosinophil, Lymphocyte, Monocyte, Neutrophil",
        "target": "normal",
        "subpopulations": ["eosinophils", "lymphocytes", "monocytes", "neutrophils"],
        "url_manual": "https://www.kaggle.com/datasets/paultimothymooney/blood-cells"
    },
    
    "blood_cell_cancer_ALL": {
        "name": "Blood Cell Cancer ALL (4-class)",
        "kaggle_id": "mohammadamireshraghi/blood-cell-cancer-all-4class",
        "description": "4 classes: Benign, Early Pre-B ALL, Pre-B ALL, Pro-B ALL",
        "target": "leukemia",
        "subpopulations": ["lymphocytes"],
        "url_manual": "https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class"
    }
}

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

def check_kaggle_cli() -> bool:
    """Check if Kaggle CLI is installed and configured."""
    try:
        result = subprocess.run(
            ['kaggle', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def download_via_kaggle(kaggle_id: str, dest_dir: Path) -> bool:
    """Download dataset using Kaggle CLI."""
    if not check_kaggle_cli():
        logger.warning("Kaggle CLI not found. Install with: pip install kaggle")
        logger.info("Get API token from: https://www.kaggle.com/settings")
        logger.info(f"Place kaggle.json in ~/.kaggle/")
        return False
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {kaggle_id} from Kaggle...")
    
    try:
        # Download
        result = subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', kaggle_id, '-p', str(dest_dir)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutes
        )
        
        if result.returncode != 0:
            logger.error(f"Kaggle download failed: {result.stderr}")
            return False
        
        # Extract zip files
        zip_files = list(dest_dir.glob("*.zip"))
        for zf in zip_files:
            logger.info(f"Extracting {zf.name}...")
            try:
                with zipfile.ZipFile(zf, 'r') as z:
                    z.extractall(dest_dir)
                zf.unlink()
                logger.info(f"✅ Extracted {zf.name}")
            except Exception as e:
                logger.error(f"Failed to extract {zf.name}: {e}")
        
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Download timeout. Dataset may be large.")
        return False
    except Exception as e:
        logger.error(f"Error downloading from Kaggle: {e}")
        return False


def download_via_url(url: str, dest_file: Path) -> bool:
    """Download file via HTTP/HTTPS."""
    logger.info(f"Downloading from {url}...")
    
    try:
        # Create SSL context (some servers have cert issues)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(url, context=ctx, timeout=300) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            block_size = 8192
            
            with open(dest_file, 'wb') as f:
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
        
        print()  # New line
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def organize_bccd_for_wbc(bccd_dir: Path, wbc_out_dir: Path) -> bool:
    """
    Organize BCCD dataset for WBC analysis.
    
    BCCD contains normal blood cells. We'll extract WBC images.
    """
    logger.info("Organizing BCCD dataset for WBC analysis...")
    
    jpeg_dir = bccd_dir / "BCCD" / "JPEGImages"
    annotations_dir = bccd_dir / "BCCD" / "Annotations"
    
    if not jpeg_dir.exists():
        logger.error(f"BCCD JPEGImages not found at {jpeg_dir}")
        return False
    
    # Create output structure
    normal_dir = wbc_out_dir / "normal" / "all"
    normal_dir.mkdir(parents=True, exist_ok=True)
    
    # If annotations exist, we can filter by cell type
    # For now, copy all images (we'll filter later during analysis)
    images = list(jpeg_dir.glob("*.jpg"))
    logger.info(f"Found {len(images)} images in BCCD")
    
    # Copy images
    copied = 0
    for img in images[:100]:  # Limit for now
        dest = normal_dir / img.name
        if not dest.exists():
            shutil.copy2(img, dest)
            copied += 1
    
    logger.info(f"✅ Copied {copied} images to {normal_dir}")
    return True


def organize_leukemia_dataset(leukemia_dir: Path, wbc_out_dir: Path) -> bool:
    """
    Organize leukemia dataset by subpopulation and condition.
    
    Handles multiple dataset structures:
    - Original/Benign, Original/Malignant
    - Benign/, Malignant/ directly
    - Recursive search for images
    """
    logger.info(f"Organizing leukemia dataset from {leukemia_dir}...")
    
    # Check if directory exists and has content
    if not leukemia_dir.exists():
        logger.warning(f"Directory does not exist: {leukemia_dir}")
        return False
    
    # Look for image files directly
    all_images = list(leukemia_dir.rglob("*.jpg")) + \
                 list(leukemia_dir.rglob("*.png")) + \
                 list(leukemia_dir.rglob("*.jpeg")) + \
                 list(leukemia_dir.rglob("*.bmp"))
    
    if len(all_images) == 0:
        logger.warning(f"No images found in {leukemia_dir}")
        logger.info("  If you manually downloaded a dataset, extract it here first.")
        return False
    
    logger.info(f"Found {len(all_images)} total images")
    
    # Look for common structures
    possible_structures = [
        ("Original", ["Benign", "Malignant"]),
        ("original", ["benign", "malignant"]),
        ("Benign", None),
        ("Malignant", None),
        ("Benign_", None),
        ("Malignant_", None),
    ]
    
    benign_dir = None
    malignant_dirs = []  # Can have multiple ALL subtypes
    
    # Check for ALL subtypes (Early, Pre, Pro)
    all_subtypes = ["Early", "Pre", "Pro", "early", "pre", "pro", 
                    "Early Pre-B", "Pre-B", "Pro-B"]
    
    for base, subdirs in possible_structures:
        base_path = leukemia_dir / base
        if base_path.exists():
            if subdirs:
                benign_path = base_path / subdirs[0]
                malignant_path = base_path / subdirs[1]
                if benign_path.exists():
                    benign_dir = benign_path
                if malignant_path.exists():
                    malignant_dirs.append(malignant_path)
            else:
                # Direct structure
                if "benign" in base.lower():
                    benign_dir = base_path
                elif "malignant" in base.lower():
                    malignant_dirs.append(base_path)
            
            # Check for ALL subtypes in this base directory
            if base_path.exists():
                for subtype in all_subtypes:
                    subtype_path = base_path / subtype
                    if subtype_path.exists() and subtype_path.is_dir():
                        subtype_images = list(subtype_path.glob("*.jpg")) + \
                                        list(subtype_path.glob("*.png"))
                        if len(subtype_images) > 0:
                            malignant_dirs.append(subtype_path)
            break
    
    # Search recursively if not found
    if not benign_dir or len(malignant_dirs) == 0:
        all_dirs = [d for d in leukemia_dir.rglob("*") if d.is_dir()]
        for d in all_dirs:
            name_lower = d.name.lower()
            # Check if directory contains images
            dir_images = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if len(dir_images) == 0:
                continue
                
            if "benign" in name_lower and benign_dir is None:
                benign_dir = d
            elif "normal" in name_lower and benign_dir is None:
                benign_dir = d
            elif any(subtype.lower() in name_lower for subtype in all_subtypes):
                if d not in malignant_dirs:
                    malignant_dirs.append(d)
            elif "malignant" in name_lower:
                if d not in malignant_dirs:
                    malignant_dirs.append(d)
    
    # If still not found, try to infer from filenames
    if not benign_dir and len(malignant_dirs) == 0:
        logger.info("Could not detect structure. Organizing all images as leukemia...")
        leukemia_out = wbc_out_dir / "leukemia" / "lymphocytes"
        leukemia_out.mkdir(parents=True, exist_ok=True)
        for img in all_images[:100]:  # Limit
            dest = leukemia_out / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        logger.info(f"✅ Organized {min(100, len(all_images))} images as leukemia")
        return True
    
    # Organize by condition
    organized_count = 0
    
    if benign_dir:
        normal_out = wbc_out_dir / "normal" / "lymphocytes"
        normal_out.mkdir(parents=True, exist_ok=True)
        images = list(benign_dir.rglob("*.jpg")) + \
                 list(benign_dir.rglob("*.png")) + \
                 list(benign_dir.rglob("*.jpeg"))
        logger.info(f"Found {len(images)} benign/normal images")
        for img in images[:100]:  # Sample
            dest = normal_out / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
                organized_count += 1
        logger.info(f"✅ Organized {organized_count} benign images")
    
    # Organize ALL subtypes (Early, Pre, Pro) as leukemia
    if len(malignant_dirs) > 0:
        leukemia_out = wbc_out_dir / "leukemia" / "lymphocytes"
        leukemia_out.mkdir(parents=True, exist_ok=True)
        
        total_leukemia_images = 0
        for malignant_dir in malignant_dirs:
            images = list(malignant_dir.rglob("*.jpg")) + \
                     list(malignant_dir.rglob("*.png")) + \
                     list(malignant_dir.rglob("*.jpeg"))
            total_leukemia_images += len(images)
            
            # Copy samples from each subtype
            for img in images[:50]:  # Sample from each subtype
                dest = leukemia_out / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
                    organized_count += 1
        
        logger.info(f"Found {total_leukemia_images} total leukemia images across {len(malignant_dirs)} subtypes")
        logger.info(f"✅ Organized {organized_count} leukemia images from all subtypes")
    
    return organized_count > 0


def organize_wbc_classification_dataset(wbc_class_dir: Path, wbc_out_dir: Path) -> bool:
    """
    Organize WBC classification dataset by subpopulation.
    
    Expected structure:
    - dataset2-master/
      - dataset2-master/
        - TRAIN/
          - EOSINOPHIL/
          - LYMPHOCYTE/
          - MONOCYTE/
          - NEUTROPHIL/
    """
    logger.info("Organizing WBC classification dataset...")
    
    # Find TRAIN directory
    train_dir = None
    for path in wbc_class_dir.rglob("TRAIN"):
        if path.is_dir():
            train_dir = path
            break
    
    if not train_dir:
        logger.error("TRAIN directory not found in WBC classification dataset")
        return False
    
    # Map subpopulations
    subpop_map = {
        "EOSINOPHIL": "eosinophils",
        "LYMPHOCYTE": "lymphocytes",
        "MONOCYTE": "monocytes",
        "NEUTROPHIL": "neutrophils",
    }
    
    organized = 0
    for subpop_src, subpop_dst in subpop_map.items():
        src_dir = train_dir / subpop_src
        if src_dir.exists():
            dest_dir = wbc_out_dir / "normal" / subpop_dst
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            images = list(src_dir.glob("*.jpg")) + list(src_dir.glob("*.jpeg")) + list(src_dir.glob("*.png"))
            logger.info(f"Found {len(images)} {subpop_src} images")
            
            for img in images[:30]:  # Sample per subpopulation
                dest = dest_dir / img.name
                if not dest.exists():
                    shutil.copy2(img, dest)
            
            organized += len(images[:30])
    
    logger.info(f"✅ Organized {organized} WBC classification images")
    return organized > 0


# ============================================================================
# MAIN DOWNLOAD FUNCTION
# ============================================================================

def download_all_wbc_datasets(force: bool = False):
    """
    Download and organize all WBC datasets.
    """
    WBC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("🩸 Downloading White Blood Cell (WBC) Image Datasets")
    logger.info("=" * 80)
    
    results = {}
    
    # 1. BCCD (already downloaded for RBC)
    bccd_dir = DATA_DIR / "BCCD_Dataset-master"
    if bccd_dir.exists():
        logger.info("\n1️⃣  Organizing BCCD dataset for WBC...")
        results["bccd"] = organize_bccd_for_wbc(bccd_dir, WBC_DATA_DIR)
    else:
        logger.warning("BCCD dataset not found. Run download_malaria_nih.py first?")
        results["bccd"] = False
    
    # 2. Leukemia ALL (Kaggle)
    logger.info("\n2️⃣  Downloading Leukemia ALL dataset...")
    leukemia_dir = WBC_DATA_DIR / "leukemia_ALL_raw"
    if not leukemia_dir.exists() or force:
        if download_via_kaggle("mehradaria/leukemia", leukemia_dir):
            results["leukemia"] = organize_leukemia_dataset(leukemia_dir, WBC_DATA_DIR)
        else:
            logger.warning("Leukemia download failed. Check Kaggle CLI configuration.")
            logger.info(f"Manual download: {WBC_DATASETS['leukemia_ALL_kaggle']['url_manual']}")
            results["leukemia"] = False
    else:
        logger.info("Leukemia dataset already exists. Use --force to re-download.")
        results["leukemia"] = organize_leukemia_dataset(leukemia_dir, WBC_DATA_DIR)
    
    # 3. WBC Classification (Kaggle)
    logger.info("\n3️⃣  Downloading WBC Classification dataset...")
    wbc_class_dir = WBC_DATA_DIR / "wbc_classification_raw"
    if not wbc_class_dir.exists() or force:
        if download_via_kaggle("paultimothymooney/blood-cells", wbc_class_dir):
            results["wbc_class"] = organize_wbc_classification_dataset(wbc_class_dir, WBC_DATA_DIR)
        else:
            logger.warning("WBC Classification download failed.")
            logger.info(f"Manual download: {WBC_DATASETS['wbc_classification']['url_manual']}")
            results["wbc_class"] = False
    else:
        logger.info("WBC Classification dataset already exists. Use --force to re-download.")
        results["wbc_class"] = organize_wbc_classification_dataset(wbc_class_dir, WBC_DATA_DIR)
    
    # 4. Blood Cell Cancer ALL (if exists from previous download)
    blood_cancer_dir = DATA_DIR / "blood_cell_cancer_ALL"
    if blood_cancer_dir.exists():
        logger.info("\n4️⃣  Organizing Blood Cell Cancer ALL dataset...")
        results["blood_cancer"] = organize_leukemia_dataset(blood_cancer_dir, WBC_DATA_DIR)
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 DOWNLOAD SUMMARY")
    logger.info("=" * 80)
    
    for name, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"{status} {name}")
    
    # Check final organization
    check_wbc_datasets()
    
    return results


def check_wbc_datasets():
    """Check which WBC datasets are organized and available."""
    logger.info("\n" + "=" * 80)
    logger.info("📊 AVAILABLE WBC DATASETS")
    logger.info("=" * 80)
    
    if not WBC_DATA_DIR.exists():
        logger.info("No WBC datasets found. Run download first.")
        return
    
    total_images = 0
    
    # Check by condition and subpopulation
    for condition in ["normal", "leukemia", "sepsis", "leukopenia"]:
        condition_dir = WBC_DATA_DIR / condition
        if condition_dir.exists():
            logger.info(f"\n📁 {condition.upper()}:")
            for subpop_dir in sorted(condition_dir.iterdir()):
                if subpop_dir.is_dir():
                    images = list(subpop_dir.glob("*.jpg")) + \
                             list(subpop_dir.glob("*.png")) + \
                             list(subpop_dir.glob("*.jpeg"))
                    n = len(images)
                    total_images += n
                    if n > 0:
                        logger.info(f"  ✅ {subpop_dir.name}: {n} images")
    
    logger.info(f"\n📊 Total images: {total_images}")
    logger.info("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and organize WBC (leukocyte) image datasets"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if datasets exist"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check existing datasets"
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_wbc_datasets()
    else:
        download_all_wbc_datasets(force=args.force)


if __name__ == "__main__":
    main()

