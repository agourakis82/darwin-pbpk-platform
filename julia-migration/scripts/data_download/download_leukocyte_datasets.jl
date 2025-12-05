#!/usr/bin/env julia
"""
Download Leukocyte (WBC) Image Datasets for Fractal Analysis

Downloads datasets of white blood cells organized by subpopulation:
- Normal WBCs (neutrophils, lymphocytes, monocytes, etc.)
- Leukemia (ALL, AML)
- Sepsis (neutrophilia with morphological changes)
- Other pathologies

Author: Darwin PBPK Platform
Date: 2025-12-01
"""

using Downloads
using ZipFile
using HTTP
using JSON

# ============================================================================
# DATASET SOURCES
# ============================================================================

const DATASETS = Dict(
    "leukemia_ALL" => Dict(
        "name" => "Acute Lymphoblastic Leukemia (ALL)",
        "source" => "Kaggle",
        "kaggle_id" => "mehradaria/leukemia",
        "description" => "3,256 images from 89 patients with ALL",
        "classes" => ["Benign", "Malignant_Early_Pre-B", "Malignant_Pre-B", "Malignant_Pro-B"],
        "url_manual" => "https://www.kaggle.com/datasets/mehradaria/leukemia"
    ),
    
    "leukemia_ALL_IDB" => Dict(
        "name" => "ALL-IDB (Acute Lymphoblastic Leukemia)",
        "source" => "University of Milan",
        "description" => "Smaller public dataset with annotated cells",
        "url_manual" => "https://homes.di.unimi.it/scotti/all/",
        "requires_registration" => true
    ),
    
    "wbc_normal_BCCD" => Dict(
        "name" => "BCCD - Normal Blood Cell Count",
        "source" => "GitHub",
        "github_repo" => "Shenggan/BCCD_Dataset",
        "description" => "Normal blood cells including WBCs",
        "url" => "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip"
    ),
    
    "wbc_classification" => Dict(
        "name" => "Blood Cell Classification Dataset",
        "source" => "Kaggle",
        "kaggle_id" => "paultimothymooney/blood-cells",
        "description" => "Classified WBCs: Eosinophil, Lymphocyte, Monocyte, Neutrophil",
        "url_manual" => "https://www.kaggle.com/datasets/paultimothymooney/blood-cells"
    ),
    
    "wbc_sepsis" => Dict(
        "name" => "Sepsis - Neutrophil Morphology",
        "source" => "Literature/Manual Collection",
        "description" => "Neutrophils from septic patients showing morphological changes",
        "note" => "May require manual collection from published papers"
    )
)

# ============================================================================
# DOWNLOAD FUNCTIONS
# ============================================================================

"""
download_bccd_dataset(data_dir)

Download BCCD dataset which contains normal blood cells including WBCs.
"""
function download_bccd_dataset(data_dir::String)
    data_path = joinpath(data_dir, "leukocytes", "bccd_normal")
    mkpath(data_path)
    
    bccd_zip = joinpath(data_dir, "bccd.zip")
    bccd_url = "https://github.com/Shenggan/BCCD_Dataset/archive/refs/heads/master.zip"
    
    # Check if already downloaded
    extracted_dir = joinpath(data_dir, "leukocytes", "BCCD_Dataset-master")
    if isdir(extracted_dir)
        println("✅ BCCD dataset already exists at $extracted_dir")
        return true
    end
    
    println("📥 Downloading BCCD dataset (normal blood cells)...")
    
    try
        Downloads.download(bccd_url, bccd_zip; timeout=300.0)
        println("✅ Download complete. Extracting...")
        
        # Extract
        z = ZipFile.Reader(bccd_zip)
        for f in z.files
            # Skip __MACOSX files
            if occursin("__MACOSX", f.name)
                continue
            end
            
            # Extract to data_dir
            out_path = joinpath(data_dir, f.name)
            mkpath(dirname(out_path))
            
            if !endswith(f.name, "/")
                write(out_path, read(f))
            end
        end
        close(z)
        
        rm(bccd_zip)  # Remove zip
        println("✅ BCCD dataset extracted successfully")
        return true
        
    catch e
        println("❌ Error downloading BCCD: $e")
        return false
    end
end

"""
download_via_kaggle(kaggle_id, dest_dir)

Download dataset from Kaggle using kaggle CLI (requires authentication).
"""
function download_via_kaggle(kaggle_id::String, dest_dir::String)::Bool
    println("📥 Attempting to download $kaggle_id from Kaggle...")
    println("   Note: Requires Kaggle CLI and API token")
    
    # Check if kaggle CLI is available
    try
        run(`which kaggle`)
    catch
        println("⚠️  Kaggle CLI not found. Install with: pip install kaggle")
        println("   Then create API token at: https://www.kaggle.com/settings")
        println("   Save kaggle.json to ~/.kaggle/kaggle.json")
        return false
    end
    
    mkpath(dest_dir)
    
    try
        run(`kaggle datasets download -d $kaggle_id -p $dest_dir`)
        println("✅ Download complete")
        
        # Extract zip files
        for zip_file in readdir(dest_dir)
            if endswith(zip_file, ".zip")
                zip_path = joinpath(dest_dir, zip_file)
                println("   Extracting $zip_file...")
                
                z = ZipFile.Reader(zip_path)
                for f in z.files
                    if occursin("__MACOSX", f.name)
                        continue
                    end
                    out_path = joinpath(dest_dir, f.name)
                    mkpath(dirname(out_path))
                    if !endswith(f.name, "/")
                        write(out_path, read(f))
                    end
                end
                close(z)
                rm(zip_path)
            end
        end
        
        return true
    catch e
        println("❌ Error downloading from Kaggle: $e")
        return false
    end
end

"""
show_manual_download_instructions(dataset_name)

Show instructions for manual download of datasets.
"""
function show_manual_download_instructions(dataset_name::String)
    if !haskey(DATASETS, dataset_name)
        println("Unknown dataset: $dataset_name")
        return
    end
    
    dataset = DATASETS[dataset_name]
    data_dir = joinpath(@__DIR__, "..", "..", "analysis", "fractal_poc", "data", "leukocytes")
    
    println("\n" * "=" ^ 80)
    println("MANUAL DOWNLOAD INSTRUCTIONS: $(dataset["name"])")
    println("=" ^ 80)
    println("\nDescription: $(dataset["description"])")
    
    if haskey(dataset, "url_manual")
        println("\n1. Go to: $(dataset["url_manual"])")
    end
    
    if haskey(dataset, "kaggle_id")
        println("\n   OR use Kaggle CLI:")
        println("   pip install kaggle")
        println("   # Create API token at: https://www.kaggle.com/settings")
        println("   kaggle datasets download -d $(dataset["kaggle_id"])")
    end
    
    println("\n2. Extract to: $data_dir/$dataset_name")
    
    if haskey(dataset, "classes")
        println("\n3. Expected structure:")
        for class in dataset["classes"]
            println("   - $class/")
        end
    end
    
    println("\n" * "=" ^ 80)
end

"""
check_dataset_availability(data_dir)

Check which leukocyte datasets are already downloaded.
"""
function check_dataset_availability(data_dir::String)
    leukocyte_dir = joinpath(data_dir, "leukocytes")
    
    datasets_checked = Dict(
        "BCCD (Normal WBCs)" => joinpath(leukocyte_dir, "BCCD_Dataset-master"),
        "Leukemia ALL" => joinpath(leukocyte_dir, "leukemia_ALL"),
        "WBC Classification" => joinpath(leukocyte_dir, "wbc_classification")
    )
    
    println("\n" * "=" ^ 80)
    println("📊 AVAILABLE LEUKOCYTE DATASETS")
    println("=" ^ 80)
    
    total_images = 0
    for (name, path) in datasets_checked
        if isdir(path)
            # Count images
            images = [f for f in readdir(path, join=true, recursive=true) 
                     if endswith(lowercase(f), ".jpg") || 
                        endswith(lowercase(f), ".png") ||
                        endswith(lowercase(f), ".bmp")]
            n = length(images)
            total_images += n
            println("✅ $name: $n images")
            println("   Location: $path")
        else
            println("❌ $name: Not found")
        end
    end
    
    println("\nTotal images: $total_images")
    println("=" ^ 80)
    
    return total_images
end

"""
download_all_leukocyte_datasets(data_dir; interactive=true)

Download all available leukocyte datasets.
"""
function download_all_leukocyte_datasets(data_dir::String; interactive::Bool=true)
    leukocyte_dir = joinpath(data_dir, "leukocytes")
    mkpath(leukocyte_dir)
    
    println("🩸 Downloading Leukocyte (WBC) Image Datasets")
    println("=" ^ 80)
    println()
    
    results = Dict()
    
    # 1. BCCD (normal WBCs)
    println("1️⃣  BCCD Dataset (Normal Blood Cells)...")
    results["bccd"] = download_bccd_dataset(data_dir)
    println()
    
    # 2. Leukemia ALL (via Kaggle)
    println("2️⃣  Leukemia ALL Dataset...")
    leukemia_dir = joinpath(leukocyte_dir, "leukemia_ALL")
    if !isdir(leukemia_dir) || isempty(readdir(leukemia_dir))
        results["leukemia"] = download_via_kaggle("mehradaria/leukemia", leukemia_dir)
        if !results["leukemia"]
            show_manual_download_instructions("leukemia_ALL")
        end
    else
        println("✅ Leukemia dataset already exists")
        results["leukemia"] = true
    end
    println()
    
    # 3. WBC Classification (via Kaggle)
    println("3️⃣  WBC Classification Dataset...")
    wbc_class_dir = joinpath(leukocyte_dir, "wbc_classification")
    if !isdir(wbc_class_dir) || isempty(readdir(wbc_class_dir))
        results["wbc_class"] = download_via_kaggle("paultimothymooney/blood-cells", wbc_class_dir)
        if !results["wbc_class"]
            show_manual_download_instructions("wbc_classification")
        end
    else
        println("✅ WBC Classification dataset already exists")
        results["wbc_class"] = true
    end
    println()
    
    # Summary
    println("=" ^ 80)
    println("📊 DOWNLOAD SUMMARY")
    println("=" ^ 80)
    
    for (name, success) in results
        status = success ? "✅" : "❌"
        println("$status $name")
    end
    
    println()
    check_dataset_availability(data_dir)
    
    return results
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    # Determine data directory
    script_dir = @__DIR__
    project_root = joinpath(script_dir, "..", "..")
    data_dir = joinpath(project_root, "analysis", "fractal_poc", "data")
    mkpath(data_dir)
    
    println("🎯 Darwin PBPK - Leukocyte Dataset Downloader")
    println("=" ^ 80)
    println()
    println("Data directory: $data_dir")
    println()
    
    # Check existing datasets
    total = check_dataset_availability(data_dir)
    
    if total == 0
        println("\n⚠️  No datasets found. Starting downloads...")
        download_all_leukocyte_datasets(data_dir)
    else
        println("\n✅ Found $total images. Use --force to re-download.")
        println("   Run: julia download_leukocyte_datasets.jl --all")
    end
    
    return true
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

