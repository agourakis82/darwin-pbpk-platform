#!/usr/bin/env python3
"""
Lista e analisa datasets PBPK disponíveis em /mnt/f

Autor: Dr. Demetrios Agourakis + AI Assistant
Data: 2025-11-18
"""

import os
import numpy as np
import json

def find_pbpk_datasets():
    """Encontra todos os datasets NPZ relacionados a PBPK."""
    datasets = []
    
    search_dirs = [
        "/mnt/f/datasets/pbpk",
        "/mnt/f/DARWIN_VALIDATION/datasets",
        "/mnt/f/datasets",
    ]
    
    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.npz') and ('pbpk' in file.lower() or 'dynamic' in file.lower()):
                        full_path = os.path.join(root, file)
                        try:
                            size_mb = os.path.getsize(full_path) / (1024*1024)
                            datasets.append({
                                'path': full_path,
                                'name': file,
                                'size_mb': size_mb,
                                'directory': root
                            })
                        except:
                            pass
    
    return datasets

def analyze_dataset(npz_path):
    """Analisa estrutura de um dataset NPZ."""
    try:
        data = np.load(npz_path)
        info = {
            'keys': list(data.keys()),
            'shapes': {},
            'dtypes': {},
            'num_samples': None
        }
        
        for key in data.keys():
            arr = data[key]
            if hasattr(arr, 'shape'):
                info['shapes'][key] = arr.shape
                info['dtypes'][key] = str(arr.dtype)
                
                # Tentar identificar número de amostras
                if info['num_samples'] is None and len(arr.shape) > 0:
                    info['num_samples'] = arr.shape[0]
        
        return info
    except Exception as e:
        return {'error': str(e)}

def main():
    print("=" * 80)
    print("DATASETS PBPK DISPONÍVEIS EM /mnt/f")
    print("=" * 80)
    print()
    
    datasets = find_pbpk_datasets()
    
    if not datasets:
        print("⚠️  Nenhum dataset NPZ PBPK encontrado em /mnt/f")
        return
    
    print(f"✅ Encontrados {len(datasets)} datasets:")
    print()
    
    results = []
    
    for i, ds in enumerate(datasets, 1):
        print(f"{i}. {ds['name']}")
        print(f"   Caminho: {ds['path']}")
        print(f"   Tamanho: {ds['size_mb']:.1f} MB")
        print(f"   Diretório: {ds['directory']}")
        
        # Analisar estrutura
        info = analyze_dataset(ds['path'])
        if 'error' not in info:
            print(f"   Amostras: {info.get('num_samples', 'N/A')}")
            print(f"   Chaves: {', '.join(info['keys'])}")
            
            results.append({
                'dataset': ds,
                'info': info
            })
        else:
            print(f"   ⚠️  Erro ao analisar: {info['error']}")
        
        print()
    
    # Salvar relatório
    output_dir = 'julia-migration/logs/dataset_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/available_datasets.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Relatório salvo em: {output_dir}/available_datasets.json")
    
    # Recomendar dataset
    if results:
        best = max(results, key=lambda x: x['info'].get('num_samples', 0) if x['info'].get('num_samples') else 0)
        print()
        print("💡 Dataset recomendado para treinamento:")
        print(f"   {best['dataset']['name']}")
        print(f"   Caminho: {best['dataset']['path']}")
        print(f"   Amostras: {best['info'].get('num_samples', 'N/A')}")

if __name__ == '__main__':
    main()
