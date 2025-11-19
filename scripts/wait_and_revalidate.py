#!/usr/bin/env python3
"""
Aguarda fine-tuning terminar e executa revalidação automaticamente
"""
import time
import subprocess
from pathlib import Path
import sys

def check_finetuning_complete(checkpoint_path: Path, log_path: Path, max_wait_hours: int = 24) -> bool:
    """Verifica se fine-tuning terminou"""
    # Verificar se checkpoint existe
    if checkpoint_path.exists():
        # Verificar se log contém "Fine-tuning concluído"
        if log_path.exists():
            with open(log_path, 'r') as f:
                content = f.read()
                if "Fine-tuning concluído" in content or "✅ Fine-tuning concluído" in content:
                    return True
    return False

def main():
    finetuned_checkpoint = Path("models/dynamic_gnn_v4_compound/finetuned/best_finetuned_model.pt")
    finetuning_log = Path("models/dynamic_gnn_v4_compound/finetuned/finetuning.log")

    print("⏳ Aguardando fine-tuning terminar...")
    print(f"   Checkpoint esperado: {finetuned_checkpoint}")
    print(f"   Log: {finetuning_log}")

    wait_interval = 300  # 5 minutos
    max_wait = 24 * 3600  # 24 horas

    start_time = time.time()
    while time.time() - start_time < max_wait:
        if check_finetuning_complete(finetuned_checkpoint, finetuning_log):
            print("\n✅ Fine-tuning concluído! Iniciando revalidação...")
            break
        time.sleep(wait_interval)
        elapsed = (time.time() - start_time) / 60
        print(f"   Aguardando... ({elapsed:.1f} minutos)")
    else:
        print("\n⚠️  Timeout: Fine-tuning não terminou no tempo esperado")
        sys.exit(1)

    # Executar revalidação
    cmd = [
        "python", "scripts/revalidate_after_finetuning.py",
        "--original-checkpoint", "models/dynamic_gnn_v4_compound/best_model.pt",
        "--finetuned-checkpoint", str(finetuned_checkpoint),
        "--calibration-results", "models/dynamic_gnn_v4_compound/calibration/calibration_results.json",
        "--experimental-data", "data/processed/pbpk_enriched/audited/experimental_validation_data_refined.npz",
        "--experimental-metadata", "data/processed/pbpk_enriched/audited/experimental_validation_data_audited.metadata.json",
        "--output-dir", "models/dynamic_gnn_v4_compound/revalidation",
        "--device", "cuda",
    ]

    print("\n🔬 Executando revalidação...")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        print("\n✅ Revalidação concluída com sucesso!")
    else:
        print("\n❌ Erro na revalidação")
        sys.exit(1)

if __name__ == "__main__":
    main()

