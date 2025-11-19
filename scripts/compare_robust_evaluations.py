#!/usr/bin/env python3
"""
Compara avaliações robustas de múltiplos modelos
Criado: 2025-11-16 22:00 -03
Autor: AI Assistant + Dr. Agourakis
"""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def load_eval_json(path: Path) -> dict:
    """Carrega JSON de avaliação robusta."""
    with open(path, 'r') as f:
        return json.load(f)

def create_comparison_table(eval_paths: dict) -> pd.DataFrame:
    """Cria tabela comparativa de R² por janela e modelo."""
    rows = []
    for model_name, eval_path in eval_paths.items():
        data = load_eval_json(eval_path)
        for win_key in data['model_linear'].keys():
            rows.append({
                'Modelo': model_name,
                'Janela': win_key,
                'R² (linear)': data['model_linear'][win_key]['r2'],
                'R² (log1p)': data['model_log1p'][win_key]['r2'],
                'MSE (linear)': data['model_linear'][win_key]['mse'],
                'MAE (linear)': data['model_linear'][win_key]['mae'],
                'Baseline Mean R² (linear)': data['baseline_mean_linear'][win_key]['r2'],
                'Baseline Mean R² (log1p)': data['baseline_mean_log1p'][win_key]['r2'],
            })
    return pd.DataFrame(rows)

def plot_comparison(df: pd.DataFrame, output_path: Path):
    """Gera gráficos comparativos."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # R² linear por janela
    ax1 = axes[0, 0]
    pivot_lin = df.pivot(index='Janela', columns='Modelo', values='R² (linear)')
    pivot_lin.plot(kind='bar', ax=ax1, rot=0)
    ax1.set_title('R² (escala linear) por janela temporal')
    ax1.set_ylabel('R²')
    ax1.legend(title='Modelo')
    ax1.grid(True, alpha=0.3)

    # R² log1p por janela
    ax2 = axes[0, 1]
    pivot_log = df.pivot(index='Janela', columns='Modelo', values='R² (log1p)')
    pivot_log.plot(kind='bar', ax=ax2, rot=0)
    ax2.set_title('R² (log1p) por janela temporal')
    ax2.set_ylabel('R²')
    ax2.legend(title='Modelo')
    ax2.grid(True, alpha=0.3)

    # MSE por janela
    ax3 = axes[1, 0]
    pivot_mse = df.pivot(index='Janela', columns='Modelo', values='MSE (linear)')
    pivot_mse.plot(kind='bar', ax=ax3, rot=0, logy=True)
    ax3.set_title('MSE (escala log) por janela temporal')
    ax3.set_ylabel('MSE (log scale)')
    ax3.legend(title='Modelo')
    ax3.grid(True, alpha=0.3)

    # Comparação modelo vs baseline mean
    ax4 = axes[1, 1]
    models = df['Modelo'].unique()
    for model in models:
        model_df = df[df['Modelo'] == model]
        ax4.plot(model_df['Janela'], model_df['R² (linear)'], 'o-', label=f'{model} (modelo)')
        ax4.plot(model_df['Janela'], model_df['Baseline Mean R² (linear)'], 's--', alpha=0.5, label=f'{model} (baseline mean)')
    ax4.set_title('Modelo vs Baseline Mean (R² linear)')
    ax4.set_ylabel('R²')
    ax4.set_xlabel('Janela temporal')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print(f"✅ Gráfico salvo: {output_path}")

def main():
    ap = argparse.ArgumentParser(description="Compara avaliações robustas de múltiplos modelos")
    ap.add_argument("--evals", nargs='+', required=True, help="Lista de pares 'nome:path' para cada avaliação")
    ap.add_argument("--output", required=True, help="Diretório de saída")
    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # Parse eval paths
    eval_paths = {}
    for pair in args.evals:
        name, path = pair.split(':', 1)
        eval_paths[name] = Path(path)
        if not eval_paths[name].exists():
            raise FileNotFoundError(f"Avaliação não encontrada: {eval_paths[name]}")

    # Criar tabela comparativa
    df = create_comparison_table(eval_paths)

    # Salvar CSV
    csv_path = outdir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Tabela salva: {csv_path}")

    # Salvar JSON
    json_path = outdir / "comparison_summary.json"
    summary = {
        'models': list(eval_paths.keys()),
        'windows': df['Janela'].unique().tolist(),
        'r2_linear_mean': df.groupby('Modelo')['R² (linear)'].mean().to_dict(),
        'r2_log1p_mean': df.groupby('Modelo')['R² (log1p)'].mean().to_dict(),
        'mse_mean': df.groupby('Modelo')['MSE (linear)'].mean().to_dict(),
    }
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Resumo JSON salvo: {json_path}")

    # Gerar gráficos
    plot_path = outdir / "comparison_plots.png"
    plot_comparison(df, plot_path)

    # Print resumo
    print("\n📊 RESUMO COMPARATIVO:")
    print(df.groupby('Modelo')[['R² (linear)', 'R² (log1p)', 'MSE (linear)']].mean().to_string())

if __name__ == "__main__":
    main()


