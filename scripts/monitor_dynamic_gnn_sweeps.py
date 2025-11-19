#!/usr/bin/env python3
"""
DynamicPBPKGNN Sweep Monitor
Criado: 2025-11-15 14:30 -03
Autor: AI Assistant + Dr. Agourakis

Resumo:
    Lê continuamente os arquivos de log dos sweeps batched
    (B e C por padrão) e imprime snapshots tabulares com
    epoch atual, perdas e melhor validação observada.

Uso:
    python scripts/monitor_dynamic_gnn_sweeps.py \
        --logs models/dynamic_gnn_sweep_b/training.log \
               models/dynamic_gnn_sweep_c/training.log \
        --interval 60 --plot

Notas:
    - Opcionalmente gera gráficos PNG com as curvas de perda
      (mesma convenção usada no notebook) quando --plot é
      habilitado. Os PNGs são gravados ao lado de cada log.
    - Pode ser executado em paralelo aos treinos sem interferir
      (apenas operações de leitura).
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - matplotlib opcional
    plt = None


EPOCH_PATTERN = re.compile(r"Epoch (\d+)/")
TRAIN_PATTERN = re.compile(r"Train Loss: ([0-9.e-]+)")
VAL_PATTERN = re.compile(r"Val Loss: ([0-9.e-]+)")


def parse_log(path: Path) -> pd.DataFrame:
    """Extrai epoch, train e val loss de um log."""
    epochs: List[int] = []
    train_losses: List[float] = []
    val_losses: List[float] = []
    current_epoch = None
    last_train = None

    if not path.exists():
        return pd.DataFrame(columns=["epoch", "train", "val"])

    with path.open() as fh:
        for line in fh:
            if match := EPOCH_PATTERN.search(line):
                current_epoch = int(match.group(1))
            elif match := TRAIN_PATTERN.search(line):
                last_train = float(match.group(1))
            elif match := VAL_PATTERN.search(line):
                if current_epoch is None or last_train is None:
                    continue
                epochs.append(current_epoch)
                train_losses.append(last_train)
                val_losses.append(float(match.group(1)))

    return pd.DataFrame({"epoch": epochs, "train": train_losses, "val": val_losses})


def summarize(df: pd.DataFrame) -> Tuple[str, str]:
    """Gera linhas de resumo textual para impressão."""
    if df.empty:
        return ("(aguardando dados)", "")
    last = df.iloc[-1]
    best_val = df["val"].min()
    summary = (
        f"Epoch {int(last.epoch):>3} | Train {last.train:>9.3e} | "
        f"Val {last.val:>9.3e} | Best {best_val:>9.3e}"
    )
    return (summary, f"total pontos: {len(df)}")


def plot(df: pd.DataFrame, output: Path) -> None:
    """Salva gráfico PNG com as curvas de perda."""
    if df.empty or plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["epoch"], df["train"], label="Train Loss", color="#1f77b4")
    ax.plot(df["epoch"], df["val"], label="Val Loss", color="#d62728")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE por órgão")
    ax.set_title(output.stem.replace("_", " "))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def monitor(logs: Iterable[Path], interval: int, do_plot: bool) -> None:
    """Loop principal de monitoramento."""
    log_paths = list(logs)
    print("Monitoramento iniciado. Ctrl+C para sair.")
    while True:
        print(f"\n=== Snapshot {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
        for log_path in log_paths:
            df = parse_log(log_path)
            summary, extra = summarize(df)
            print(f"[{log_path}] {summary}")
            if extra:
                print(f"    {extra}")
            if do_plot and not df.empty:
                plot_path = log_path.with_suffix(".png")
                plot(df, plot_path)
                print(f"    Gráfico atualizado: {plot_path}")
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor de sweeps DynamicPBPKGNN")
    parser.add_argument(
        "--logs",
        nargs="+",
        default=[
            "models/dynamic_gnn_sweep_b/training.log",
            "models/dynamic_gnn_sweep_c/training.log",
        ],
        help="Lista de arquivos de log a acompanhar.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=120,
        help="Intervalo (segundos) entre snapshots.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Salvar PNG com as curvas de perda para cada log.",
    )
    args = parser.parse_args()
    logs = [Path(p) for p in args.logs]
    monitor(logs, args.interval, args.plot)


if __name__ == "__main__":
    main()

