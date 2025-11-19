#!/usr/bin/env python3
"""
Script para executar a API do Darwin PBPK Platform

Uso:
    python scripts/run_api.py
    python scripts/run_api.py --host 0.0.0.0 --port 8000
    python scripts/run_api.py --reload  # Desenvolvimento

Autor: Dr. Demetrios Chiuratto Agourakis
Criado: 2025-11-08
"""

import argparse
import uvicorn
from pathlib import Path
import sys

# Adicionar path do projeto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Executa a API do Darwin PBPK Platform")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host para bind (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Porta (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Ativar reload automático (desenvolvimento)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Número de workers (default: 1)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Nível de log (default: info)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🚀 Darwin PBPK Platform API")
    print("=" * 80)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Workers: {args.workers}")
    print(f"Log Level: {args.log_level}")
    print("=" * 80)
    print(f"\n📖 Documentação: http://{args.host}:{args.port}/api/v1/docs")
    print(f"🔍 Health Check: http://{args.host}:{args.port}/health")
    print("=" * 80)
    print()

    uvicorn.run(
        "apps.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,  # Reload não funciona com múltiplos workers
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()

