#!/usr/bin/env python3
"""
Atualiza o GitHub Release v1.0.0 com a descrição correta
"""
import subprocess
import sys
from pathlib import Path

REPO = "agourakis82/darwin-pbpk-platform"
TAG = "v1.0.0"
NOTES_FILE = Path(__file__).parent.parent / "RELEASE_DESCRIPTION.md"

def update_release():
    """Atualiza o release usando GitHub API via gh CLI"""
    
    if not NOTES_FILE.exists():
        print(f"❌ Arquivo não encontrado: {NOTES_FILE}")
        return False
    
    # Ler o conteúdo do arquivo
    with open(NOTES_FILE, 'r', encoding='utf-8') as f:
        notes = f.read()
    
    # Criar arquivo temporário
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
        tmp.write(notes)
        tmp_path = tmp.name
    
    try:
        # Usar gh api para atualizar o release
        cmd = [
            'gh', 'api',
            f'repos/{REPO}/releases/tags/{TAG}',
            '-X', 'PATCH',
            '-f', f'body=@{tmp_path}'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Release {TAG} atualizado com sucesso!")
            print(f"   URL: https://github.com/{REPO}/releases/tag/{TAG}")
            return True
        else:
            print(f"❌ Erro ao atualizar release:")
            print(result.stderr)
            return False
    finally:
        # Limpar arquivo temporário
        import os
        os.unlink(tmp_path)

if __name__ == "__main__":
    success = update_release()
    sys.exit(0 if success else 1)

