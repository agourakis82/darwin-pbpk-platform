# 🚀 Quick Start: Upload Zenodo via API

Guia rápido para fazer upload dos datasets usando a API do Zenodo.

## ⚡ Setup Rápido (2 minutos)

### 1. Obter Token

```bash
# Acesse e crie um token:
# https://zenodo.org/account/settings/applications/tokens/new/

# Configure (escolha uma opção):
export ZENODO_TOKEN='seu_token_aqui'
# OU
echo 'seu_token_aqui' > ~/.zenodo_token
```

### 2. Preparar Arquivos

```bash
bash scripts/prepare_zenodo_upload.sh
```

### 3. Upload!

```bash
# Produção
python scripts/upload_to_zenodo.py

# OU Sandbox (para testes)
python scripts/upload_to_zenodo.py --sandbox
```

## 📝 Exemplo Completo

```bash
# 1. Preparar arquivos
cd ~/workspace/darwin-pbpk-platform
bash scripts/prepare_zenodo_upload.sh

# 2. Configurar token
export ZENODO_TOKEN='seu_token_zenodo'

# 3. Testar no sandbox primeiro (opcional)
python scripts/upload_to_zenodo.py --sandbox

# 4. Upload em produção
python scripts/upload_to_zenodo.py

# 5. Copiar o DOI retornado e atualizar README
python scripts/update_readme_with_doi.py --doi 10.5281/zenodo.XXXXXX

# 6. Commit
git add README.md RELEASE_DESCRIPTION.md
git commit -m "docs: Add Zenodo dataset DOI"
git push origin main
```

## 🔍 Troubleshooting

### Token não encontrado
```bash
# Verificar se está configurado
echo $ZENODO_TOKEN

# Ou verificar arquivo
cat ~/.zenodo_token
```

### Erro 401 (Unauthorized)
- Verifique se o token está correto
- Verifique se o token tem permissões `deposit:write` e `deposit:actions`

### Erro 413 (File too large)
- Zenodo aceita até 50 GB por arquivo (grátis)
- Se ainda assim falhar, tente upload manual via web

### Testar no Sandbox primeiro
```bash
# Sempre teste no sandbox antes de produção!
python scripts/upload_to_zenodo.py --sandbox --dry-run
```

## 📚 Mais Informações

- Guia completo: `docs/ZENODO_UPLOAD_GUIDE.md`
- Script de upload: `scripts/upload_to_zenodo.py --help`

