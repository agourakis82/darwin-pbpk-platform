#!/bin/bash
# Script para preparar arquivos para upload no Zenodo

set -e

echo "📦 Preparando arquivos para upload no Zenodo"
echo "============================================"

# Caminhos dos arquivos
SOURCE_DIR="$HOME/workspace/kec-biomaterials-scaffolds/data/processed"
TEMP_DIR="/tmp/darwin-pbpk-datasets-v1.0.0"
ZIP_FILE="$HOME/workspace/darwin-pbpk-platform/darwin-pbpk-datasets-v1.0.0.zip"

# Criar diretório temporário
mkdir -p "$TEMP_DIR"

echo ""
echo "📋 Arquivos a serem incluídos:"
echo ""

# Copiar arquivos
echo "1. Copiando consolidated_pbpk_v1.parquet..."
cp "$SOURCE_DIR/consolidated/consolidated_pbpk_v1.parquet" "$TEMP_DIR/" && \
    ls -lh "$TEMP_DIR/consolidated_pbpk_v1.parquet"

echo ""
echo "2. Copiando chemberta_embeddings_consolidated.npz..."
cp "$SOURCE_DIR/embeddings/chemberta_768d/chemberta_embeddings_consolidated.npz" "$TEMP_DIR/" && \
    ls -lh "$TEMP_DIR/chemberta_embeddings_consolidated.npz"

echo ""
echo "3. Copiando molecular_graphs.pkl..."
cp "$SOURCE_DIR/molecular_graphs/molecular_graphs.pkl" "$TEMP_DIR/" && \
    ls -lh "$TEMP_DIR/molecular_graphs.pkl"

echo ""
echo "4. Copiando README dos datasets..."
cp "$HOME/workspace/darwin-pbpk-platform/docs/DATASETS_README.md" "$TEMP_DIR/README.md" 2>/dev/null || \
    echo "# Darwin PBPK Platform - Training Datasets v1.0.0

## Contents

- consolidated_pbpk_v1.parquet: Processed PBPK data (44,779 compounds)
- chemberta_embeddings_consolidated.npz: ChemBERTa embeddings (768d)
- molecular_graphs.pkl: Molecular graphs (PyTorch Geometric format)

## Usage

See main repository: https://github.com/agourakis82/darwin-pbpk-platform
" > "$TEMP_DIR/README.md"

# Criar ZIP (opcional, Zenodo aceita arquivos individuais)
echo ""
echo "📦 Criando arquivo ZIP (opcional)..."
cd "$TEMP_DIR"
zip -r "$ZIP_FILE" . -q
echo "✅ ZIP criado: $ZIP_FILE"
echo "   Tamanho: $(du -h "$ZIP_FILE" | cut -f1)"

echo ""
echo "✅ Preparação completa!"
echo ""
echo "📤 Próximos passos:"
echo "1. Acesse: https://zenodo.org/deposit/new"
echo "2. Faça upload dos arquivos de: $TEMP_DIR"
echo "   OU use o ZIP: $ZIP_FILE"
echo "3. Preencha os metadados conforme PROXIMOS_PASSOS.md"
echo ""
echo "💡 Dica: Zenodo aceita upload de arquivos individuais (mais rápido que ZIP)"

