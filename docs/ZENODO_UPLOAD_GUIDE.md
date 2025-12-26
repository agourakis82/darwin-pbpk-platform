# 📤 Guia Completo: Upload Datasets no Zenodo

**Data:** 06 de Novembro de 2025  
**Status:** Preparado para upload (API disponível!)

---

## ✅ PRÉ-REQUISITOS

1. ✅ Conta Zenodo criada (login com GitHub)
2. ✅ Zenodo conectado ao GitHub (Settings → GitHub → Sync)
3. ✅ Arquivos preparados (usar `scripts/prepare_zenodo_upload.sh`)

---

## 🚀 OPÇÃO 1: UPLOAD VIA API (RECOMENDADO - AUTOMÁTICO!)

### Passo 1: Obter Token do Zenodo

1. Acesse: https://zenodo.org/account/settings/applications/tokens/new/
   - (Sandbox para testes: https://sandbox.zenodo.org/account/settings/applications/tokens/new/)
2. Crie um token com permissões:
   - ✅ `deposit:write`
   - ✅ `deposit:actions`
3. Configure o token (escolha uma opção):

**Opção A: Variável de ambiente**
```bash
export ZENODO_TOKEN='seu_token_aqui'
```

**Opção B: Arquivo de configuração**
```bash
echo 'seu_token_aqui' > ~/.zenodo_token
chmod 600 ~/.zenodo_token
```

**Opção C: Passar via linha de comando**
```bash
python scripts/upload_to_zenodo.py --token seu_token_aqui
```

### Passo 2: Preparar Arquivos

```bash
cd ~/workspace/darwin-pbpk-platform
bash scripts/prepare_zenodo_upload.sh
```

### Passo 3: Fazer Upload (AUTOMÁTICO!)

**Produção:**
```bash
python scripts/upload_to_zenodo.py
```

**Sandbox (para testes):**
```bash
python scripts/upload_to_zenodo.py --sandbox
```

**Dry-run (simular sem fazer upload):**
```bash
python scripts/upload_to_zenodo.py --dry-run
```

O script irá:
1. ✅ Criar depósito no Zenodo
2. ✅ Fazer upload de todos os arquivos
3. ✅ Preencher metadados automaticamente
4. ✅ Publicar o depósito
5. ✅ Retornar o DOI

### Passo 4: Atualizar README com DOI

Após obter o DOI, atualize automaticamente:

```bash
python scripts/update_readme_with_doi.py --doi 10.5281/zenodo.XXXXXX
```

Ou manualmente (veja Opção 2 abaixo).

---

## 📋 OPÇÃO 2: UPLOAD MANUAL (VIA INTERFACE WEB)

### PASSO 1: Preparar Arquivos (2 min)

Execute o script de preparação:

```bash
cd ~/workspace/darwin-pbpk-platform
bash scripts/prepare_zenodo_upload.sh
```

Isso criará:
- Diretório temporário: `/tmp/darwin-pbpk-datasets-v1.0.0/`
- ZIP opcional: `darwin-pbpk-datasets-v1.0.0.zip`

**Arquivos a fazer upload:**
1. `consolidated_pbpk_v1.parquet` (~1.5 MB)
2. `chemberta_embeddings_consolidated.npz` (~123 MB)
3. `molecular_graphs.pkl` (~286 MB)
4. `README.md` (do diretório docs/)

**Total:** ~410 MB

---

### PASSO 2: Acessar Zenodo (1 min)

1. Acesse: https://zenodo.org/deposit/new
2. Faça login (se necessário)
3. Selecione: **"New Upload"** → **"Dataset"**

---

### PASSO 3: Upload de Arquivos (5-10 min)

**Opção A: Upload Individual (Recomendado)**
- Clique em **"Choose files"**
- Selecione os 4 arquivos de `/tmp/darwin-pbpk-datasets-v1.0.0/`
- Aguarde upload completo (~410 MB)

**Opção B: Upload ZIP**
- Use o arquivo: `darwin-pbpk-datasets-v1.0.0.zip`
- Mais rápido, mas menos flexível

**💡 Dica:** Zenodo aceita uploads grandes (até 50 GB grátis)

---

### PASSO 4: Preencher Metadados (5 min)

#### Basic Information

**Title:**
```
Darwin PBPK Platform - Training Datasets v1.0.0
```

**Upload type:**
```
Dataset
```

**Publication date:**
```
2025-11-05
```

**Description:**
```markdown
Training datasets for Darwin PBPK Platform v1.0.0, including:

- **consolidated_pbpk_v1.parquet**: Processed PBPK data for 44,779 compounds (ChEMBL + TDC + KEC)
- **chemberta_embeddings_consolidated.npz**: ChemBERTa embeddings (768d, 44,779 molecules)
- **molecular_graphs.pkl**: Molecular graphs in PyTorch Geometric format

**Dataset Details:**
- Total compounds: 44,779
- Train/Val/Test split: 80/10/10 (scaffold-based, zero leakage)
- PBPK parameters: Fu, Vd, CL
- Sources: ChEMBL, TDC (Therapeutics Data Commons), KEC

**Related Software:**
- Repository: https://github.com/agourakis82/darwin-pbpk-platform
- Software DOI: https://doi.org/10.5281/zenodo.17536674
```

#### Creators

**Name:**
```
Sounio Chiuratto Agourakis
```

**Affiliation:**
```
PUCRS - Pontifícia Universidade Católica do Rio Grande do Sul
```

**ORCID (se tiver):**
```
[Seu ORCID]
```

#### Related Identifiers

**Identifier:**
```
10.5281/zenodo.17536674
```

**Relation:**
```
IsSupplementTo
```

**Resource type:**
```
Software
```

#### License

**License:**
```
Creative Commons Attribution 4.0 International (CC-BY-4.0)
```

#### Keywords

Adicione:
```
pharmacokinetics
PBPK
machine learning
drug discovery
ADMET
ChEMBL
molecular graphs
ChemBERTa
```

#### Communities

Opcional: Adicione comunidades relevantes (ex: "Machine Learning", "Chemistry")

---

### PASSO 5: Publicar (1 min)

1. Revise todos os metadados
2. Clique em **"Publish"**
3. ⚠️ **ATENÇÃO:** Após publicar, não é possível editar facilmente!

---

### PASSO 6: Copiar DOI (1 min)

Após publicação:
1. Zenodo gerará automaticamente um DOI
2. Formato: `10.5281/zenodo.XXXXXX`
3. **COPIE ESTE DOI!** Você precisará dele para atualizar o README

---

## 🔄 ATUALIZAR REPOSITÓRIO

Após obter o DOI dos datasets:

### 1. Atualizar README.md

Substituir `YYYYYY` pelo DOI real:

```bash
cd ~/workspace/darwin-pbpk-platform
# Editar README.md linha 59
sed -i 's/zenodo.YYYYYY/zenodo.XXXXXX/g' README.md
```

### 2. Atualizar RELEASE_DESCRIPTION.md

```bash
# Editar RELEASE_DESCRIPTION.md linha 84
sed -i 's/zenodo.YYYYYY/zenodo.XXXXXX/g' RELEASE_DESCRIPTION.md
```

### 3. Commit e Push

```bash
git add README.md RELEASE_DESCRIPTION.md
git commit -m "docs: Add Zenodo dataset DOI"
git push origin main
```

---

## ✅ CHECKLIST FINAL

- [ ] Arquivos preparados (`prepare_zenodo_upload.sh` executado)
- [ ] Upload concluído no Zenodo
- [ ] Metadados preenchidos corretamente
- [ ] Dataset publicado
- [ ] DOI copiado
- [ ] README.md atualizado com DOI
- [ ] RELEASE_DESCRIPTION.md atualizado
- [ ] Mudanças commitadas e enviadas para GitHub

---

## 🎯 RESULTADO ESPERADO

Após completar todos os passos:

✅ **DOI Software:** `10.5281/zenodo.17536674` (já obtido)  
✅ **DOI Datasets:** `10.5281/zenodo.XXXXXX` (a obter)  
✅ **README atualizado** com ambos DOIs  
✅ **Paper-ready** para citação em Nature Machine Intelligence

---

## 📚 CITAÇÃO NO PAPER

### Code Availability

```
The complete source code for Darwin PBPK Platform v1.0.0 is freely 
available at https://doi.org/10.5281/zenodo.17536674 under MIT License.
```

### Data Availability

```
Training datasets (ChemBERTa embeddings, molecular graphs, processed 
parquets, 44,779 compounds) are available at 
https://doi.org/10.5281/zenodo.XXXXXX under CC-BY-4.0 License.
```

---

**"Rigorous science. Honest results. Real impact."**

