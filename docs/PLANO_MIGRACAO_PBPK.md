# 🚀 PLANO MIGRAÇÃO DARWIN PBPK PLATFORM

**Data:** 05 de Novembro de 2025  
**Decisão:** Migrar PBPK hoje (Opção A)  
**Tempo Estimado:** 1-2 horas

---

## 📋 ORDEM DE EXECUÇÃO

### FASE 1: Completar Darwin (25 min) ← PRIMEIRO!

Antes de migrar PBPK, COMPLETAR Darwin:

1. ✅ Abrir nova janela Cursor (darwin-scaffold-studio)
2. ✅ Criar GitHub Release v1.0.0
3. ✅ Conectar Zenodo
4. ✅ Aguardar DOI
5. ✅ Atualizar badge README

**Motivo:** Um de cada vez, sem confusão!

### FASE 2: Migrar PBPK (1-2h) ← DEPOIS!

Após Darwin 100% completo, migrar PBPK.

---

## 📊 ARQUIVOS PBPK A MIGRAR

### Scripts de Training (~2,000 linhas)

```
scripts/phase2_training/
├── 01_baseline_mlp.py (~400 linhas)
├── 02_gnn_model.py (~500 linhas)
├── 03_ensemble.py (~300 linhas, se existir)
└── launch_parallel.sh (~100 linhas)
```

### Data Pipeline (~2,500 linhas)

```
scripts/data_pipeline/
├── 01_consolidate_datasets.py (~500 linhas)
├── 02_generate_chemberta_embeddings.py (~400 linhas)
├── 03_create_molecular_graphs.py (~600 linhas)
├── 04_compute_rdkit_descriptors.py (~500 linhas)
└── 05_scaffold_split.py (~300 linhas)
```

### Dados Processados (GRANDES!)

```
data/processed/
├── consolidated/
│   └── consolidated_pbpk_v1.parquet (~100 MB)
├── embeddings/chemberta_768d/
│   └── chemberta_embeddings_consolidated.npz (~1 GB)
├── molecular_graphs/
│   └── molecular_graphs.pkl (~500 MB)
├── rdkit_descriptors/
│   └── rdkit_descriptors.parquet (~50 MB)
└── splits/
    ├── train.parquet
    ├── val.parquet
    └── test.parquet
```

**ATENÇÃO:** ~1.7 GB total! GitHub tem limite 100 MB/arquivo!

### Modelos Treinados (GRANDES!)

```
models/
├── baseline_mlp/
│   ├── best_model.pt (~2 MB)
│   ├── results.json
│   └── training_history.json
└── gnn_model/
    ├── best_model.pt (~6 MB)
    ├── results.json
    └── training_history.json
```

### Logs

```
logs/parallel_training/
├── mlp_20251028_150532.log
└── gnn_20251028_*.log
```

### Documentação PBPK

```
docs/ (filtrar apenas PBPK):
- STACK_PBPK.md (se existir)
- METRICAS_PBPK.md (se existir)
- Outros PBPK-específicos
```

**TOTAL ESTIMADO:** ~5,000+ linhas código + ~1.7 GB dados

---

## ⚠️ PROBLEMA: ARQUIVOS GRANDES

GitHub limita:
- ❌ 100 MB por arquivo
- ❌ 1 GB total repo (soft limit)

**Solução:**

### Opção A: Git LFS (Large File Storage)

```bash
# Instalar Git LFS
git lfs install

# Track arquivos grandes
git lfs track "*.npz"
git lfs track "*.pkl"
git lfs track "*.parquet"
git lfs track "*.pt"
```

**Pros:**
- ✅ Tudo no GitHub
- ✅ Versionamento completo
- ✅ Download sob demanda

**Cons:**
- ⚠️ 1 GB gratuito/mês
- ⚠️ $5/mês para 50 GB

### Opção B: Zenodo para Datasets (RECOMENDADO!)

```
GitHub: Código apenas (~5,000 linhas)
Zenodo: Datasets grandes (1.7 GB)
```

**Workflow:**
1. Código → GitHub → DOI Software
2. Datasets → Zenodo separado → DOI Data
3. README.md → Link para ambos DOIs

**Pros:**
- ✅ GitHub leve (só código)
- ✅ Zenodo ilimitado (GRÁTIS!)
- ✅ 2 DOIs (software + data)
- ✅ Melhor para papers Q1 (Nature prefere!)

**Exemplo Nature:**
```
Code Availability: DOI 10.5281/zenodo.XXXXXX (software)
Data Availability: DOI 10.5281/zenodo.YYYYYY (datasets)
```

### Opção C: Dados Sintéticos/Demo

```
GitHub: Código + dados PEQUENOS de demo
Zenodo: Software DOI
README: "Full dataset available upon request"
```

**Pros:**
- ✅ Repo leve
- ✅ Demos funcionam
- ✅ Proteção IP (se necessário)

---

## 🎯 RECOMENDAÇÃO PARA PBPK

### Estratégia Dual-DOI (IDEAL PARA Q1!)

```
darwin-pbpk-platform (GitHub):
├── scripts/ (código apenas, ~5,000 linhas)
├── data/demo/ (dados pequenos de exemplo)
├── models/ (pequenos, <10 MB)
├── docs/
├── .zenodo.json (software metadata)
└── README.md

DOI Software: 10.5281/zenodo.XXXXXX
Download: Código completo

──────────────────────────────────────────

darwin-pbpk-datasets (Zenodo separado):
├── consolidated_pbpk_v1.parquet (100 MB)
├── chemberta_embeddings.npz (1 GB)
├── molecular_graphs.pkl (500 MB)
└── README.txt (descrição)

DOI Data: 10.5281/zenodo.YYYYYY
Download: Datasets completos
```

**Paper cita AMBOS:**
```
Code: DOI 10.5281/zenodo.XXXXXX
Data: DOI 10.5281/zenodo.YYYYYY
```

**Vantagens:**
- ✅ GitHub leve e rápido
- ✅ Zenodo ilimitado (GRÁTIS!)
- ✅ 2 DOIs (alinha com Nature/Science best practices)
- ✅ Datasets preservados permanentemente
- ✅ Software e dados versionados independente

---

## 📦 ESTRUTURA DARWIN PBPK PLATFORM

### Diretórios:

```
darwin-pbpk-platform/
├── apps/
│   ├── training/
│   │   ├── baseline_mlp.py
│   │   ├── gnn_model.py
│   │   └── ensemble.py
│   ├── prediction/
│   │   └── pbpk_predictor.py
│   └── api/
│       └── pbpk_api.py (se existir)
├── data/
│   ├── demo/ (PEQUENOS exemplos)
│   │   └── demo_molecules.parquet (1,000 moléculas)
│   └── README.md (link para DOI datasets)
├── models/
│   └── demo/ (modelos pequenos)
├── scripts/
│   ├── data_pipeline/
│   │   ├── 01_consolidate_datasets.py
│   │   ├── 02_generate_chemberta_embeddings.py
│   │   ├── 03_create_molecular_graphs.py
│   │   ├── 04_compute_rdkit_descriptors.py
│   │   └── 05_scaffold_split.py
│   └── training/
│       └── launch_parallel.sh
├── docs/
│   ├── PBPK_METHODS.md
│   ├── FAZER_RELEASE_ZENODO.md
│   └── INSTRUCOES_TREINAMENTO.md
├── tests/
├── notebooks/ (exemplos)
├── README.md (drug discovery focus)
├── requirements.txt (torch, rdkit, transformers)
├── LICENSE (MIT)
├── .gitignore
├── .zenodo.json (PhysioQM, ADMET refs)
└── CITATION.cff
```

---

## 🚀 SCRIPT DE MIGRAÇÃO PBPK

### Criar Script Automático:

```bash
scripts/migrate_to_pbpk_platform.sh
```

Similar ao `migrate_to_darwin_studio.sh`, mas:
- Copia arquivos PBPK específicos
- Cria estrutura PBPK
- Gera README focado em drug discovery
- requirements.txt com torch, rdkit, transformers

---

## ⏱️ CRONOGRAMA HOJE

### Manhã/Tarde Atual:

**13:00-13:25 (25 min):** Completar Darwin
- GitHub Release v1.0.0
- Zenodo connect
- Aguardar DOI
- Badge README

**13:25-14:00 (35 min):** Preparar PBPK
- Criar script migração PBPK
- Identificar arquivos a copiar
- Criar repo GitHub

**14:00-15:30 (1.5h):** Executar Migração PBPK
- Copiar código (~5,000 linhas)
- Criar README, requirements
- Commit + tag v1.0.0
- Zenodo connect
- Aguardar DOI

**15:30:** ✅ Arquitetura completa!

---

## 🎊 RESULTADO FINAL ESPERADO

### Repositórios (6 total):

```
CIENTÍFICOS COM DOI (4):
1. ✅ darwin-scaffold-studio       v1.0.0 | DOI: XXXXXX
2. ✅ darwin-pbpk-platform         v1.0.0 | DOI: YYYYYY
3. ✅ pcs-meta-repo                v2.3.1 | DOI: ZZZZZZ (futuro)
4. ✅ hyperbolic-semantic          v0.8.0 | DOI: WWWWWW (futuro)

COMERCIAL SEM DOI (1):
5. ✅ chiuratto-AI                 v1.5.0 | N/A

META-REPO (1):
6. ✅ kec-biomaterials-scaffolds   (coord)
```

### Papers Q1 Prontos:

- ✅ **Paper 1:** Tissue Engineering → Cita Darwin Scaffolds DOI
- ✅ **Paper 2:** Drug Discovery → Cita Darwin PBPK DOI
- ✅ Cada paper com citação LIMPA e ESPECÍFICA!

---

## 📄 Documento Criado:

**`ARQUITETURA_FINAL_MULTI_REPO.md`** (700+ linhas)
- Por que PBPK separado
- Estrutura completa 6 repos
- Plano migração PBPK
- Dual-DOI strategy (code + data)

---

## 🎯 Próximos Passos Imediatos:

**AGORA (você):**
1. Abrir nova janela Cursor → darwin-scaffold-studio
2. Seguir PROXIMOS_PASSOS.md
3. Completar release Darwin + DOI (25 min)

**DEPOIS (eu):**
1. Criar script migração PBPK
2. Você executa migração PBPK (1-2h)
3. ✅ Arquitetura completa!

---

**Vamos começar! Abra nova janela Cursor para Darwin primeiro!** 🚀

**"Ciência rigorosa. Resultados honestos. Impacto real."**
