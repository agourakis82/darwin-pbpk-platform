# 🚨 Pending Issues - PBPK & Repository Cleanup

**Date:** 2025-11-02  
**Status:** 2 critical issues identified

---

## 🔴 ISSUE 1: PBPK Model Validation Failing

### Current Status:
- **Problem:** PBPK models não estão validando corretamente
- **Context:** Tentando predizer parâmetros farmacocinéticos (Fu, Vd, Clearance)
- **Last attempt:** Multi-task learning com missing data extremo (82% Fu, 81% Vd)
- **Result:** R² negativo ou muito baixo (< 0.30)

### Why It's Failing:
1. **Missing Data Extremo:**
   - Fu: 82% missing (6,449 samples)
   - Vd: 81% missing (6,966 samples)
   - Clearance: 10% missing (32,291 samples) ← ÚNICA com dados suficientes!

2. **Multi-Task Learning Challenges:**
   - Tasks com poucos dados poluem o treinamento
   - Weighted loss ajuda mas não resolve
   - Network compartilhada prejudica tasks independentes

3. **Dataset Issues:**
   - 44,779 compostos MAS dados desbalanceados
   - Scaffold split correto (zero leakage) mas isso agrava missing data
   - TDC + ChEMBL + KEC original = heterogeneidade

### Honest Assessment:
- **Multi-task não funciona** com 80%+ missing
- **Single-task models** são a solução correta
- **Clearance only** deve ter R² > 0.50 (dados suficientes!)
- **Fu/Vd** precisam modelos separados ou mais dados

---

## 🎯 SOLUTION 1: PBPK Model Validation Strategy

### Phase 1: Single-Task Models (IMMEDIATE, 2-3h)

**1. Clearance-Only Model (HIGH PRIORITY!):**
```python
# Focus APENAS em Clearance (32,291 samples, 90% coverage!)
# Target: R² > 0.50 (realistic with this much data)

# Model:
- Input: ChemBERTa 768d + RDKit 25 features
- Architecture: Deep MLP [1024, 512, 256, 128]
- Output: Single task (Clearance)
- Loss: MSE with log1p transform
- Epochs: 200 (sem early stopping prematuro)
- Learning rate: 1e-4 (baixo para estabilidade)
```

**Expected Result:** R² 0.50-0.60 (publishable!)

**2. Fu-Only Model (MEDIUM PRIORITY):**
```python
# 6,449 samples (18% coverage)
# Target: R² > 0.30 (realistic dado missing data)

# Same architecture mas:
- More regularization (dropout 0.4)
- Data augmentation (SMILES enumeration)
- Ensemble (5 models averaging)
```

**Expected Result:** R² 0.30-0.40 (acceptable)

**3. Vd-Only Model (MEDIUM PRIORITY):**
```python
# 6,966 samples (19% coverage)
# Target: R² > 0.35

# Similar to Fu-only
```

**Expected Result:** R² 0.35-0.45 (acceptable)

### Phase 2: Ensemble & Refinement (2-3h)

**Clearance Ensemble:**
- 5x MLP (different seeds)
- 3x GNN (molecular graphs)
- Average predictions
- **Target: R² > 0.60** (Nature-level!)

**Hyperparameter Optimization (Optuna):**
- Only for Clearance (best data)
- 50-100 trials
- Focus: learning rate, architecture, dropout

### Phase 3: Multi-Task ONLY if Single-Task Works (1-2h)

```python
# ONLY try multi-task IF:
# - Clearance R² > 0.55
# - Fu R² > 0.30
# - Vd R² > 0.30

# Then try joint training with:
# - Pre-trained single-task models
# - Task-specific heads
# - Shared representations only if improves
```

---

## 🗑️ ISSUE 2: Repository "Dispensa de Bar de Quinta"

### Current Status:
- **Problem:** Estrutura caótica, arquivos espalhados, sem organização
- **Context:** 611 arquivos na raiz, múltiplos scripts soltos, sem packages
- **Severity:** HIGH - dificulta desenvolvimento, quebra reprodutibilidade

### Why It's a Problem:
1. **Arquivos na raiz:** 611 files (deveria ser ~20)
2. **Sem estrutura packages/:** Código misturado
3. **Docs espalhados:** 50+ markdown files na raiz
4. **Scripts soltos:** Sem organização clara
5. **Tests desorganizados:** Difícil rodar/manter

### Q1 Impact:
- ❌ Reviewers vão questionar reprodutibilidade
- ❌ Difícil para outros pesquisadores usarem
- ❌ Não segue best practices 2025
- ❌ Dificulta manutenção e debugging

---

## 🎯 SOLUTION 2: Repository Cleanup Strategy

### Phase 1: Archive Historical Files (30 min)

```bash
# Criar archive/ e mover documentos antigos
mkdir -p docs/archive/sessions
mkdir -p docs/archive/checkpoints
mkdir -p docs/archive/reports

# Mover:
- CHECKPOINT_*.md → docs/archive/checkpoints/
- SESSION_*.md → docs/archive/sessions/
- REPORT_*.md → docs/archive/reports/
- Tudo com data < 2025-11-01 → docs/archive/
```

**Result:** Raiz limpa de arquivos históricos

### Phase 2: Organize Current Docs (30 min)

```bash
# Estrutura final docs/
docs/
  ├── README.md (index master)
  ├── architecture/
  │   ├── ARCHITECTURE.md
  │   ├── PACKAGES.md
  │   └── VERSIONING.md
  ├── guides/
  │   ├── AGENT_GUIDE.md
  │   ├── INTEGRATION_GUIDE_DARWIN_3.0.md
  │   ├── CLUSTER_QUICKSTART.md
  │   └── WORKFLOW_GUIDE.md
  ├── reference/
  │   ├── START_HERE_DARWIN_3.0.md
  │   ├── README_DARWIN_3.0.md
  │   ├── Q1_SCIENTIFIC_HONESTY_PROTOCOL.md
  │   └── API_REFERENCE.md
  ├── archive/ (arquivos históricos)
  └── papers/ (drafts de papers Q1)
```

### Phase 3: Consolidate Code into Packages (1-2h)

```bash
# Estrutura final packages/
packages/
  ├── darwin_core/          # Base (KEC, Persistent Homology, etc)
  ├── darwin_preprocessing/ # Preprocessing
  ├── darwin_ml/            # Machine Learning
  ├── darwin_microct/       # MicroCT analysis
  ├── darwin_sem/           # SEM analysis
  ├── darwin_rag/           # Enhanced RAG
  ├── darwin_ethics/        # Ethics Layer
  ├── darwin_api/           # Backend API
  └── darwin_frontend/      # Streamlit UI

# Move arquivos soltos:
- kec_*.py → packages/darwin_core/
- preprocessing_*.py → packages/darwin_preprocessing/
- ml_*.py → packages/darwin_ml/
- etc.
```

### Phase 4: Clean Root Directory (30 min)

**Keep in root (max 20 files):**
```
├── README.md (master index)
├── pyproject.toml
├── setup.py
├── requirements.txt
├── .gitignore
├── .cursorrules
├── LICENSE
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── packages/ (dir)
├── tests/ (dir)
├── docs/ (dir)
├── scripts/ (dir)
├── data/ (dir)
├── .github/ (dir)
├── k8s/ (dir)
└── infra/ (dir)
```

**Delete or archive everything else**

### Phase 5: Update Imports & Tests (1-2h)

```python
# Update all imports:
# OLD: from kec_algorithms import ...
# NEW: from packages.darwin_core.kec_algorithms import ...

# Update tests:
# tests/
#   ├── darwin_core/
#   ├── darwin_preprocessing/
#   ├── darwin_ml/
#   └── integration/
```

### Phase 6: CI/CD & Quality Gates (1h)

```yaml
# .github/workflows/quality.yml
- Black formatting (enforce)
- Flake8 linting (enforce)
- MyPy type checking (enforce)
- Pytest (>80% coverage required)
- Documentation build
```

---

## 📊 Prioritization

### IMMEDIATE (This Week):
1. ✅ **PBPK Clearance-Only Model** (2-3h)
   - Highest chance of success
   - 32k samples = good data
   - R² > 0.50 = publishable
   
2. ✅ **Repository Cleanup Phase 1-2** (1h)
   - Archive historical files
   - Organize current docs
   - Quick wins, big visual impact

### HIGH PRIORITY (Next Week):
3. ⏳ **PBPK Fu/Vd Single-Task** (2-3h)
   - After Clearance works
   - Separate models
   
4. ⏳ **Repository Cleanup Phase 3-4** (2-3h)
   - Consolidate code into packages
   - Clean root directory

### MEDIUM PRIORITY (After Streamlit):
5. ⏳ **PBPK Ensemble & HPO** (2-3h)
   - Optimize Clearance model
   - R² > 0.60 target

6. ⏳ **Repository Cleanup Phase 5-6** (2-3h)
   - Update imports & tests
   - CI/CD setup

---

## 🎯 Success Metrics

### PBPK:
- ✅ Clearance R² > 0.50 (MINIMUM for publication)
- ✅ Fu R² > 0.30 (acceptable dado missing data)
- ✅ Vd R² > 0.35 (acceptable dado missing data)
- 🎯 Clearance R² > 0.60 (IDEAL for Nature)

### Repository:
- ✅ Root directory < 25 files
- ✅ All code in packages/
- ✅ All docs in docs/
- ✅ Tests > 80% coverage
- ✅ CI/CD passing
- ✅ Linting/formatting enforced

---

## 💡 Honest Assessment

**PBPK:**
- Multi-task foi estratégia ERRADA (honestidade!)
- Single-task é caminho correto
- Clearance tem MELHOR chance (32k samples)
- Fu/Vd são difíceis mas possíveis
- Timeline realista: 1 semana para R² > 0.50

**Repository:**
- Está bagunçado MAS é organizável
- Cleanup vai melhorar MUITO a usabilidade
- Essencial para Q1 papers (reprodutibilidade)
- Timeline realista: 1 semana para cleanup completo

---

## 📋 Action Plan (Next 3 Days)

**Day 1 (Tomorrow):**
- Morning: Streamlit frontend (4h)
- Afternoon: PBPK Clearance-Only (2h)
- Evening: Repo cleanup Phase 1 (1h)

**Day 2:**
- Morning: Streamlit continuation (4h)
- Afternoon: PBPK Fu/Vd single-task (2h)
- Evening: Repo cleanup Phase 2-3 (2h)

**Day 3:**
- Morning: Streamlit polish (2h)
- Afternoon: PBPK ensemble (2h)
- Evening: Repo cleanup Phase 4 (2h)

---

**Both issues are SOLVABLE with focused work!** 🎯

**PBPK:** Single-task strategy will work  
**Repository:** Cleanup will make it Q1-ready  

**Timeline:** 1 week for both ✅

---

**Dr. Sounio Agourakis**  
**Darwin Platform**  
**2025-11-02**

