# Download de Imagens de Leucócitos (WBC) - Implementação

**Data**: 2025-12-01  
**Objetivo**: Organizar download e estruturação de datasets de imagens de leucócitos para análise fractal, similar ao processo feito para RBC.

## ✅ Implementação Completa

### 1. Script de Download e Organização

**Arquivo**: `download_wbc_datasets.py`

**Funcionalidades**:
- ✅ Download automático via Kaggle CLI (quando configurado)
- ✅ Organização automática de datasets baixados manualmente
- ✅ Estruturação por condição (normal, leucemia, sepse, leucopenia)
- ✅ Estruturação por subpopulação (neutrófilos, linfócitos, monócitos, etc.)
- ✅ Verificação de datasets disponíveis
- ✅ Compatível com múltiplos formatos de dataset

**Uso**:
```bash
# Organizar datasets existentes e tentar download automático
python download_wbc_datasets.py

# Apenas verificar datasets disponíveis
python download_wbc_datasets.py --check

# Forçar re-download
python download_wbc_datasets.py --force
```

### 2. Estrutura de Diretórios

```
data/leukocytes/
├── normal/
│   ├── all/           ✅ 100 imagens (BCCD)
│   ├── neutrophils/
│   ├── lymphocytes/
│   ├── monocytes/
│   ├── eosinophils/
│   └── basophils/
├── leukemia/
│   └── lymphocytes/
├── sepsis/
│   └── neutrophils/
└── leukopenia/
    └── all/
```

### 3. Datasets Suportados

| Dataset | Fonte | Status | Imagens |
|---------|-------|--------|---------|
| BCCD Normal | GitHub | ✅ Organizado | 100 |
| Leukemia ALL | Kaggle | ⚠️ Requer download | - |
| WBC Classification | Kaggle | ⚠️ Requer download | - |
| Blood Cell Cancer ALL | Kaggle | ⚠️ Requer download | - |

### 4. Organização Automática

O script detecta automaticamente:
- Estruturas comuns de datasets Kaggle
- Diretórios "Benign" e "Malignant"
- Diretórios por subpopulação (EOSINOPHIL, LYMPHOCYTE, etc.)
- Imagens em formato jpg, png, jpeg, bmp

### 5. Documentação

- ✅ README em `data/leukocytes/README.md` com instruções completas
- ✅ Instruções para download manual
- ✅ Instruções para configuração do Kaggle CLI
- ✅ Estrutura de diretórios documentada

## 🎯 Próximos Passos

1. **Download Manual dos Datasets Kaggle** (quando necessário):
   - Configurar Kaggle CLI com API token
   - Ou fazer download manual e extrair em diretórios apropriados

2. **Aumentar Número de Imagens**:
   - Remover limite de 100 imagens por subpopulação nos scripts
   - Organizar todos os datasets disponíveis

3. **Integração com Análise Fractal**:
   - Usar datasets organizados em `leukocyte_fractal_analysis.jl`
   - Testar pipeline completo de análise

4. **Adicionar Datasets de Sepse**:
   - Buscar datasets específicos de neutrófilos em sepse
   - Organizar por grau de alteração morfológica

## 📊 Status Atual

```
📁 NORMAL:
  ✅ all: 100 images

📊 Total images: 100
```

## 🔗 Links Úteis

- Kaggle API: https://www.kaggle.com/settings
- Leukemia ALL Dataset: https://www.kaggle.com/datasets/mehradaria/leukemia
- WBC Classification: https://www.kaggle.com/datasets/paultimothymooney/blood-cells
- BCCD Dataset: https://github.com/Shenggan/BCCD_Dataset

---

**Nota**: Este processo espelha o trabalho feito para RBC, garantindo consistência metodológica entre análises de células vermelhas e brancas.

