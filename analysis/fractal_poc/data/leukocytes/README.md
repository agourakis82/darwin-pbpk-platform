# White Blood Cell (WBC/Leukocyte) Image Datasets

Este diretório contém datasets de imagens de leucócitos organizados para análise fractal e modelagem PBPK.

## Estrutura de Diretórios

```
leukocytes/
├── normal/              # Células normais
│   ├── all/            # Todas as subpopulações misturadas
│   ├── neutrophils/    # Neutrófilos
│   ├── lymphocytes/    # Linfócitos (T, B, NK)
│   ├── monocytes/      # Monócitos
│   ├── eosinophils/    # Eosinófilos
│   └── basophils/      # Basófilos
│
├── leukemia/           # Leucemia (linfóide e mieloide)
│   ├── lymphocytes/    # Linfócitos anormais (ALL)
│   └── myeloid/        # Células mieloides anormais (AML)
│
├── sepsis/             # Sepse (neutrófilos alterados morfologicamente)
│   └── neutrophils/
│
└── leukopenia/         # Leucopenia (redução geral)
    └── all/
```

## Datasets Disponíveis

### 1. BCCD - Normal Blood Cells
- **Fonte**: GitHub (Shenggan/BCCD_Dataset)
- **Descrição**: Células sanguíneas normais incluindo leucócitos
- **Status**: ✅ Organizado automaticamente
- **Localização**: `normal/all/`

### 2. Leukemia ALL (Acute Lymphoblastic Leukemia)
- **Fonte**: Kaggle (mehradaria/leukemia)
- **Descrição**: 3,256 imagens de 89 pacientes com ALL
- **Classes**: Benign, Early Pre-B ALL, Pre-B ALL, Pro-B ALL
- **Download Manual**: https://www.kaggle.com/datasets/mehradaria/leukemia
- **Status**: ⚠️ Requer download manual ou Kaggle CLI

### 3. WBC Classification Dataset
- **Fonte**: Kaggle (paultimothymooney/blood-cells)
- **Descrição**: Leucócitos classificados por tipo
- **Subpopulações**: Eosinophil, Lymphocyte, Monocyte, Neutrophil
- **Download Manual**: https://www.kaggle.com/datasets/paultimothymooney/blood-cells
- **Status**: ⚠️ Requer download manual ou Kaggle CLI

### 4. Blood Cell Cancer ALL (4-class)
- **Fonte**: Kaggle (mohammadamireshraghi/blood-cell-cancer-all-4class)
- **Descrição**: 4 classes de leucemia linfoblástica aguda
- **Download Manual**: https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class
- **Status**: ⚠️ Requer download manual ou Kaggle CLI

## Como Fazer Download

### Opção 1: Script Automático (Requer Kaggle CLI)

1. Instalar Kaggle CLI:
```bash
pip install kaggle
```

2. Configurar API token:
   - Acesse: https://www.kaggle.com/settings
   - Baixe `kaggle.json`
   - Coloque em `~/.kaggle/kaggle.json`
   - Execute: `chmod 600 ~/.kaggle/kaggle.json`

3. Executar script:
```bash
cd analysis/fractal_poc
python download_wbc_datasets.py
```

### Opção 2: Download Manual

1. **Leukemia ALL**:
   - Acesse: https://www.kaggle.com/datasets/mehradaria/leukemia
   - Faça login no Kaggle
   - Baixe o dataset
   - Extraia em: `data/leukemia_ALL_raw/`

2. **WBC Classification**:
   - Acesse: https://www.kaggle.com/datasets/paultimothymooney/blood-cells
   - Faça login no Kaggle
   - Baixe o dataset
   - Extraia em: `data/wbc_classification_raw/`

3. **Organizar após download manual**:
```bash
python download_wbc_datasets.py
```

O script detectará automaticamente os datasets extraídos e os organizará na estrutura correta.

## Verificar Datasets Disponíveis

```bash
python download_wbc_datasets.py --check
```

## Uso na Análise Fractal

Os datasets organizados aqui são usados por:
- `julia-migration/src/DarwinPBPK/image_analysis/leukocyte_fractal_analysis.jl`
- Scripts de análise fractal em Julia
- Modelagem PBPK com parâmetros ajustados por morfologia fractal

## Notas

- Os scripts organizam automaticamente até 100 imagens por subpopulação para testes iniciais
- Para análise completa, aumente o limite nos scripts ou remova a limitação
- Imagens são copiadas (não movidas) para preservar datasets originais
- Formatos suportados: `.jpg`, `.png`, `.jpeg`, `.bmp`

## Referências

1. BCCD Dataset: https://github.com/Shenggan/BCCD_Dataset
2. Leukemia ALL Dataset: https://www.kaggle.com/datasets/mehradaria/leukemia
3. WBC Classification: https://www.kaggle.com/datasets/paultimothymooney/blood-cells

---

**Última atualização**: 2025-12-01  
**Autor**: Darwin PBPK Platform

