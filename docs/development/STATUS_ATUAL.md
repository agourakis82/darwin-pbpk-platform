# 📊 Status Atual - Darwin PBPK Platform

**Data:** 06 de Novembro de 2025
**Última atualização:** 16:30

---

## 🗓️ Atualização Operacional — 15 de Novembro de 2025 09:15 -03

- 🚀 **Sweep B** (hidden_dim=128, 4 camadas, batch 24, 120 passos, `dt=0,1`, `lr=5e-4`) em execução: Epoch 56/200 com `Train/Val ≈ 1.0 × 10⁻⁶`, log centralizado em `models/dynamic_gnn_sweep_b/training.log`.
- 📓 **Notebook `pbpk_enriched_analysis.ipynb`** atualizado para incluir seção “Sweep B” com parsing automático do log parcial e curvas de perda em tempo real.
- 🧠 **Preparação do Sweep C** concluída: diretório `models/dynamic_gnn_sweep_c/` criado e configuração proposta (`hidden_dim=160`, `num_gnn_layers=4`, `batch=28`, `lr=3e-4`, 120 passos, `dt=0,1`) aguardando disponibilidade da GPU para disparo (`CUDA_VISIBLE_DEVICES=0 python scripts/train_dynamic_gnn_pbpk.py ...`).
- 📄 `docs/DYNAMIC_GNN_IMPLEMENTATION.md` expandido com a seção “Hyperparameter Sweeps (Nov/2025)” descrevendo o status das execuções (Sweep A concluído, Sweep B em curso, Sweep C planejado).
- 🔄 Próximas ações imediatas: acompanhar convergência de Sweep B até ~Epoch 100, gerar simulação de validação com `best_model.pt` assim que disponível e disparar Sweep C usando o shell script preparado.

---

## 🗓️ Atualização Operacional — 14 de Novembro de 2025 06:25 -03

- ✅ Treinamento batched (batch 24, lr=5e-4, 200 épocas) concluído com `Val Loss=5.2e-5`; artefatos gerados em `models/dynamic_gnn_enriched_v3/` (`best_model.pt`, `final_model.pt`, `training_curve.png`, `training.log`).
- 📈 Curva e métricas documentadas em `training_curve.png`; log detalhado disponível via `models/dynamic_gnn_enriched_v3/training.log`.
- 🧪 CLI `apps.pbpk_core.simulation.dynamic_gnn_pbpk` validado com o novo checkpoint em GPU/CPU (`logs/dynamic_gnn_enriched_v3_cuda_sim.md` e `logs/dynamic_gnn_enriched_v3_cpu_sim.md`), exibindo cinética multiórgãos (picos ~1.55 mg/L em tecidos periféricos, `Final blood=0.3166 mg/L`).
- 🧷 Checkpoint padrão do simulador atualizado: `DEFAULT_DYNAMIC_GNN_CHECKPOINT` aponta para `models/dynamic_gnn_enriched_v3/best_model.pt` (configurável via CLI `--checkpoint`).
- 🧷 Regressão numérica (`tests/test_dynamic_gnn_regression.py`) executada após refatoração batched – estabilidade confirmada.
- 🗂️ Pendências: incorporar métricas no notebook `pbpk_enriched_analysis.ipynb`, atualizar documentação (`docs/DYNAMIC_GNN_IMPLEMENTATION.md`) com o fluxo batched e promover o novo checkpoint como padrão no CLI.
- 📘 Plano de sweeps documentado em `docs/DYNAMIC_GNN_SWEEP_PLAN.md` (combos de hidden_dim, layers, batch e temporalidade).

---

## 🗓️ Atualização Operacional — 13 de Novembro de 2025 11:30 -03

- ✅ `scripts/analysis/build_dynamic_gnn_dataset_from_enriched.py` executado sem limite de amostras → `data/processed/pbpk_enriched/dynamic_gnn_dataset_enriched_v3.npz` (6 551 amostras, 100 passos temporais) consolidado para treino.
- ✅ `models/dynamic_gnn_enriched_v3/best_model.pt` atualizado incrementalmente durante retomada do fine-tuning.
- 🔄 Treinamento `DynamicPBPKGNN` atualizado para forward batched (`CUDA_VISIBLE_DEVICES=0`, `batch_size=24`, `epochs=200`, `lr=5e-4`) com logging em `models/dynamic_gnn_enriched_v3/training.log`, mantendo ~10 GB de VRAM e acelerando as épocas.
- 🧪 Suite de regressão numérica (`tests/test_dynamic_gnn_regression.py`) pronta para validar consistência pós-treino.
- 📈 Notebook `notebooks/pbpk_enriched_analysis.ipynb` preparado para incorporar métricas pós-treino (pendente de atualização após convergência).
- 🗂️ Próximos passos paralelos: (i) atualizar gráficos de clearance vs. parâmetros no notebook, (ii) integrar pesos finais ao CLI de inferência (`apps/pbpk_core/simulation/dynamic_gnn_pbpk.py`) e (iii) documentar a estratégia de throttling de GPU em `docs/DYNAMIC_GNN_IMPLEMENTATION.md`.

---

## 🗓️ Atualização Operacional — 11 de Novembro de 2025 12:25 -03

- ✅ `pytest` (6 testes) executado sem falhas — validação do módulo `DynamicPBPKGNN`.
- ✅ Simulação rápida `DynamicPBPKSimulator` (`dose=100 mg`, `dt=0.5 h`, 24 passos) registrada em `logs/dynamic_gnn_simulation_20251111_122506.md`.
- ✅ Nova simulação com pesos treinados (`models/dynamic_gnn_full/best_model.pt`) gerou curvas multiórgãos plausíveis — ver `logs/dynamic_gnn_simulation_full_20251111_154011.md`.
- ⚠️ Warnings conhecidos durante execução:
  - `torch-scatter` e `torch-sparse` recompilados para `torch==2.8.0+cu128`; validar em GPU nos próximos treinos.
  - Depreciação `TRANSFORMERS_CACHE`; alinhar para `HF_HOME` nas próximas releases.
- 📌 Resultados-chave: `Cmax(blood)=20.0 mg/L`, dispersão multiórgãos com concentrações finais ~0.43 mg/L em compartimentos periféricos.
- 🔁 Próximo passo recomendado: carregar pesos treinados ou concluir fine-tuning para gerar perfis multiórgãos realistas.
- 🧭 Próximas ações (executadas nesta sessão): carregamento do checkpoint `dynamic_gnn_full`, geração de log multiórgãos, normalização do ambiente HuggingFace (`HF_HOME`) e documentação de dependências CUDA para `torch-scatter/torch-sparse`.
- 🧪 Regressão adicional disponível: `tests/test_dynamic_gnn_regression.py` compara resultados do checkpoint com os valores logados; CLI `python -m apps.pbpk_core.simulation.dynamic_gnn_pbpk --help` expõe parâmetros reproduzíveis.
- 📈 Script `scripts/analysis/analyze_literature_clearance.py` explora o dataset real (`clearance_hepatocyte_az`) e gera resumos em `analysis/literature_clearance_stats.json` e `analysis/literature_simulation_summary.csv`.
- 📊 Relatório consolidado em `analysis/literature_clearance_report.md` resume as variações de fu e os compostos extremos simulados.
- 🧮 Script `scripts/analysis/build_pbpk_parameter_table.py` consolida TDC + ChEMBL em `analysis/pbpk_parameters_(long|wide).csv` (6.5k compostos com SMILES).
- 🧷 `scripts/analysis/generate_chemberta_embeddings.py` gerou embeddings ChemBERTa (`analysis/pbpk_chemberta_embeddings.npz`, 1.8k SMILES únicos).
- 🧬 `scripts/analysis/generate_chemberta_embeddings.py --input analysis/pbpk_parameters_wide_enriched.csv` gerou 4.5k embeddings ChemBERTa (`analysis/pbpk_chemberta_embeddings_enriched.npz`).
- 🌐 `scripts/analysis/enrich_pbpk_dataset_pubchem.py` adicionou 1.5k SMILES via PubChem; `analysis/pbpk_parameters_wide_enriched_v2.csv` cobre 5.9k moléculas (504 ainda sem estrutura).
- 🧠 Embeddings atualizados: `analysis/pbpk_chemberta_embeddings_enriched_v2.npz` (5.9k SMILES).
- ✅ Cobertura total de SMILES via `build_pbpk_parameter_table.py` + merges sucessivos (`analysis/pbpk_parameters_wide_enriched_v3.csv`).
- 🧠 Embeddings finais: `analysis/pbpk_chemberta_embeddings_enriched_v3.npz` (6.4k SMILES) + dataset MLP `data/processed/pbpk_enriched/pbpk_enriched_v3.npz`.
- 📦 Dataset sintético para GNN: `scripts/analysis/build_dynamic_gnn_dataset_from_enriched.py --max-samples` gera `data/processed/pbpk_enriched/dynamic_gnn_dataset_enriched_v3.npz`.
- 📓 Notebook `notebooks/pbpk_enriched_analysis.ipynb` documenta correlações (Clearance vs fu/Vd).

---

## ✅ CONCLUÍDO

### 1. Repositório GitHub
- ✅ Código migrado: 55 arquivos, 14,826 linhas
- ✅ Tag v1.0.0 criada
- ✅ Push para GitHub concluído
- ✅ DOI Software obtido: **10.5281/zenodo.17536674**

### 2. GitHub Release
- ✅ Release v1.0.0 criado e publicado
- ✅ URL: https://github.com/agourakis82/darwin-pbpk-platform/releases/tag/v1.0.0
- ⚠️ Descrição pode ser atualizada manualmente se necessário (já tem DOI correto)

### 3. Preparação para Zenodo
- ✅ Scripts criados:
  - `scripts/prepare_zenodo_upload.sh` - Prepara arquivos para upload
  - `scripts/update_github_release.py` - Atualiza release (se necessário)
- ✅ Documentação criada:
  - `docs/DATASETS_README.md` - README para os datasets
  - `docs/ZENODO_UPLOAD_GUIDE.md` - Guia completo de upload
- ✅ Arquivos preparados em: `/tmp/darwin-pbpk-datasets-v1.0.0/`
  - `consolidated_pbpk_v1.parquet` (1.5 MB)
  - `chemberta_embeddings_consolidated.npz` (123 MB)
  - `molecular_graphs.pkl` (286 MB)
  - `README.md`
- ✅ ZIP criado: `darwin-pbpk-datasets-v1.0.0.zip` (136 MB)

---

## 🎉 UPLOAD CONCLUÍDO COM SUCESSO!

### 1. Upload Datasets no Zenodo ✅ **CONCLUÍDO!**

**Status:** ✅ Upload completo e publicado!

**DOI dos Datasets:** `10.5281/zenodo.17541874`
**URL:** https://doi.org/10.5281/zenodo.17541874

**Arquivos enviados:**
- ✅ consolidated_pbpk_v1.parquet (1.4 MB)
- ✅ chemberta_embeddings_consolidated.npz (122.2 MB)
- ✅ molecular_graphs.pkl (285.7 MB)
- ✅ README.md (2.9 KB)

**Total:** 409.3 MB

**Data do Upload:** 06 de Novembro de 2025

---

### 2. Atualizar README com DOI Datasets ✅ **CONCLUÍDO!**

**Status:** ✅ README.md e RELEASE_DESCRIPTION.md atualizados com DOI

**DOI configurado:**
- README.md: `10.5281/zenodo.17541874`
- RELEASE_DESCRIPTION.md: `10.5281/zenodo.17541874`

---

## 🚀 BREAKTHROUGH: Dynamic GNN para PBPK ✅ **IMPLEMENTADO!**

**Data:** 06 de Novembro de 2025
**Status:** ✅ Arquitetura completa implementada e testada

### Implementação:
- ✅ **DynamicPBPKGNN**: Modelo completo (586 LOC)
- ✅ **14 compartimentos PBPK**: Graph com órgãos como nodes
- ✅ **Message Passing**: Custom layer para interações entre órgãos
- ✅ **Evolução Temporal**: GNN layers + GRU
- ✅ **Attention**: Órgãos críticos (liver, kidney, brain)
- ✅ **Simulator Wrapper**: Interface similar ao ODE solver
- ✅ **Testes Unitários**: 6 testes passando (177 LOC)

### Arquivos Criados:
- `apps/pbpk_core/simulation/dynamic_gnn_pbpk.py` (586 LOC)
- `apps/pbpk_core/simulation/__init__.py` (exports)
- `tests/test_dynamic_gnn_pbpk.py` (177 LOC)
- `docs/DYNAMIC_GNN_IMPLEMENTATION.md` (215 LOC)

### Baseado em:
- **arXiv 2024**: Dynamic GNN for PBPK (R² 0.9342)
- Supera ODE tradicional (R² 0.85-0.90)

### Competitive Advantage:
- **Único software open-source** com Dynamic GNN para PBPK!
- Simcyp: ❌ Não tem
- GastroPlus: ❌ Não tem
- PK-Sim: ❌ Não tem
- **Darwin: ✅ IMPLEMENTADO!**

### Status do Treinamento:
- ✅ **Pipeline completo implementado**
- ✅ **Bug de shapes corrigido** (time_points batch)
- ✅ **Treinamento funcionando** (shapes [14, 100] corretos)
- ✅ **Teste rápido concluído** (2 épocas, Val Loss: 36.43)
- ✅ **Treinamento completo em andamento** (50 épocas, 1000 amostras)

**Resultados do teste:**
- Train Loss: 13.05 → 10.90 (melhoria)
- Val Loss: 50.25 → 36.43 (27% redução)
- Modelo gerando 100 pontos temporais corretamente ✅

**Treinamentos em andamento:**
1. **Node DemetriosPCS (RTX 4000 Ada):**
   - Status: ✅ Rodando (Época 2, Val Loss: 9.82)
   - Batch size: 16
   - Tempo estimado: ~12-13 horas
   - Output: `models/dynamic_gnn_full/`

2. **Node Maria (L4 24GB) - K8s Job:**
   - Status: ✅ Job K8s rodando
   - Batch size: 32 (otimizado para L4)
   - Tempo estimado: ~6-7 horas
   - Output: `models/dynamic_gnn_maria/`
   - Job: `dynamic-gnn-training-maria`

**Monitoramento:**
- Node atual: `tail -f training.log`
- Node maria: `kubectl logs <pod-name>` ou `./scripts/monitor_k8s_training.sh`

---

## 📋 CHECKLIST FINAL

### Já Feito ✅
- [x] Repositório criado no GitHub
- [x] Código migrado (55 arquivos)
- [x] Tag v1.0.0 criada
- [x] GitHub Release publicado
- [x] DOI Software obtido
- [x] Scripts de preparação criados
- [x] Documentação criada
- [x] Arquivos preparados para upload

### Para Fazer ⏳
- [ ] Integrar pesos atualizados (`models/dynamic_gnn_enriched_v3`) ao CLI de inferência em `apps/pbpk_core/simulation/dynamic_gnn_pbpk.py`.
- [ ] Atualizar `notebooks/pbpk_enriched_analysis.ipynb` com métricas pós-treino (curvas de perda, distribuição de erro por órgão).
- [ ] Documentar a estratégia de redução de footprint de GPU em `docs/DYNAMIC_GNN_IMPLEMENTATION.md` e `STATUS_ATUAL.md`.
- [ ] Planejar sweep adicional de hiperparâmetros (lr × batch) após convergência do treino atual.
- [ ] Consolidar log de treinamento (`models/dynamic_gnn_enriched_v3/training_curve.png` + métricas) em `STATUS_ATUAL.md` e `PROXIMOS_PASSOS.md`.

---

## 📊 ESTATÍSTICAS

**Código:**
- 55 arquivos
- 14,826 linhas totais
- 7,601 linhas Python

**Datasets:**
- 44,779 compostos
- ~410 MB total (uncompressed)
- 3 arquivos principais

**DOIs:**
- Software: ✅ `10.5281/zenodo.17536674`
- Datasets: ⏳ Aguardando upload

---

## 🎯 PRÓXIMOS PASSOS IMEDIATOS

1. **AGORA:** Fazer upload dos datasets no Zenodo
   - Seguir `docs/ZENODO_UPLOAD_GUIDE.md`
   - Tempo: 20-30 min

2. **DEPOIS:** Atualizar READMEs com DOI
   - Tempo: 5 min

3. **FINAL:** Commit e push
   - Tempo: 2 min

**Total restante:** ~35 minutos

---

## 📚 RECURSOS CRIADOS

### Scripts
- `scripts/prepare_zenodo_upload.sh` - Prepara arquivos
- `scripts/upload_to_zenodo.py` - **Upload automático via API do Zenodo** ⭐
- `scripts/update_readme_with_doi.py` - Atualiza READMEs com DOI
- `scripts/update_github_release.py` - Atualiza release

### Documentação
- `docs/DATASETS_README.md` - README para datasets
- `docs/ZENODO_UPLOAD_GUIDE.md` - Guia completo de upload
- `PROXIMOS_PASSOS.md` - Lista de tarefas original
- `STATUS_ATUAL.md` - Este arquivo

### Arquivos Preparados
- `/tmp/darwin-pbpk-datasets-v1.0.0/` - Arquivos prontos para upload
- `darwin-pbpk-datasets-v1.0.0.zip` - ZIP opcional

---

## 🎊 RESULTADO ESPERADO

Quando completar o upload no Zenodo:

✅ **DOI Software:** `10.5281/zenodo.17536674`
✅ **DOI Datasets:** `10.5281/zenodo.XXXXXX`
✅ **README atualizado** com ambos DOIs
✅ **Paper-ready** para Nature Machine Intelligence

---

**"Rigorous science. Honest results. Real impact."**

