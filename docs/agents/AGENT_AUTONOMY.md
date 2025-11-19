# Darwin PBPK Platform — Guia de Autonomia dos Agentes
# Criado: 2025-11-07 23:54:00 -03
# Autor: Dr. Demetrios Agourakis

## Objetivo
- Atuar com liberdade total na plataforma PBPK, preservando padrão científico de classe mundial.
- Integrar modelos multi-modais (ChemBERTa, GNNs, RDKit) com RAG++ e serviços do `darwin-core`.

## Princípios
1. **Autonomia**
   Rodar pipelines (`apps/training/*.py`, `scripts/`) sem confirmações adicionais. Automação primeiro.

2. **Rigor Científico**
   Garantir datasets com DOI, versões claras de modelos e métricas reproduzíveis (R², RMSE, etc.).

3. **Integração Darwin**
   Manter compatibilidade com GraphRAG, Self-RAG e APIs compartilhadas. Documentar novas integrações.

4. **Memória & Insights**
   Registrar insights relevantes via `update_memory` ou Notion (ex.: benchmarks, correlações).

5. **Observabilidade**
   Priorizar scripts que gerem relatórios, logs e notebooks reprodutíveis para cada experimento.

## Fluxo Recomendado
1. Preparar ambiente (`pip install -r requirements.txt`).
2. Executar treinamentos (`python apps/training/02_gnn_model.py`).
3. Validar resultados (`scripts/regression_analysis_real.py`, notebooks).
4. Atualizar changelog/versão quando publicar modelos/datasets.
5. Sincronizar Notion/Zenodo usando automações existentes.

## Regras Simplificadas
- Livre para executar qualquer comando; sem necessidade de timestamps automáticos.
- Apenas mantenha versionamento e documentação crítica para reproducibilidade.
- Tokens e segredos permanecem sob controle do Dr. Demetrios.

