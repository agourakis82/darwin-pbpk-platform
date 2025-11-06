# 🚀 Treinamento Dynamic GNN - Em Andamento

**Data de Início:** 06 de Novembro de 2025  
**Status:** ⏳ **RODANDO EM BACKGROUND**

---

## 📊 CONFIGURAÇÃO DO TREINAMENTO

### Dataset:
- **Arquivo:** `data/dynamic_gnn_training_full/training_data.npz`
- **Amostras:** 1000 (800 train, 200 val)
- **Tamanho:** 10.26 MB
- **Time points:** 100 por amostra
- **Órgãos:** 14 compartimentos

### Modelo:
- **Parâmetros:** 155,972
- **Arquitetura:** Dynamic GNN (14 órgãos, 3 GNN layers, GRU temporal)
- **Device:** CPU (GPU recomendado para mais velocidade)

### Hiperparâmetros:
- **Épocas:** 50
- **Batch size:** 8
- **Learning rate:** 1e-3
- **Optimizer:** Adam
- **Scheduler:** ReduceLROnPlateau (patience=5, factor=0.5)

### Output:
- **Diretório:** `models/dynamic_gnn_full/`
- **Arquivos:**
  - `best_model.pt` - Melhor modelo (menor val loss)
  - `final_model.pt` - Última época
  - `training_curve.png` - Curva de treinamento

---

## ⏱️ ESTIMATIVA DE TEMPO

### CPU (atual):
- **Tempo por iteração:** ~4-5 segundos
- **Iterações por época:** 100 (800 amostras / batch_size 8)
- **Tempo por época:** ~7-8 minutos
- **Tempo total (50 épocas):** ~6-7 horas ⏰

### GPU (recomendado):
- **Tempo por iteração:** ~0.1-0.2 segundos (20-50x mais rápido)
- **Tempo total (50 épocas):** ~15-30 minutos ⚡

---

## 📈 MONITORAMENTO

### Ver Progresso:
```bash
# Seguir log em tempo real
tail -f training.log

# Ver últimas linhas
tail -30 training.log

# Ver apenas épocas e losses
tail -f training.log | grep -E "(Epoch|Loss|✅)"
```

### Verificar Processo:
```bash
# Ver se está rodando
ps aux | grep train_dynamic_gnn_pbpk

# Ver uso de CPU/RAM
top -p $(pgrep -f train_dynamic_gnn_pbpk)
```

### Parar Treinamento:
```bash
# Encontrar PID
pgrep -f train_dynamic_gnn_pbpk

# Parar (salva modelo atual)
kill <PID>
```

### Script de Monitoramento:
```bash
./scripts/monitor_training.sh
```

---

## 🎯 RESULTADOS ESPERADOS

### Baseado no Paper (arXiv 2024):
- **R²:** 0.9342 (target)
- **RMSE:** 0.0159
- **MAE:** 0.0116

### Progresso Esperado:
- **Épocas 1-10:** Loss alto, aprendendo padrões básicos
- **Épocas 10-30:** Convergência, loss diminuindo
- **Épocas 30-50:** Refinamento, otimização final

---

## 📊 MÉTRICAS A MONITORAR

### Durante Treinamento:
- **Train Loss:** Deve diminuir consistentemente
- **Val Loss:** Deve diminuir (sem overfitting)
- **Gap Train-Val:** Deve ser pequeno (< 20%)

### Após Treinamento:
- **R² vs ODE Solver:** > 0.90 (target)
- **RMSE:** < 0.02
- **MAE:** < 0.015
- **Per-organ accuracy:** Especialmente blood, liver, kidney

---

## 🔍 VALIDAÇÃO PÓS-TREINAMENTO

Após o treinamento completar, validar:

1. **Comparação com ODE Solver:**
   ```python
   # Carregar modelo treinado
   # Comparar predições vs ODE solver
   # Calcular R², RMSE, MAE
   ```

2. **Visualização:**
   - Curvas de concentração vs tempo
   - Comparação por órgão
   - Residual plots

3. **Métricas por Órgão:**
   - Blood (crítico)
   - Liver (metabolismo)
   - Kidney (excreção)
   - Brain (BBB)

---

## ⚠️ NOTAS IMPORTANTES

1. **Performance:** Treinamento em CPU é muito lento. Se disponível, usar GPU acelera 20-50x.

2. **Interrupção:** Se precisar parar, o modelo atual será salvo. Pode continuar depois.

3. **Convergência:** Se val loss parar de melhorar por 10+ épocas, considerar early stopping.

4. **Recursos:** Monitorar uso de RAM (dataset carregado na memória).

---

## 📁 ARQUIVOS GERADOS

Durante o treinamento:
- `training.log` - Log completo
- `models/dynamic_gnn_full/best_model.pt` - Melhor modelo (atualizado a cada melhoria)
- `models/dynamic_gnn_full/final_model.pt` - Modelo final (ao completar)
- `models/dynamic_gnn_full/training_curve.png` - Gráfico de losses

---

## 🎉 PRÓXIMOS PASSOS APÓS TREINAMENTO

1. ✅ Validar modelo vs ODE solver
2. ✅ Calcular métricas (R², RMSE, MAE)
3. ✅ Visualizar curvas de concentração
4. ✅ Comparar com paper SOTA
5. ✅ Documentar resultados
6. ✅ Integrar no pipeline PBPK

---

**"Rigorous science. Honest results. Real impact."**

**Status:** ⏳ Treinamento em andamento (PID: verificar com `ps aux | grep train`)

