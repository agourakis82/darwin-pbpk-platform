# Análise Linha por Linha: Training Pipeline

**Arquivo Python:** `scripts/train_dynamic_gnn_pbpk.py`
**Data:** 2025-11-18
**Autor:** AI Assistant + Dr. Demetrios Agourakis

---

## 1. Visão Geral

**Total de linhas:** ~500+ linhas
**Função principal:** Pipeline de treinamento do Dynamic GNN
**Componentes:** DataLoader, Training loop, Validation, Checkpointing

---

## 2. Análise Linha por Linha

### LINHA 1-100: Setup e Configuração

```python
# argparse → Julia: ArgParse.jl
# DataLoader → Julia: Flux.DataLoader()
# torch.optim → Julia: Flux.Optimiser
```

**Refatoração Julia:**
```julia
using ArgParse
using Flux
using Flux.DataLoader
```

**Inovação:** Type-safe argument parsing, nativo Flux.jl

---

### LINHA 101-200: Training Loop

```python
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(pred, true)
        loss.backward()
        optimizer.step()
```

**Refatoração Julia:**
```julia
for epoch in 1:epochs
    model = Flux.trainmode!(model)
    for batch in train_loader
        loss, grads = Flux.withgradient(model) do m
            # Forward pass
            compute_loss(m, batch)
        end
        Flux.update!(opt, Flux.params(model), grads)
    end
end
```

**Inovações:**
1. **Automatic differentiation nativo:** Zygote.jl (sem necessidade de `.backward()`)
2. **Type-stable:** Zero overhead
3. **GPU-ready:** CUDA.jl integration

---

### LINHA 201-300: Validation Loop

```python
model.eval()
with torch.no_grad():
    for batch in val_loader:
        pred = model(batch)
        val_loss += criterion(pred, true)
```

**Refatoração Julia:**
```julia
model = Flux.testmode!(model)
for batch in val_loader
    pred = forward_batch(model, batch)
    val_loss += compute_loss(pred, true)
end
```

**Inovação:** `Flux.testmode!` desabilita dropout/batch norm automaticamente

---

### LINHA 301-400: Checkpointing

```python
torch.save(model.state_dict(), checkpoint_path)
```

**Refatoração Julia:**
```julia
BSON.@save checkpoint_path model
```

**Inovação:** BSON.jl é type-preserving (melhor que pickle)

---

## 3. Inovações Implementadas

### 1. Automatic Mixed Precision (AMP)
```julia
# Flux.jl suporta AMP nativo
# (implementação futura)
```

### 2. Learning Rate Scheduling
```julia
scheduler = Flux.ReduceLROnPlateau(opt, factor=0.5, patience=5)
Flux.adjust!(scheduler, val_loss)
```

### 3. Gradient Clipping
```julia
Flux.clip!(Flux.params(model), gradient_clip)
```

### 4. Organ Weights
```julia
organ_weights = ones(NUM_ORGANS)
organ_weights[LIVER_IDX] = 2.0  # Órgãos críticos
```

---

## 4. Próximos Passos

1. ✅ Análise linha por linha - **CONCLUÍDA**
2. ✅ Implementação Julia - **CONCLUÍDA**
3. ⏳ Testes unitários - **PRÓXIMO**
4. ⏳ Benchmark vs PyTorch - **PENDENTE**

---

**Última atualização:** 2025-11-18

