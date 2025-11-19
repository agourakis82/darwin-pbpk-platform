# Darwin PBPK Platform - API REST

**Criado:** 2025-11-08
**Status:** ✅ Implementado

## 📊 Resumo

API REST completa para o Darwin PBPK Platform usando FastAPI.

### Arquivos Criados

- `apps/api/main.py` - Aplicação FastAPI principal (150 LOC)
- `apps/api/models.py` - Modelos Pydantic para validação (120 LOC)
- `apps/api/routers/pbpk.py` - Endpoints de predição PBPK (150 LOC)
- `apps/api/routers/simulation.py` - Endpoints de simulação (120 LOC)
- `apps/api/routers/models.py` - Endpoints de modelos (100 LOC)
- `apps/api/dependencies.py` - Dependencies (20 LOC)
- `scripts/run_api.py` - Script para executar API (60 LOC)

**Total:** ~720 linhas de código

### Endpoints Implementados

1. ✅ `POST /api/v1/predict/pbpk` - Predição PBPK completa
2. ✅ `POST /api/v1/predict/parameters` - Predição de parâmetros PK
3. ✅ `POST /api/v1/simulate/dynamic-gnn` - Simulação Dynamic GNN
4. ✅ `POST /api/v1/simulate/ode` - Simulação ODE (placeholder)
5. ✅ `GET /api/v1/models` - Lista modelos disponíveis
6. ✅ `GET /api/v1/models/{name}` - Informações de modelo específico
7. ✅ `GET /health` - Health check
8. ✅ `GET /` - Root endpoint

### Features

- ✅ Documentação automática (Swagger/ReDoc)
- ✅ Validação de dados com Pydantic
- ✅ Suporte a CUDA/CPU automático
- ✅ Error handling global
- ✅ CORS configurado
- ✅ Logging estruturado

### Próximos Passos

- [ ] Integrar predição de parâmetros PK do SMILES
- [ ] Carregar modelos treinados automaticamente
- [ ] Cache de modelos em memória
- [ ] Autenticação (JWT)
- [ ] Rate limiting
- [ ] Métricas Prometheus
- [ ] Testes unitários

### Como Usar

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar API
python scripts/run_api.py --reload

# Acessar documentação
# http://localhost:8000/api/v1/docs
```

### Exemplo de Uso

```python
import requests

# Predição PBPK
response = requests.post(
    "http://localhost:8000/api/v1/predict/pbpk",
    json={
        "smiles": "CCO",
        "dose": 100.0,
        "route": "iv",
        "model_type": "dynamic_gnn"
    }
)

result = response.json()
print(f"Cmax blood: {result['summary']['blood_cmax']}")
```

---

**"Rigorous science. Honest results. Real impact."**

