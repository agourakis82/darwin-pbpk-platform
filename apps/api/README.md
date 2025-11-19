# 🚀 Darwin PBPK Platform - API REST

API REST completa para predições e simulações PBPK usando FastAPI.

## 📦 Instalação

```bash
pip install -r requirements.txt
```

## 🏃 Executar

```bash
# Desenvolvimento (com reload automático)
python scripts/run_api.py --reload

# Produção
python scripts/run_api.py --host 0.0.0.0 --port 8000 --workers 4
```

## 📖 Documentação

- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc
- **Health Check**: http://localhost:8000/health

## 🔌 Endpoints Principais

### Predição PBPK

```bash
POST /api/v1/predict/pbpk
```

### Simulação Dynamic GNN

```bash
POST /api/v1/simulate/dynamic-gnn
```

### Listar Modelos

```bash
GET /api/v1/models
```

Veja `docs/API_DOCUMENTATION.md` para documentação completa.

## 🧪 Exemplo de Uso

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

## 📚 Estrutura

```
apps/api/
├── main.py              # Aplicação FastAPI
├── models.py           # Modelos Pydantic
├── dependencies.py     # Dependencies
└── routers/
    ├── pbpk.py         # Predições PBPK
    ├── simulation.py   # Simulações
    └── models.py       # Gerenciamento de modelos
```

## 🔧 Desenvolvimento

Veja `docs/API_DOCUMENTATION.md` para guia completo de desenvolvimento.

## 📝 Licença

MIT License - Veja LICENSE

---

**"Rigorous science. Honest results. Real impact."**

