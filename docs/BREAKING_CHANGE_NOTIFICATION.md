# 🚨 NOTIFICAÇÃO: Breaking Change - Migração para Julia

**Data:** 2025-11-18
**Versão:** v2.0.0-julia
**Tipo:** 🚨 **BREAKING CHANGE**

---

## 📢 Anúncio Importante

O **Darwin PBPK Platform** foi completamente migrado para **Julia**. Esta é uma mudança **breaking** que requer ação dos usuários.

---

## 🚨 O Que Mudou?

### ❌ Removido (Python)
- **96 arquivos Python** removidos
- **Dependências Python** (PyTorch, NumPy, SciPy, etc.)
- `requirements.txt`
- Código Python obsoleto

### ✅ Novo (Julia)
- **100% código Julia**
- **Dependências Julia** (DifferentialEquations.jl, Flux.jl, etc.)
- `Project.toml` (Julia)
- **4× melhor performance**

---

## ⚠️ Ação Necessária

### Para Usuários Existentes:

1. **Instalar Julia 1.9+**
   ```bash
   # Linux (via juliaup)
   curl -fsSL https://install.julialang.org | sh
   ```

2. **Clonar/Atualizar Repositório**
   ```bash
   git clone https://github.com/agourakis82/darwin-pbpk-platform.git
   cd darwin-pbpk-platform/julia-migration
   ```

3. **Setup do Projeto**
   ```julia
   using Pkg
   Pkg.activate(".")
   Pkg.instantiate()
   ```

4. **Usar Nova API**
   ```julia
   using DarwinPBPK
   # Ver documentação: julia-migration/EXECUTION_GUIDE.md
   ```

---

## 📊 Benefícios da Migração

### Performance
- **ODE Solver:** 4.5ms (4× mais rápido que Python)
- **Validação científica:** GMFE 1.036, 100% within folds

### Qualidade
- **Type Safety:** Unitful.jl (verificação de unidades)
- **Testes:** 6/6 passando
- **Documentação:** Completa e atualizada

---

## 📚 Documentação

- **Guia de Execução:** `julia-migration/EXECUTION_GUIDE.md`
- **Tutorial:** `julia-migration/docs/TUTORIAL.md`
- **Migração Completa:** `docs/MIGRATION_TO_JULIA_COMPLETE.md`
- **Release Notes:** `RELEASE_v2.0.0-julia.md`

---

## 🔗 Links

- **GitHub Release:** https://github.com/agourakis82/darwin-pbpk-platform/releases/tag/v2.0.0-julia
- **Tag:** `v2.0.0-julia`
- **Documentação:** `docs/`

---

## ❓ Suporte

Para dúvidas ou problemas:
1. Verificar documentação em `julia-migration/`
2. Abrir issue no GitHub
3. Consultar `docs/MIGRATION_TO_JULIA_COMPLETE.md`

---

## 🙏 Agradecimentos

Obrigado por usar o Darwin PBPK Platform! A migração para Julia traz melhorias significativas em performance e qualidade científica.

---

**Autor:** Dr. Demetrios Agourakis
**Data:** 2025-11-18


