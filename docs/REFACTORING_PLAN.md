# Plano de Refatoração - Darwin PBPK Platform

**Data:** 2025-11-18
**Status:** Em Execução
**Autor:** Dr. Demetrios Agourakis + AI Assistant

---

## 🎯 Objetivos

Refatorar o repositório Python para:
- ✅ Organização clara e profissional
- ✅ Código limpo e manutenível
- ✅ Documentação consolidada
- ✅ Estrutura pronta para publicação Q1
- ✅ Compatibilidade com migração Julia

---

## 📋 Áreas de Refatoração Identificadas

### 1. Estrutura de Diretórios
**Problema:** Arquivos espalhados, sem organização clara

**Ação:**
- [ ] Consolidar scripts em `scripts/` por categoria
- [ ] Mover documentação para `docs/`
- [ ] Organizar notebooks em `notebooks/`
- [ ] Limpar arquivos temporários/logs

### 2. Código Duplicado
**Problema:** Funções similares em múltiplos arquivos

**Ação:**
- [ ] Identificar duplicações
- [ ] Criar módulos compartilhados
- [ ] Refatorar para reutilização

### 3. Imports e Dependências
**Problema:** Imports não utilizados, dependências desnecessárias

**Ação:**
- [ ] Limpar imports não utilizados
- [ ] Consolidar dependências
- [ ] Verificar compatibilidade

### 4. TODOs e Código Legado
**Problema:** 83+ arquivos com TODOs/FIXMEs

**Ação:**
- [ ] Resolver TODOs críticos
- [ ] Documentar TODOs futuros
- [ ] Remover código obsoleto

### 5. Documentação
**Problema:** Documentação espalhada, duplicada

**Ação:**
- [ ] Consolidar documentação
- [ ] Criar índice central
- [ ] Remover duplicações

### 6. Testes
**Problema:** Testes desorganizados

**Ação:**
- [ ] Organizar testes por módulo
- [ ] Adicionar testes faltantes
- [ ] Melhorar cobertura

---

## 🚀 Fases de Refatoração

### FASE 1: Limpeza Estrutural (Prioridade ALTA)
- [ ] Organizar estrutura de diretórios
- [ ] Mover arquivos para locais apropriados
- [ ] Limpar arquivos temporários
- [ ] Criar `.gitignore` adequado

### FASE 2: Refatoração de Código (Prioridade ALTA)
- [ ] Identificar e remover duplicações
- [ ] Consolidar funções comuns
- [ ] Limpar imports não utilizados
- [ ] Padronizar estilo de código

### FASE 3: Documentação (Prioridade MÉDIA)
- [ ] Consolidar documentação
- [ ] Criar índice central
- [ ] Atualizar READMEs
- [ ] Remover duplicações

### FASE 4: Testes e Qualidade (Prioridade MÉDIA)
- [ ] Organizar testes
- [ ] Adicionar testes faltantes
- [ ] Melhorar cobertura
- [ ] Adicionar type hints

---

## 📊 Métricas de Sucesso

- ✅ Estrutura organizada (arquivos na raiz < 30)
- ✅ Código sem duplicações críticas
- ✅ Imports limpos
- ✅ Documentação consolidada
- ✅ Testes organizados

---

**Última atualização:** 2025-11-18

