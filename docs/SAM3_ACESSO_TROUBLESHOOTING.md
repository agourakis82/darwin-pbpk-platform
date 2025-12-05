# SAM-3 - Troubleshooting de Acesso

**Data**: 2025-12-01  
**Status**: Aprovação recebida, mas acesso ainda negado

---

## ⚠️ PROBLEMA

Acesso foi aprovado, mas ainda recebe erro 403 ao tentar baixar o modelo.

---

## 🔧 SOLUÇÕES

### Solução 1: Verificar e Aceitar Termos no Site

1. **Acesse**: https://huggingface.co/facebook/sam3
2. **Procure por**:
   - Botão "Accept terms" (Aceitar termos)
   - Botão "I accept" (Eu aceito)
   - Qualquer botão de aceitação de licença
3. **Clique e aceite** os termos
4. **Tente novamente**

### Solução 2: Re-autenticar

```bash
# Fazer logout
hf auth logout

# Fazer login novamente
hf auth login
# Cole o token quando solicitado
```

### Solução 3: Verificar Token no Site

1. Acesse: https://huggingface.co/settings/tokens
2. Verifique se há um token ativo
3. Se necessário, gere um novo token
4. Use o novo token: `hf auth login`

### Solução 4: Aguardar Propagação

Às vezes a aprovação leva alguns minutos para se propagar:
- Aguarde 5-10 minutos
- Tente novamente

---

## 🧪 TESTE RÁPIDO

```bash
cd analysis/fractal_poc
python check_sam3_access.py
```

Se ainda der erro:
```bash
# Verificar autenticação
hf auth whoami

# Tentar carregar modelo diretamente
python -c "from sam3.model_builder import build_sam3_image_model; model = build_sam3_image_model()"
```

---

## 📞 SE NADA FUNCIONAR

1. Verifique se realmente foi aprovado:
   - Acesse: https://huggingface.co/facebook/sam3
   - Veja se ainda há botão "Request access" ou se mostra "Access granted"

2. Verifique o email de aprovação (se recebeu)

3. Entre em contato com suporte do HuggingFace se necessário

---

**Última atualização**: 2025-12-01

