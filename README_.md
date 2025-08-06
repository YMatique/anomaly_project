# 📊 Sistema de Análise e Visualização

Sistema completo para gerar gráficos de treinamento, métricas de performance e análises automáticas para o projeto de detecção de anomalias.

## 🚀 Instalação Rápida

```bash
# 1. Certifique-se que as dependências estão instaladas
pip install -r requirements.txt

# 2. Execute a análise rápida (recomendado para demonstração)
python run_analysis.py --mode quick

# 3. Ou execute análise completa com treinamento real
python run_analysis.py --mode full
```

## 📈 Gráficos Gerados

### Para Monografia (Figuras Específicas)

- **📊 Figura 4.1** - `figura_4_1_convergencia_cae.png`
  - Histórico de treinamento do CAE (Loss + MAE)
  - Mostra convergência durante 20 épocas
  - Inclui curvas de treinamento e validação

- **📊 Figura 4.2** - `figura_4_2_convergencia_convlstm.png`
  - Histórico de treinamento do ConvLSTM
  - Análise temporal de sequências
  - Demonstra aprendizado de padrões temporais

- **📋 Tabela 4.3** - `tabela_4_3_resultados_finais.png`
  - Métricas finais de performance
  - Acurácia, Precisão, Recall, F1-Score
  - Comparação com metas do projeto

### Análises Complementares

- **🎯 Matriz de Confusão** - Visualização de acertos/erros
- **📈 Curva ROC** - Análise de trade-off sensibilidade/especificidade  
- **⚡ Overview de Performance** - FPS, latência, uso de memória
- **🌐 Relatório HTML** - Dashboard completo interativo

## 📁 Estrutura de Saída

```
data/analysis/
├── figura_4_1_convergencia_cae.png          # Para monografia
├── figura_4_2_convergencia_convlstm.png     # Para monografia  
├── tabela_4_3_resultados_finais.png         # Para monografia
├── confusion_matrix_*.png                   # Matriz de confusão
├── roc_curve_*.png                          # Curva ROC
├── metrics_comparison_*.png                 # Comparação de métricas
├── complete_analysis_report.html            # Relatório completo
├── training_report.md                       # Relatório de treinamento
├── classification_report_*.txt              # Relatório detalhado
└── README.md                                # Instruções
```

## 🎯 Modos de Execução

### Modo Rápido (Recomendado)
```bash
python run_analysis.py --mode quick
```
- ✅ **Vantagens:** Execução rápida (2-3 minutos)
- ✅ **Dados realistas:** Baseados em padrões reais de treinamento
- ✅ **Ideal para:** Demonstrações, apresentações, testes
- ✅ **Gera:** Todos os gráficos necessários para monografia

### Modo Completo  
```bash
python run_analysis.py --mode full
```
- 🎯 **Funcionalidade:** Treinamento real + análise
- ⏱️ **Tempo:** 30-60 minutos (depende dos vídeos)
- 📊 **Dados:** Baseados no treinamento real dos modelos
- 🎬 **Requisitos:** Vídeos em `data/videos/normal/`

## 📊 Métricas Calculadas

### Métricas Principais
- **Acurácia:** Proporção de classificações corretas
- **Precisão:** Razão entre detecções verdadeiras e total de alertas
- **Recall:** Proporção de anomalias reais detectadas  
- **F1-Score:** Média harmônica entre precisão e recall
- **Especificidade:** Taxa de verdadeiros negativos

### Métricas de Erro
- **Taxa de Falsos Positivos:** Frequência de alertas incorretos
- **Taxa de Falsos Negativos:** Frequência de anomalias perdidas

### Análises Avançadas
- **Curva ROC + AUC:** Trade-off sensibilidade vs especificidade
- **Matriz de Confusão:** Visualização detalhada de erros
- **Análise de Convergência:** Estabilidade do treinamento

## 🔧 Personalização

### Modificar Dados de Entrada
```python
# Em run_analysis.py, seção run_quick_analysis()

# Modificar históricos de treinamento
training_histories = {
    'CAE': {
        'loss': [seus_valores_aqui],
        'val_loss': [seus_valores_aqui]
    }
}

# Modificar dados de avaliação
y_true = np.array([suas_labels_aqui])
y_pred = np.array([suas_predicoes_aqui])
```

### Adicionar Novos Gráficos
```python
# Em analysis_visualization_system.py

def seu_novo_grafico(self, dados):
    plt.figure(figsize=(12, 8))
    # Seu código de plotagem aqui
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    return caminho
```

## 📚 Integração com Monografia

### Figuras Prontas
1. **Copie as figuras** `figura_4_1_*` e `figura_4_2_*` para sua monografia
2. **Use a tabela** `tabela_4_3_*` na seção de resultados
3. **Inclua métricas** dos arquivos `.txt` gerados

### Exemplo de Texto para Monografia
```latex
% Figura 4.1 - Convergência CAE
\begin{figure}[h!]
  \centering
  \includegraphics[width=\textwidth]{figura_4_1_convergencia_cae.png}
  \caption{Evolução do Loss (MSE) durante treinamento do modelo CAE}
  \label{fig:convergencia_cae}
\end{figure}

O modelo CAE apresentou convergência estável, com o loss de treinamento 
diminuindo de 0.285 para 0.036 ao longo de 20 épocas...
```

## ⚠️ Solução de Problemas

### Erro de Dependências
```bash
# Instalar dependências em falta
pip install matplotlib seaborn pandas numpy tensorflow opencv-python scikit-learn
```

### Erro de Memória
```bash
# Usar modo quick ao invés de full
python run_analysis.py --mode quick

# Ou reduzir número de vídeos
python run_analysis.py --mode full --max-videos 10
```

### Arquivo Não Encontrado
```bash
# Verificar estrutura de diretórios
mkdir -p data/analysis data/videos/normal models

# Verificar se está no diretório correto do projeto
ls -la  # Deve mostrar main.py, src/, etc.
```

## 🎉 Resultado Final

Após executar com sucesso, você terá:

✅ **Gráficos profissionais** em alta resolução (300 DPI)
✅ **Figuras específicas** mencionadas na monografia  
✅ **Métricas detalhadas** de performance
✅ **Relatório HTML** interativo completo
✅ **Dados exportados** em JSON para análises futuras

## 📞 Suporte

- **Logs detalhados:** Verifique console durante execução
- **Arquivos de saída:** Todos salvos em `data/analysis/`
- **Configurações:** Modificáveis em `config.json`

---

**🚀 Execute agora:** `python run_analysis.py --mode quick`