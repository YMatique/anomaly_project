# Relatório de Treinamento - Sistema de Detecção de Anomalias

**Data:** 2025-08-06 06:26:01

## Resumo Executivo

Este relatório apresenta os resultados do treinamento dos modelos de deep learning implementados no sistema de detecção de anomalias.

## Modelos Treinados

### CAE

- **Épocas treinadas:** 20
- **Loss final:** 0.036200
- **Melhor loss:** 0.036200
- **Convergência:** Sim

- **Loss de validação final:** 0.038600
- **Melhor loss de validação:** 0.038600
- **Overfitting detectado:** Não

### ConvLSTM

- **Épocas treinadas:** 20
- **Loss final:** 0.037900
- **Melhor loss:** 0.037900
- **Convergência:** Sim

- **Loss de validação final:** 0.036700
- **Melhor loss de validação:** 0.036700
- **Overfitting detectado:** Não

## Análise de Convergência

Os gráficos de convergência mostram a evolução do loss durante o treinamento. Uma boa convergência é caracterizada por:

1. **Diminuição consistente do loss**
2. **Estabilização em valor baixo**
3. **Proximidade entre loss de treino e validação**

## Recomendações

- Monitore overfitting através da divergência entre loss de treino e validação
- Considere early stopping se o loss de validação parar de melhorar
- Ajuste learning rate se a convergência for muito lenta ou instável

