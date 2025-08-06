#!/usr/bin/env python3
"""
Sistema de Análise e Visualização para Detecção de Anomalias - VERSÃO COMPLETA
Gera gráficos de treinamento, métricas e análise de resultados
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import time
from datetime import datetime, timedelta
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para português e estilo
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

class TrainingVisualizer:
    """
    Classe para visualizar resultados de treinamento dos modelos
    Gera gráficos de convergência, loss e métricas
    """
    
    def __init__(self, output_dir: str = "data/analysis"):
        """
        Inicializa o visualizador
        
        Args:
            output_dir: Diretório para salvar gráficos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurar cores
        self.colors = {
            'train': '#2E86AB',     # Azul
            'validation': '#A23B72', # Roxo
            'test': '#F18F01',      # Laranja
            'anomaly': '#C73E1D',   # Vermelho
            'normal': '#4CAF50',    # Verde
            'grid': '#E8E8E8'       # Cinza claro
        }
        
        print(f"TrainingVisualizer inicializado - output: {output_dir}")
    
    def plot_training_history(self, history: Dict, model_name: str = "Modelo") -> str:
        """
        Plota histórico de treinamento (loss e métricas)
        
        Args:
            history: Histórico do treinamento (formato Keras)
            model_name: Nome do modelo para título
            
        Returns:
            Caminho do arquivo salvo
        """
        # Criar figura com subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Histórico de Treinamento - {model_name}', fontsize=16, fontweight='bold')
        
        # Gráfico 1: Loss
        epochs = range(1, len(history['loss']) + 1)
        axes[0].plot(epochs, history['loss'], color=self.colors['train'], linewidth=2, label='Loss de Treinamento')
        if 'val_loss' in history:
            axes[0].plot(epochs, history['val_loss'], color=self.colors['validation'], linewidth=2, label='Loss de Validação')
        
        axes[0].set_title('Evolução do Loss (MSE)', fontweight='bold')
        axes[0].set_xlabel('Épocas')
        axes[0].set_ylabel('MSE')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')  # Escala logarítmica para loss
        
        # Gráfico 2: MAE (Mean Absolute Error)
        if 'mae' in history:
            axes[1].plot(epochs, history['mae'], color=self.colors['train'], linewidth=2, label='MAE de Treinamento')
        if 'val_mae' in history:
            axes[1].plot(epochs, history['val_mae'], color=self.colors['validation'], linewidth=2, label='MAE de Validação')
        
        axes[1].set_title('Mean Absolute Error (MAE)', fontweight='bold')
        axes[1].set_xlabel('Épocas')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Salvar
        filename = f"training_history_{model_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Gráfico de treinamento salvo: {filepath}")
        return filepath


class MetricsAnalyzer:
    """
    Classe para análise de métricas de performance
    Calcula e visualiza acurácia, precisão, recall, F1-score, etc.
    """
    
    def __init__(self, output_dir: str = "data/analysis"):
        """
        Inicializa o analisador de métricas
        
        Args:
            output_dir: Diretório para salvar gráficos
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"MetricsAnalyzer inicializado - output: {output_dir}")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_scores: np.ndarray = None) -> Dict:
        """
        Calcula métricas completas de classificação
        
        Args:
            y_true: Labels verdadeiros (0/1)
            y_pred: Predições (0/1)
            y_scores: Scores de confiança (opcional)
            
        Returns:
            Dict com todas as métricas
        """
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Métricas derivadas
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Taxa de Falsos Positivos
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Taxa de Falsos Negativos
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'false_positive_rate': float(fpr),
            'false_negative_rate': float(fnr),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': len(y_true)
        }
        
        # ROC AUC se scores disponíveis
        if y_scores is not None:
            try:
                fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr_curve, tpr_curve)
                metrics['roc_auc'] = float(roc_auc)
                metrics['roc_curve'] = {
                    'fpr': fpr_curve.tolist(),
                    'tpr': tpr_curve.tolist()
                }
            except Exception as e:
                print(f"Erro ao calcular ROC AUC: {e}")
        
        # Adicionar timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        
        print(f"Métricas calculadas - Acurácia: {accuracy:.3f}, F1: {f1:.3f}")
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             title: str = "Matriz de Confusão") -> str:
        """
        Plota matriz de confusão
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
            title: Título do gráfico
            
        Returns:
            Caminho do arquivo salvo
        """
        # Calcular matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        
        # Criar figura
        plt.figure(figsize=(10, 8))
        
        # Plot com seaborn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomalia'],
                   yticklabels=['Normal', 'Anomalia'],
                   cbar_kws={'label': 'Número de Amostras'})
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predições', fontsize=14)
        plt.ylabel('Valores Reais', fontsize=14)
        
        # Adicionar estatísticas
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        plt.figtext(0.02, 0.02, f'Acurácia: {accuracy:.3f}', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        # Salvar
        filename = f"confusion_matrix_{int(time.time())}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Matriz de confusão salva: {filepath}")
        return filepath
    
    def plot_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                       title: str = "Curva ROC") -> str:
        """
        Plota curva ROC
        
        Args:
            y_true: Labels verdadeiros
            y_scores: Scores de confiança
            title: Título do gráfico
            
        Returns:
            Caminho do arquivo salvo
        """
        # Calcular ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Criar figura
        plt.figure(figsize=(10, 8))
        
        # Plot da curva ROC
        plt.plot(fpr, tpr, color='#2E86AB', linewidth=3, 
                label=f'Curva ROC (AUC = {roc_auc:.3f})')
        
        # Linha diagonal (classificador aleatório)
        plt.plot([0, 1], [0, 1], color='#C73E1D', linestyle='--', linewidth=2,
                label='Classificador Aleatório (AUC = 0.5)')
        
        # Configurações
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (1 - Especificidade)', fontsize=14)
        plt.ylabel('Taxa de Verdadeiros Positivos (Sensibilidade)', fontsize=14)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Salvar
        filename = f"roc_curve_{int(time.time())}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Curva ROC salva: {filepath}")
        return filepath


class ComprehensiveAnalyzer:
    """
    Classe principal que integra todos os analisadores
    Gera relatórios completos do sistema
    """
    
    def __init__(self, config_file: str = "config.json"):
        """
        Inicializa o analisador completo
        
        Args:
            config_file: Arquivo de configuração
        """
        self.output_dir = "data/analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Inicializar sub-analisadores
        self.training_visualizer = TrainingVisualizer(self.output_dir)
        self.metrics_analyzer = MetricsAnalyzer(self.output_dir)
        
        print("ComprehensiveAnalyzer inicializado")
    
    def generate_training_report(self, training_histories: Dict[str, Dict],
                                model_paths: Dict[str, str] = None) -> Dict[str, str]:
        """
        Gera relatório completo de treinamento
        
        Args:
            training_histories: Históricos de treinamento dos modelos
            model_paths: Caminhos dos modelos salvos
            
        Returns:
            Dict com caminhos dos arquivos gerados
        """
        print("Gerando relatório completo de treinamento...")
        
        generated_files = {}
        
        # 1. Gráficos individuais de treinamento
        for model_name, history in training_histories.items():
            filepath = self.training_visualizer.plot_training_history(history, model_name)
            generated_files[f"training_{model_name.lower()}"] = filepath
        
        # 2. Relatório de texto
        report_text = self._generate_training_text_report(training_histories, model_paths)
        report_file = os.path.join(self.output_dir, "training_report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        generated_files["training_report"] = report_file
        
        print(f"Relatório de treinamento gerado - {len(generated_files)} arquivos")
        return generated_files
    
    def generate_evaluation_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_scores: np.ndarray = None,
                                  model_name: str = "Sistema") -> Dict[str, str]:
        """
        Gera relatório completo de avaliação
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Predições
            y_scores: Scores de confiança (opcional)
            model_name: Nome do modelo
            
        Returns:
            Dict com caminhos dos arquivos gerados
        """
        print(f"Gerando relatório de avaliação para {model_name}")
        
        generated_files = {}
        
        # 1. Calcular métricas
        metrics = self.metrics_analyzer.calculate_metrics(y_true, y_pred, y_scores)
        
        # 2. Matriz de confusão
        cm_path = self.metrics_analyzer.plot_confusion_matrix(
            y_true, y_pred, f"Matriz de Confusão - {model_name}")
        generated_files["confusion_matrix"] = cm_path
        
        # 3. Curva ROC (se scores disponíveis)
        if y_scores is not None:
            roc_path = self.metrics_analyzer.plot_roc_curve(
                y_true, y_scores, f"Curva ROC - {model_name}")
            generated_files["roc_curve"] = roc_path
        
        # 4. Relatório de classificação
        report_text = f"""
=== RELATÓRIO DE CLASSIFICAÇÃO ===
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Modelo: {model_name}

=== MÉTRICAS PRINCIPAIS ===
Acurácia:               {metrics['accuracy']:.3f}
Precisão:               {metrics['precision']:.3f}
Recall (Sensibilidade): {metrics['recall']:.3f}
F1-Score:               {metrics['f1_score']:.3f}
Especificidade:         {metrics['specificity']:.3f}

=== TAXAS DE ERRO ===
Taxa de Falsos Positivos: {metrics['false_positive_rate']:.3f}
Taxa de Falsos Negativos: {metrics['false_negative_rate']:.3f}

=== MATRIZ DE CONFUSÃO ===
                Predito
Real        Normal  Anomalia
Normal      {metrics['true_negatives']:6d}  {metrics['false_positives']:8d}
Anomalia    {metrics['false_negatives']:6d}  {metrics['true_positives']:8d}

=== ESTATÍSTICAS DETALHADAS ===
Total de Amostras:        {metrics['total_samples']}
Verdadeiros Positivos:    {metrics['true_positives']}
Verdadeiros Negativos:    {metrics['true_negatives']}
Falsos Positivos:         {metrics['false_positives']}
Falsos Negativos:         {metrics['false_negatives']}
        """
        
        report_path = os.path.join(self.output_dir, f"classification_report_{model_name.lower().replace(' ', '_')}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        generated_files["classification_report"] = report_path
        
        # 5. Salvar métricas em JSON
        metrics_file = os.path.join(self.output_dir, f"metrics_{model_name.lower().replace(' ', '_')}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        generated_files["metrics_json"] = metrics_file
        
        print(f"Relatório de avaliação gerado - {len(generated_files)} arquivos")
        return generated_files
    
    def generate_complete_analysis(self, data_package: Dict) -> Dict[str, str]:
        """
        Gera análise completa do sistema
        
        Args:
            data_package: Pacote com todos os dados necessários
                - training_histories: Históricos de treinamento
                - evaluation_data: Dados de avaliação (y_true, y_pred, y_scores)
                - performance_data: Dados de performance
                - model_info: Informações dos modelos
                
        Returns:
            Dict com todos os arquivos gerados
        """
        print("🚀 Iniciando análise completa do sistema...")
        
        all_generated_files = {}
        
        # 1. Relatório de Treinamento
        if 'training_histories' in data_package:
            training_files = self.generate_training_report(
                data_package['training_histories'],
                data_package.get('model_paths', {})
            )
            all_generated_files.update(training_files)
        
        # 2. Relatório de Avaliação
        if 'evaluation_data' in data_package:
            eval_data = data_package['evaluation_data']
            eval_files = self.generate_evaluation_report(
                eval_data['y_true'],
                eval_data['y_pred'],
                eval_data.get('y_scores'),
                eval_data.get('model_name', 'Sistema')
            )
            all_generated_files.update(eval_files)
        
        # 3. Relatório Final em HTML
        html_report = self._generate_html_report(all_generated_files, data_package)
        html_file = os.path.join(self.output_dir, "complete_analysis_report.html")
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        all_generated_files["html_report"] = html_file
        
        # 4. README com instruções
        readme_content = self._generate_readme(all_generated_files)
        readme_file = os.path.join(self.output_dir, "README.md")
        
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        all_generated_files["readme"] = readme_file
        
        print(f"✅ Análise completa finalizada - {len(all_generated_files)} arquivos gerados")
        return all_generated_files
    
    def _generate_training_text_report(self, histories: Dict[str, Dict],
                                     model_paths: Dict[str, str] = None) -> str:
        """Gera relatório de texto para treinamento"""
        
        report = f"""# Relatório de Treinamento - Sistema de Detecção de Anomalias

**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Resumo Executivo

Este relatório apresenta os resultados do treinamento dos modelos de deep learning implementados no sistema de detecção de anomalias.

## Modelos Treinados

"""
        
        for model_name, history in histories.items():
            epochs = len(history['loss'])
            final_loss = history['loss'][-1]
            min_loss = min(history['loss'])
            
            report += f"""### {model_name}

- **Épocas treinadas:** {epochs}
- **Loss final:** {final_loss:.6f}
- **Melhor loss:** {min_loss:.6f}
- **Convergência:** {'Sim' if final_loss < min_loss * 1.1 else 'Necessita mais épocas'}

"""
            
            if 'val_loss' in history:
                val_loss = history['val_loss'][-1]
                min_val_loss = min(history['val_loss'])
                overfitting = val_loss > min_val_loss * 1.2
                
                report += f"""- **Loss de validação final:** {val_loss:.6f}
- **Melhor loss de validação:** {min_val_loss:.6f}
- **Overfitting detectado:** {'Sim' if overfitting else 'Não'}

"""
        
        report += """## Análise de Convergência

Os gráficos de convergência mostram a evolução do loss durante o treinamento. Uma boa convergência é caracterizada por:

1. **Diminuição consistente do loss**
2. **Estabilização em valor baixo**
3. **Proximidade entre loss de treino e validação**

## Recomendações

- Monitore overfitting através da divergência entre loss de treino e validação
- Considere early stopping se o loss de validação parar de melhorar
- Ajuste learning rate se a convergência for muito lenta ou instável

"""
        
        return report
    
    def _generate_html_report(self, files: Dict[str, str], data_package: Dict) -> str:
        """Gera relatório HTML completo"""
        
        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Análise - Sistema de Detecção de Anomalias</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        h1 {{ color: #2E86AB; text-align: center; border-bottom: 3px solid #2E86AB; padding-bottom: 10px; }}
        h2 {{ color: #333; border-left: 4px solid #2E86AB; padding-left: 15px; }}
        h3 {{ color: #666; }}
        .metric {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .image-container {{ text-align: center; margin: 20px 0; }}
        .image-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #2E86AB; color: white; }}
        .success {{ color: #4CAF50; font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚨 Sistema de Detecção de Anomalias</h1>
        <h2>Relatório de Análise Completa</h2>
        
        <div class="metric">
            <strong>Data de Geração:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Arquivos Gerados:</strong> {len(files)}<br>
            <strong>Status:</strong> <span class="success">✅ Completo</span>
        </div>
        
        <h2>📈 Análise de Treinamento</h2>
        <p>Esta seção apresenta os resultados do treinamento dos modelos de deep learning.</p>
        
        <h2>📊 Análise de Performance</h2>
        <p>Métricas de classificação e análise de performance do sistema.</p>
        
        <h2>📁 Arquivos Gerados</h2>
        <table>
            <tr><th>Tipo</th><th>Arquivo</th><th>Descrição</th></tr>
"""
        
        file_descriptions = {
            "training_cae": "Gráfico de treinamento do CAE",
            "training_convlstm": "Gráfico de treinamento do ConvLSTM",
            "confusion_matrix": "Matriz de confusão",
            "roc_curve": "Curva ROC",
            "classification_report": "Relatório detalhado de classificação",
            "training_report": "Relatório completo de treinamento"
        }
        
        for file_type, filepath in files.items():
            filename = os.path.basename(filepath)
            description = file_descriptions.get(file_type, "Arquivo de análise")
            html += f"<tr><td>{file_type.replace('_', ' ').title()}</td><td>{filename}</td><td>{description}</td></tr>"
        
        html += """
        </table>
        
        <div class="footer">
            <p>Relatório gerado automaticamente pelo Sistema de Detecção de Anomalias</p>
            <p>🔬 Desenvolvido com Python, TensorFlow e OpenCV</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_readme(self, files: Dict[str, str]) -> str:
        """Gera README com instruções"""
        
        readme = f"""# 📊 Relatório de Análise - Sistema de Detecção de Anomalias

**Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📁 Arquivos Gerados

Este diretório contém {len(files)} arquivos de análise gerados automaticamente:

### 📈 Gráficos de Treinamento
"""
        
        for file_type, filepath in files.items():
            if "training" in file_type:
                filename = os.path.basename(filepath)
                readme += f"- `{filename}` - Histórico de treinamento\n"
        
        readme += """
### 📊 Métricas de Avaliação
"""
        
        for file_type, filepath in files.items():
            if file_type in ["confusion_matrix", "roc_curve", "classification_report"]:
                filename = os.path.basename(filepath)
                readme += f"- `{filename}` - Análise de performance\n"
        
        readme += f"""
## 🚀 Como Usar

1. **Visualizar Relatório HTML:** Abra `complete_analysis_report.html` no navegador
2. **Gráficos:** Todas as imagens estão em alta resolução (300 DPI)
3. **Dados Brutos:** Arquivos JSON contêm métricas numéricas
4. **Relatórios de Texto:** Arquivos .txt e .md para documentação

---
*Relatório gerado automaticamente pelo Sistema de Detecção de Anomalias*
"""
        
        return readme


# Exemplo de uso básico
if __name__ == "__main__":
    print("Sistema de Análise e Visualização - Teste")
    
    # Criar analisador
    analyzer = ComprehensiveAnalyzer()
    
    print("✅ Sistema carregado com sucesso!")
    print("Use: from analysis_visualization_system import ComprehensiveAnalyzer")