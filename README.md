# 🚨 Sistema de Detecção de Anomalias em Tempo Real

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.6+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Sistema inteligente de detecção de anomalias usando **Optical Flow**, **CAE (Convolutional Autoencoder)** e **ConvLSTM**, otimizado para hardware i5 11Gen com 16GB RAM.

## 🎯 Funcionalidades

### 🔐 Anomalias de Segurança
- **Intrusão**: Pessoas não autorizadas
- **Arrombamento**: Tentativas de forçar portas/janelas  
- **Movimento noturno**: Atividade suspeita durante a noite
- **Permanência prolongada**: Estranhos por muito tempo
- **Veículos desconhecidos**: Detecção de veículos suspeitos

### 🏥 Anomalias de Saúde/Emergência
- **Quedas**: Especialmente para idosos
- **Desmaios/Colapsos**: Detecção de emergências médicas
- **Imobilidade prolongada**: Falta de movimento por períodos anormais
- **Movimentos erráticos**: Padrões de movimento anômalos
- **Ausência de movimento**: Detecção de imobilidade total

## 🏗️ Arquitetura

```
🎥 Entrada de Vídeo → 🔍 Optical Flow → 🧠 CAE → ⏰ ConvLSTM → 🚨 Alertas
     (Webcam/Arquivo)    (Detecção rápida)  (Análise espacial)  (Análise temporal)  (Multi-canal)
```

### 🧩 Componentes Principais

- **Input Manager**: Gerencia webcam e arquivos de vídeo
- **Processing Engine**: Coordena pipeline de detecção em cascata
- **Output Manager**: Interface visual, alertas e relatórios
- **Optical Flow Detector**: Primeira camada - detecção de movimento
- **Deep Learning Detector**: CAE + ConvLSTM para análise profunda
- **Web Interface**: Dashboard moderno e responsivo

## 📋 Requisitos do Sistema

### 💻 Hardware Recomendado
- **CPU**: Intel i5 11Gen ou superior
- **RAM**: 16GB (mínimo 8GB)
- **GPU**: Opcional (melhora performance do TensorFlow)
- **Câmera**: USB/integrada funcional
- **Armazenamento**: 5GB livres

### 🔧 Software
- **Python**: 3.8 ou superior
- **Sistema Operacional**: Windows 10/11, Linux Ubuntu 20+, macOS 10.15+

## 🚀 Instalação

### 1. Clonar/Baixar o Projeto
```bash
# Se usando Git
git clone [URL_DO_REPOSITORIO]
cd projeto_anomalias

# Ou extrair arquivo ZIP baixado
```

### 2. Instalar Dependências
```bash
# Instalar dependências Python
pip install -r requirements.txt

# Para GPU (opcional - melhora performance)
pip install tensorflow-gpu>=2.10.0
```

### 3. Configuração Inicial
```bash
# Criar configuração padrão
python main.py create-config

# Teste rápido do sistema
python main.py demo
```

## 🎮 Como Usar

### 🖥️ Modo Console (Recomendado)

#### Detecção com Webcam
```bash
# Básico
python main.py --mode webcam

# Câmera específica
python main.py --mode webcam --camera 1

# Sem interface visual
python main.py --mode webcam --no-display
```

#### Análise de Vídeo
```bash
# Analisar arquivo
python main.py --mode video --video meu_video.mp4

# Com repetição
python main.py --mode video --video meu_video.mp4 --loop
```

#### Treinamento de Modelos
```bash
# Treinar por 10 minutos (padrão)
python main.py --mode train

# Treinar por 20 minutos
python main.py --mode train --train-duration 20
```

### 🌐 Interface Web

```bash
# Iniciar servidor web
python src/web/app.py

# Acessar no navegador
http://localhost:5000
```

**Características da Interface Web:**
- 📊 Dashboard em tempo real
- 📈 Gráficos de performance
- ⚙️ Controles de configuração
- 🚨 Alertas visuais
- 📱 Design responsivo

### ⌨️ Controles Durante Execução

| Tecla | Função |
|-------|--------|
| `Q` ou `ESC` | Sair do sistema |
| `P` | Pausar/Retomar processamento |
| `S` | Mostrar estatísticas |
| `R` | Exportar relatório |
| `H` | Mostrar ajuda |
| `I` | Toggle painel de informações |
| `F` | Toggle overlay optical flow |
| `A` | Toggle marcadores de anomalia |

## 📧 Configurar Alertas

### Email
```bash
python main.py --mode webcam \
  --email-server smtp.gmail.com \
  --email-port 587 \
  --email-user seu@email.com \
  --email-pass suasenha \
  --email-recipients admin@empresa.com seguranca@empresa.com
```

### Webhook
```bash
python main.py --mode webcam \
  --webhook-url https://api.empresa.com/alertas
```

## 🔧 Configuração Avançada

### Arquivo config.json
```json
{
    "video": {
        "realtime_resolution": [640, 480],
        "fps": 30,
        "frame_skip": 2
    },
    "model": {
        "optical_flow_method": "farneback",
        "flow_threshold": 2.0,
        "anomaly_threshold": 0.7,
        "movement_threshold": 15.0
    },
    "system": {
        "max_threads": 4,
        "save_alert_frames": true,
        "web_port": 5000
    }
}
```

### Otimizações de Performance

```python
# No código, você pode ajustar:
config.video.frame_skip = 3  # Pular mais frames
config.model.cae_batch_size = 8  # Reduzir batch size
config.system.max_threads = 2  # Menos threads
```

## 📁 Estrutura de Arquivos

```
projeto_anomalias/
├── main.py                    # 🚀 Sistema principal
├── requirements.txt           # 📦 Dependências
├── config.json               # ⚙️ Configurações
├── README.md                 # 📖 Este arquivo
│
├── src/                      # 📂 Código fonte
│   ├── core/                 # 🧠 Componentes principais
│   │   ├── input_manager.py
│   │   ├── processing_engine.py
│   │   └── output_manager.py
│   │
│   ├── detectors/            # 🔍 Detectores
│   │   ├── optical_flow_detector.py
│   │   └── deep_learning_detector.py
│   │
│   ├── utils/                # 🛠️ Utilitários
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── helpers.py
│   │
│   └── web/                  # 🌐 Interface web
│       ├── app.py
│       └── templates/
│           └── dashboard.html
│
├── models/                   # 🤖 Modelos treinados
├── data/                     # 📊 Dados do sistema
│   ├── videos/              # 🎬 Vídeos de teste
│   ├── logs/                # 📝 Logs do sistema
│   ├── alerts/              # 🚨 Frames de alerta
│   ├── captures/            # 📷 Capturas manuais
│   └── reports/             # 📄 Relatórios
│
└── tests/