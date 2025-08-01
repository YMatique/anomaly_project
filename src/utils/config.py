"""
Configurações centralizadas do sistema de detecção de anomalias
Otimizado para i5 11Gen, 16GB RAM, 4 cores
"""

import os
import json
from dataclasses import dataclass
from typing import Tuple, Dict, List

@dataclass
class VideoConfig:
    """Configurações de vídeo otimizadas para o hardware"""
    # Resolução adaptativa baseada no modo
    realtime_resolution: Tuple[int, int] = (640, 480)
    training_resolution: Tuple[int, int] = (1280, 720)
    fps: int = 30
    buffer_size: int = 5
    frame_skip: int = 2  # Processa 1 a cada 2 frames para economia

@dataclass
class ModelConfig:
    """Configurações dos modelos de ML"""
    # CAE (Convolutional Autoencoder)
    cae_input_shape: Tuple[int, int, int] = (64, 64, 3)
    cae_latent_dim: int = 128
    cae_batch_size: int = 16
    
    # ConvLSTM
    convlstm_sequence_length: int = 10
    convlstm_filters: int = 64
    convlstm_kernel_size: Tuple[int, int] = (3, 3)
    
    # Optical Flow
    optical_flow_method: str = "lucas_kanade"  # ou "farneback"
    flow_threshold: float = 2.0
    
    # Thresholds para detecção
    anomaly_threshold: float = 0.7
    movement_threshold: float = 15.0

@dataclass
class SystemConfig:
    """Configurações gerais do sistema"""
    # Threading (otimizado para 4 cores)
    max_threads: int = 4
    queue_size: int = 10
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "data/logs/anomaly_detection.log"
    
    # Alertas
    alert_cooldown: int = 5  # segundos entre alertas do mesmo tipo
    save_alert_frames: bool = True
    
    # Caminhos
    models_path: str = "models/"
    data_path: str = "data/"
    videos_path: str = "data/videos/"
    
    # Interface Web
    web_host: str = "127.0.0.1"
    web_port: int = 5000
    web_debug: bool = True

class Config:
    """Classe principal de configuração"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        
        # Configurações padrão
        self.video = VideoConfig()
        self.model = ModelConfig()
        self.system = SystemConfig()
        
        # Tipos de anomalias que o sistema detecta
        self.anomaly_types = {
            "security": {
                "intrusion": "Intrusão detectada",
                "break_in": "Tentativa de arrombamento",
                "night_movement": "Movimento noturno suspeito",
                "stranger_loitering": "Pessoa estranha por muito tempo",
                "unknown_vehicle": "Veículo desconhecido"
            },
            "health": {
                "fall": "Queda detectada",
                "collapse": "Colapso/desmaio detectado",
                "immobility": "Imobilidade prolongada",
                "erratic_movement": "Movimento errático",
                "no_movement": "Ausência de movimento"
            }
        }
        
        # Carregar configurações personalizadas se existirem
        self.load_config()
        
        # Garantir que diretórios existam
        self._ensure_directories()
    
    def load_config(self):
        """Carrega configurações de arquivo JSON se existir"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Atualizar configurações com dados do arquivo
                if 'video' in config_data:
                    for key, value in config_data['video'].items():
                        if hasattr(self.video, key):
                            setattr(self.video, key, value)
                
                if 'model' in config_data:
                    for key, value in config_data['model'].items():
                        if hasattr(self.model, key):
                            setattr(self.model, key, value)
                
                if 'system' in config_data:
                    for key, value in config_data['system'].items():
                        if hasattr(self.system, key):
                            setattr(self.system, key, value)
                            
            except Exception as e:
                print(f"Erro ao carregar configurações: {e}")
    
    def save_config(self):
        """Salva configurações atuais em arquivo JSON"""
        config_data = {
            'video': self.video.__dict__,
            'model': self.model.__dict__,
            'system': self.system.__dict__
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print(f"Erro ao salvar configurações: {e}")
    
    def _ensure_directories(self):
        """Garante que todos os diretórios necessários existam"""
        directories = [
            self.system.models_path,
            self.system.data_path,
            self.system.videos_path,
            os.path.dirname(self.system.log_file),
            "data/logs",
            "data/training",
            "data/alerts"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> str:
        """Retorna caminho completo para um modelo"""
        return os.path.join(self.system.models_path, f"{model_name}.h5")
    
    def get_alert_message(self, anomaly_type: str, subtype: str) -> str:
        """Retorna mensagem de alerta para um tipo de anomalia"""
        if anomaly_type in self.anomaly_types:
            if subtype in self.anomaly_types[anomaly_type]:
                return self.anomaly_types[anomaly_type][subtype]
        return f"Anomalia detectada: {anomaly_type} - {subtype}"
    
    def update_thresholds(self, **kwargs):
        """Atualiza thresholds dinamicamente"""
        for key, value in kwargs.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value)
        self.save_config()
    
    def optimize_for_hardware(self):
        """Otimizações específicas para o hardware disponível"""
        import psutil
        
        # Ajustar configurações baseadas no hardware real
        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Ajustar threads baseado nos cores disponíveis
        self.system.max_threads = min(cpu_count, 4)
        
        # Ajustar batch size baseado na memória
        if memory_gb >= 16:
            self.model.cae_batch_size = 32
        elif memory_gb >= 8:
            self.model.cae_batch_size = 16
        else:
            self.model.cae_batch_size = 8
        
        print(f"Sistema otimizado para: {cpu_count} cores, {memory_gb:.1f}GB RAM")
        self.save_config()

# Instância global de configuração
config = Config()