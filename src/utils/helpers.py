"""
Funções auxiliares e utilitários para o sistema de detecção de anomalias
Otimizado para processamento eficiente de vídeo e imagens
"""

import cv2
import numpy as np
import os
import time
import json
from datetime import datetime, timedelta
from typing import Tuple, List, Dict, Optional, Union
import threading
from collections import deque
import psutil

class PerformanceMonitor:
    """Monitor de performance do sistema"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.last_frame_time = time.time()
        
    def update_fps(self):
        """Atualiza FPS baseado no tempo entre frames"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_frame_time)
        self.fps_history.append(fps)
        self.last_frame_time = current_time
        
    def update_processing_time(self, processing_time: float):
        """Atualiza tempo de processamento"""
        self.processing_times.append(processing_time)
        
    def update_memory_usage(self):
        """Atualiza uso de memória"""
        memory_percent = psutil.virtual_memory().percent
        self.memory_usage.append(memory_percent)
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas de performance"""
        stats = {
            "fps": {
                "current": self.fps_history[-1] if self.fps_history else 0,
                "average": np.mean(self.fps_history) if self.fps_history else 0,
                "min": np.min(self.fps_history) if self.fps_history else 0,
                "max": np.max(self.fps_history) if self.fps_history else 0
            },
            "processing_time": {
                "current": self.processing_times[-1] if self.processing_times else 0,
                "average": np.mean(self.processing_times) if self.processing_times else 0,
                "min": np.min(self.processing_times) if self.processing_times else 0,
                "max": np.max(self.processing_times) if self.processing_times else 0
            },
            "memory": {
                "current": self.memory_usage[-1] if self.memory_usage else 0,
                "average": np.mean(self.memory_usage) if self.memory_usage else 0
            }
        }
        return stats

class VideoProcessor:
    """Processador de vídeo otimizado"""
    
    @staticmethod
    def resize_frame(frame: np.ndarray, target_size: Tuple[int, int], 
                    interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Redimensiona frame mantendo aspect ratio quando possível
        """
        if frame is None:
            return None
            
        height, width = frame.shape[:2]
        target_width, target_height = target_size
        
        # Calcular aspect ratio
        aspect_ratio = width / height
        target_aspect_ratio = target_width / target_height
        
        if aspect_ratio > target_aspect_ratio:
            # Frame é mais largo
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            # Frame é mais alto
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        
        # Redimensionar
        resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
        
        # Padding se necessário para atingir tamanho exato
        if new_width != target_width or new_height != target_height:
            delta_w = target_width - new_width
            delta_h = target_height - new_height
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            resized = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                       cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        return resized
    
    @staticmethod
    def preprocess_frame(frame: np.ndarray, target_size: Tuple[int, int] = (64, 64),
                        normalize: bool = True) -> np.ndarray:
        """
        Pré-processa frame para modelos de ML
        """
        if frame is None:
            return None
        
        # Redimensionar
        processed = VideoProcessor.resize_frame(frame, target_size)
        
        # Normalizar se solicitado
        if normalize:
            processed = processed.astype(np.float32) / 255.0
        
        return processed
    
    @staticmethod
    def extract_roi(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extrai região de interesse do frame
        bbox: (x, y, width, height)
        """
        if frame is None:
            return None
        
        x, y, w, h = bbox
        height, width = frame.shape[:2]
        
        # Garantir que bbox está dentro dos limites
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        
        return frame[y:y+h, x:x+w]

class MovementAnalyzer:
    """Analisador de movimento otimizado"""
    
    def __init__(self, history_size: int = 30):
        self.history_size = history_size
        self.movement_history = deque(maxlen=history_size)
        self.position_history = deque(maxlen=history_size)
        
    def analyze_movement(self, optical_flow: np.ndarray) -> Dict:
        """
        Analisa padrões de movimento do optical flow
        """
        if optical_flow is None:
            return {"magnitude": 0, "direction": 0, "pattern": "none"}
        
        # Calcular magnitude e direção
        magnitude = np.sqrt(optical_flow[..., 0]**2 + optical_flow[..., 1]**2)
        direction = np.arctan2(optical_flow[..., 1], optical_flow[..., 0])
        
        # Estatísticas de movimento
        avg_magnitude = np.mean(magnitude)
        max_magnitude = np.max(magnitude)
        movement_pixels = np.sum(magnitude > 1.0)
        
        # Adicionar ao histórico
        movement_data = {
            "magnitude": avg_magnitude,
            "max_magnitude": max_magnitude,
            "movement_pixels": movement_pixels,
            "timestamp": time.time()
        }
        self.movement_history.append(movement_data)
        
        # Analisar padrões
        pattern = self._classify_movement_pattern(magnitude, direction)
        
        return {
            "magnitude": avg_magnitude,
            "max_magnitude": max_magnitude,
            "direction": np.mean(direction),
            "movement_pixels": movement_pixels,
            "pattern": pattern,
            "is_significant": avg_magnitude > 2.0
        }
    
    def _classify_movement_pattern(self, magnitude: np.ndarray, direction: np.ndarray) -> str:
        """Classifica padrão de movimento"""
        avg_magnitude = np.mean(magnitude)
        
        if avg_magnitude < 0.5:
            return "static"
        elif avg_magnitude < 2.0:
            return "slow"
        elif avg_magnitude < 5.0:
            return "normal"
        elif avg_magnitude < 10.0:
            return "fast"
        else:
            return "erratic"
    
    def detect_anomalous_movement(self) -> Optional[Dict]:
        """Detecta movimento anômalo baseado no histórico"""
        if len(self.movement_history) < 10:
            return None
        
        recent_movements = list(self.movement_history)[-10:]
        magnitudes = [m["magnitude"] for m in recent_movements]
        
        # Detectar picos anômalos
        mean_magnitude = np.mean(magnitudes)
        std_magnitude = np.std(magnitudes)
        
        current_magnitude = magnitudes[-1]
        
        # Anomalia se muito acima da média
        if current_magnitude > mean_magnitude + 2 * std_magnitude:
            return {
                "type": "sudden_movement",
                "severity": min((current_magnitude - mean_magnitude) / std_magnitude, 5.0),
                "confidence": 0.8
            }
        
        # Detectar ausência de movimento prolongada
        if all(m < 0.5 for m in magnitudes[-20:] if len(self.movement_history) >= 20):
            return {
                "type": "no_movement",
                "severity": 3.0,
                "confidence": 0.9
            }
        
        return None

class AlertManager:
    """Gerenciador de alertas com cooldown e filtragem"""
    
    def __init__(self, cooldown_seconds: int = 5):
        self.cooldown_seconds = cooldown_seconds
        self.last_alerts = {}  # tipo -> timestamp
        self.alert_history = deque(maxlen=1000)
        
    def should_alert(self, alert_type: str) -> bool:
        """Verifica se deve emitir alerta baseado no cooldown"""
        current_time = time.time()
        
        if alert_type not in self.last_alerts:
            return True
        
        time_since_last = current_time - self.last_alerts[alert_type]
        return time_since_last >= self.cooldown_seconds
    
    def emit_alert(self, alert_type: str, message: str, confidence: float,
                  additional_data: Dict = None) -> bool:
        """
        Emite alerta se passar pelo filtro de cooldown
        """
        if not self.should_alert(alert_type):
            return False
        
        alert_data = {
            "type": alert_type,
            "message": message,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "additional_data": additional_data or {}
        }
        
        self.alert_history.append(alert_data)
        self.last_alerts[alert_type] = time.time()
        
        # Aqui você pode adicionar diferentes tipos de alertas:
        # - Log
        # - Email
        # - Webhook
        # - Interface visual
        
        return True
    
    def get_recent_alerts(self, minutes: int = 60) -> List[Dict]:
        """Retorna alertas recentes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        recent_alerts = []
        for alert in self.alert_history:
            alert_time = datetime.fromisoformat(alert["timestamp"])
            if alert_time > cutoff_time:
                recent_alerts.append(alert)
        
        return recent_alerts

class FileManager:
    """Gerenciador de arquivos do sistema"""
    
    @staticmethod
    def save_frame(frame: np.ndarray, filename: str, directory: str = "data/alerts/") -> str:
        """Salva frame em arquivo"""
        os.makedirs(directory, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{timestamp}_{filename}"
        full_path = os.path.join(directory, full_filename)
        
        cv2.imwrite(full_path, frame)
        return full_path
    
    @staticmethod
    def cleanup_old_files(directory: str, days_old: int = 7):
        """Remove arquivos antigos"""
        if not os.path.exists(directory):
            return
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict:
        """Obtém informações de um vídeo"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None
        
        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": 0
        }
        
        if info["fps"] > 0:
            info["duration"] = info["frame_count"] / info["fps"]
        
        cap.release()
        return info

class DataAugmentation:
    """Utilitários para augmentação de dados durante treinamento"""
    
    @staticmethod
    def augment_frame(frame: np.ndarray, augment_type: str = "random") -> np.ndarray:
        """Aplica augmentação em um frame"""
        if frame is None:
            return None
        
        augmented = frame.copy()
        
        if augment_type == "random":
            # Escolher augmentação aleatória
            augment_type = np.random.choice([
                "brightness", "contrast", "noise", "blur", "flip"
            ])
        
        if augment_type == "brightness":
            # Ajustar brilho
            factor = np.random.uniform(0.7, 1.3)
            augmented = cv2.convertScaleAbs(augmented, alpha=factor, beta=0)
        
        elif augment_type == "contrast":
            # Ajustar contraste
            factor = np.random.uniform(0.8, 1.2)
            augmented = cv2.convertScaleAbs(augmented, alpha=factor, beta=0)
        
        elif augment_type == "noise":
            # Adicionar ruído gaussiano
            noise = np.random.normal(0, 0.1, augmented.shape).astype(np.float32)
            augmented = np.clip(augmented.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
        
        elif augment_type == "blur":
            # Aplicar blur
            kernel_size = np.random.choice([3, 5])
            augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), 0)
        
        elif augment_type == "flip":
            # Flip horizontal
            if np.random.random() > 0.5:
                augmented = cv2.flip(augmented, 1)
        
        return augmented

class ThreadSafeCounter:
    """Contador thread-safe para IDs de frames"""
    
    def __init__(self, start_value: int = 0):
        self._value = start_value
        self._lock = threading.Lock()
    
    def increment(self) -> int:
        """Incrementa e retorna o novo valor"""
        with self._lock:
            self._value += 1
            return self._value
    
    def get_value(self) -> int:
        """Retorna valor atual"""
        with self._lock:
            return self._value

class ModelUtils:
    """Utilitários para modelos de ML"""
    
    @staticmethod
    def create_sequences(data: np.ndarray, sequence_length: int, 
                        overlap: int = 0) -> np.ndarray:
        """
        Cria sequências para modelos temporais (ConvLSTM)
        
        Args:
            data: Array de frames [num_frames, height, width, channels]
            sequence_length: Comprimento da sequência
            overlap: Sobreposição entre sequências
        """
        if len(data) < sequence_length:
            return np.array([])
        
        step = sequence_length - overlap
        sequences = []
        
        for i in range(0, len(data) - sequence_length + 1, step):
            sequence = data[i:i + sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    @staticmethod
    def calculate_reconstruction_error(original: np.ndarray, 
                                     reconstructed: np.ndarray) -> float:
        """Calcula erro de reconstrução para autoencoder"""
        if original.shape != reconstructed.shape:
            return float('inf')
        
        # MSE (Mean Squared Error)
        mse = np.mean((original - reconstructed) ** 2)
        return float(mse)
    
    @staticmethod
    def normalize_batch(data: np.ndarray, params: Dict = None) -> Tuple[np.ndarray, Dict]:
        """
        Normaliza um batch de dados
        
        Args:
            data: Array de dados para normalizar
            params: Parâmetros de normalização (se None, calcula novos)
        
        Returns:
            Tuple (dados_normalizados, parâmetros)
        """
        if params is None:
            # Calcular novos parâmetros
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std = np.where(std == 0, 1, std)  # Evitar divisão por zero
            
            params = {
                "mean": mean,
                "std": std
            }
        else:
            mean = params["mean"]
            std = params["std"]
        
        # Normalizar
        normalized = (data - mean) / std
        
        return normalized, params
    
    @staticmethod
    def denormalize_batch(normalized_batch: np.ndarray, 
                         params: Dict) -> np.ndarray:
        """Desnormaliza batch usando parâmetros salvos"""
        return normalized_batch * params["std"] + params["mean"]

# Instâncias globais de utilitários
performance_monitor = PerformanceMonitor()
movement_analyzer = MovementAnalyzer()
alert_manager = AlertManager()
frame_counter = ThreadSafeCounter()

def time_function(func):
    """Decorator para medir tempo de execução de funções"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        performance_monitor.update_processing_time(execution_time)
        
        return result
    return wrapper

def create_circular_roi(center: Tuple[int, int], radius: int, 
                       frame_shape: Tuple[int, int]) -> np.ndarray:
    """
    Cria máscara circular para região de interesse
    
    Args:
        center: (x, y) centro do círculo
        radius: raio do círculo
        frame_shape: (height, width) do frame
    
    Returns:
        Máscara binária
    """
    height, width = frame_shape
    y, x = np.ogrid[:height, :width]
    
    center_x, center_y = center
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    return mask.astype(np.uint8) * 255

def calculate_iou(box1: Tuple[int, int, int, int], 
                 box2: Tuple[int, int, int, int]) -> float:
    """
    Calcula Intersection over Union entre duas bounding boxes
    
    Args:
        box1, box2: (x, y, width, height)
    
    Returns:
        IoU score (0.0 to 1.0)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calcular coordenadas de interseção
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    # Área de interseção
    intersection = (xi2 - xi1) * (yi2 - yi1)
    
    # Área de união
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union