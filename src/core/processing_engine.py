"""
Motor de Processamento Principal
Coordena todos os detectores em pipeline de cascata otimizado
"""

import threading
import time
import queue
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import json
from datetime import datetime

from ..detectors.optical_flow_detector import OpticalFlowDetector
from ..detectors.deep_learning_detector import DeepLearningDetector
from ..utils.logger import logger
from ..utils.helpers import calculate_metrics, create_visualization

@dataclass
class ProcessingResult:
    """Resultado do processamento de um frame"""
    frame_id: int
    timestamp: datetime
    optical_flow_score: float
    cae_score: float
    convlstm_score: float
    final_anomaly_score: float
    is_anomaly: bool
    anomaly_type: str
    processing_time: float
    frame_shape: Tuple[int, int, int]

class PerformanceMetrics:
    """Classe para coletar métricas de performance"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.anomaly_scores = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)
        
        # Contadores
        self.total_frames = 0
        self.anomalies_detected = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # Timestamps
        self.start_time = time.time()
        self.last_frame_time = time.time()
    
    def update(self, result: ProcessingResult, system_stats: Dict):
        """Atualiza métricas com novo resultado"""
        current_time = time.time()
        
        # FPS
        fps = 1.0 / (current_time - self.last_frame_time)
        self.fps_history.append(fps)
        self.last_frame_time = current_time
        
        # Tempos de processamento
        self.processing_times.append(result.processing_time)
        
        # Scores de anomalia
        self.anomaly_scores.append(result.final_anomaly_score)
        
        # Métricas do sistema
        self.memory_usage.append(system_stats.get('memory_percent', 0))
        self.cpu_usage.append(system_stats.get('cpu_percent', 0))
        
        # Contadores
        self.total_frames += 1
        if result.is_anomaly:
            self.anomalies_detected += 1
    
    def get_current_metrics(self) -> Dict:
        """Retorna métricas atuais"""
        return {
            'fps': {
                'current': self.fps_history[-1] if self.fps_history else 0,
                'average': np.mean(self.fps_history) if self.fps_history else 0,
                'min': np.min(self.fps_history) if self.fps_history else 0,
                'max': np.max(self.fps_history) if self.fps_history else 0
            },
            'processing_time': {
                'current': self.processing_times[-1] if self.processing_times else 0,
                'average': np.mean(self.processing_times) if self.processing_times else 0,
                'p95': np.percentile(self.processing_times, 95) if self.processing_times else 0
            },
            'anomaly_detection': {
                'total_frames': self.total_frames,
                'anomalies_detected': self.anomalies_detected,
                'anomaly_rate': self.anomalies_detected / max(self.total_frames, 1),
                'current_score': self.anomaly_scores[-1] if self.anomaly_scores else 0,
                'average_score': np.mean(self.anomaly_scores) if self.anomaly_scores else 0
            },
            'system': {
                'uptime': time.time() - self.start_time,
                'memory_usage': self.memory_usage[-1] if self.memory_usage else 0,
                'cpu_usage': self.cpu_usage[-1] if self.cpu_usage else 0,
                'avg_memory': np.mean(self.memory_usage) if self.memory_usage else 0,
                'avg_cpu': np.mean(self.cpu_usage) if self.cpu_usage else 0
            }
        }

class ProcessingEngine:
    """
    Motor principal de processamento
    Implementa pipeline em cascata: Optical Flow → CAE → ConvLSTM
    """
    
    def __init__(self, config):
        """Inicializa o motor de processamento"""
        self.config = config
        
        # Estado do sistema
        self.running = False
        self.paused = False
        
        # Filas thread-safe
        self.input_queue = queue.Queue(maxsize=config.get('processing.queue_size', 10))
        self.result_queue = queue.Queue()
        
        # Detectores (inicialização lazy)
        self.optical_flow_detector = None
        self.deep_learning_detector = None
        
        # Threads de processamento
        self.worker_threads = []
        self.num_workers = config.get('processing.num_workers', 2)
        
        # Callbacks
        self.frame_callbacks = []
        self.result_callbacks = []
        self.anomaly_callbacks = []
        
        # Métricas e estatísticas
        self.metrics = PerformanceMetrics()
        self.frame_counter = 0
        
        # Buffer para ConvLSTM (sequências temporais)
        self.frame_sequence_buffer = deque(maxlen=config.get('convlstm.sequence_length', 10))
        
        # Thresholds de detecção
        self.optical_flow_threshold = config.get('thresholds.optical_flow', 0.3)
        self.cae_threshold = config.get('thresholds.cae', 0.5)
        self.convlstm_threshold = config.get('thresholds.convlstm', 0.6)
        
        # Sistema de cooldown para alertas
        self.last_alert_time = {}
        self.alert_cooldown = config.get('alerts.cooldown_seconds', 5)
        
        logger.info("ProcessingEngine inicializado")
    
    def initialize_detectors(self):
        """Inicializa detectores (lazy loading)"""
        if not self.optical_flow_detector:
            self.optical_flow_detector = OpticalFlowDetector(self.config)
            logger.info("OpticalFlowDetector inicializado")
        
        if not self.deep_learning_detector:
            self.deep_learning_detector = DeepLearningDetector(self.config)
            logger.info("DeepLearningDetector inicializado")
    
    def start(self):
        """Inicia o motor de processamento"""
        if self.running:
            logger.warning("ProcessingEngine já está em execução")
            return
        
        self.running = True
        self.paused = False
        
        # Inicializar detectores
        self.initialize_detectors()
        
        # Criar threads de trabalho
        for i in range(self.num_workers):
            thread = threading.Thread(target=self._worker_loop, name=f"ProcessingWorker-{i}")
            thread.daemon = True
            thread.start()
            self.worker_threads.append(thread)
        
        # Thread para métricas de sistema
        metrics_thread = threading.Thread(target=self._metrics_loop, name="MetricsCollector")
        metrics_thread.daemon = True
        metrics_thread.start()
        self.worker_threads.append(metrics_thread)
        
        logger.info(f"ProcessingEngine iniciado com {self.num_workers} workers")
    
    def stop(self):
        """Para o motor de processamento"""
        self.running = False
        
        # Sinalizar parada para todas as threads
        for _ in range(self.num_workers):
            try:
                self.input_queue.put(None, timeout=1)
            except queue.Full:
                pass
        
        # Aguardar threads terminarem
        for thread in self.worker_threads:
            thread.join(timeout=5)
        
        self.worker_threads.clear()
        logger.info("ProcessingEngine parado")
    
    def pause(self):
        """Pausa o processamento"""
        self.paused = True
        logger.info("ProcessingEngine pausado")
    
    def resume(self):
        """Resume o processamento"""
        self.paused = False
        logger.info("ProcessingEngine resumido")
    
    def add_frame_to_queue(self, frame_data: Dict):
        """Adiciona frame à fila de processamento"""
        if not self.running:
            return False
        
        try:
            self.input_queue.put(frame_data, timeout=0.1)
            return True
        except queue.Full:
            logger.warning("Fila de processamento cheia, descartando frame")
            return False
    
    def _worker_loop(self):
        """Loop principal de processamento em thread separada"""
        logger.info(f"Worker thread {threading.current_thread().name} iniciada")
        
        while self.running:
            try:
                # Buscar próximo frame
                frame_data = self.input_queue.get(timeout=1)
                
                if frame_data is None:  # Sinal de parada
                    break
                
                if self.paused:
                    continue
                
                # Processar frame
                result = self._process_frame(frame_data)
                
                if result:
                    # Enviar resultado
                    self.result_queue.put(result)
                    
                    # Executar callbacks
                    self._execute_callbacks(result, frame_data)
                
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erro no worker: {e}")
        
        logger.info(f"Worker thread {threading.current_thread().name} finalizada")
    
    def _process_frame(self, frame_data: Dict) -> Optional[ProcessingResult]:
        """Processa um frame usando pipeline em cascata"""
        start_time = time.time()
        
        try:
            frame = frame_data['frame']
            frame_id = frame_data.get('frame_id', self.frame_counter)
            timestamp = frame_data.get('timestamp', datetime.now())
            
            self.frame_counter += 1
            
            # Adicionar frame ao buffer de sequência
            self.frame_sequence_buffer.append(frame)
            
            # ETAPA 1: Optical Flow (sempre executado)
            optical_flow_score = self.optical_flow_detector.detect(frame)
            
            # ETAPA 2: CAE (só se movimento detectado)
            cae_score = 0.0
            if optical_flow_score > self.optical_flow_threshold:
                cae_score = self.deep_learning_detector.detect_cae(frame)
            
            # ETAPA 3: ConvLSTM (só se CAE detectou anomalia)
            convlstm_score = 0.0
            if cae_score > self.cae_threshold and len(self.frame_sequence_buffer) >= 5:
                sequence = list(self.frame_sequence_buffer)[-10:]  # Últimos 10 frames
                convlstm_score = self.deep_learning_detector.detect_convlstm(sequence)
            
            # Calcular score final
            final_score = self._calculate_final_score(
                optical_flow_score, cae_score, convlstm_score
            )
            
            # Determinar se é anomalia
            is_anomaly = final_score > self.convlstm_threshold
            anomaly_type = self._classify_anomaly_type(
                optical_flow_score, cae_score, convlstm_score, frame
            )
            
            processing_time = time.time() - start_time
            
            # Criar resultado
            result = ProcessingResult(
                frame_id=frame_id,
                timestamp=timestamp,
                optical_flow_score=optical_flow_score,
                cae_score=cae_score,
                convlstm_score=convlstm_score,
                final_anomaly_score=final_score,
                is_anomaly=is_anomaly,
                anomaly_type=anomaly_type,
                processing_time=processing_time,
                frame_shape=frame.shape
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erro no processamento do frame {frame_id}: {e}")
            return None
    
    def _calculate_final_score(self, optical_flow: float, cae: float, convlstm: float) -> float:
        """Calcula score final combinando os três detectores"""
        # Pesos para cada detector
        w_optical = self.config.get('weights.optical_flow', 0.2)
        w_cae = self.config.get('weights.cae', 0.4)
        w_convlstm = self.config.get('weights.convlstm', 0.4)
        
        # Score ponderado
        final_score = (w_optical * optical_flow + 
                      w_cae * cae + 
                      w_convlstm * convlstm)
        
        return min(final_score, 1.0)  # Limitar a 1.0
    
    def _classify_anomaly_type(self, optical_flow: float, cae: float, 
                              convlstm: float, frame: np.ndarray) -> str:
        """Classifica o tipo de anomalia detectada"""
        
        if convlstm > 0.8:
            return "anomalia_temporal_crítica"
        elif cae > 0.7:
            return "anomalia_espacial"
        elif optical_flow > 0.6:
            return "movimento_suspeito"
        elif optical_flow < 0.1 and cae > 0.4:
            return "objeto_estático_anômalo"
        else:
            return "anomalia_geral"
    
    def _execute_callbacks(self, result: ProcessingResult, frame_data: Dict):
        """Executa callbacks registrados"""
        try:
            # Callbacks de resultado (sempre executados)
            for callback in self.result_callbacks:
                callback(result, frame_data)
            
            # Callbacks de anomalia (só quando detectada)
            if result.is_anomaly:
                # Verificar cooldown
                current_time = time.time()
                last_alert = self.last_alert_time.get(result.anomaly_type, 0)
                
                if current_time - last_alert >= self.alert_cooldown:
                    self.last_alert_time[result.anomaly_type] = current_time
                    
                    for callback in self.anomaly_callbacks:
                        callback(result, frame_data)
        
        except Exception as e:
            logger.error(f"Erro ao executar callbacks: {e}")
    
    def _metrics_loop(self):
        """Thread para coleta de métricas do sistema"""
        import psutil
        
        while self.running:
            try:
                # Coletar métricas do sistema
                system_stats = {
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage': psutil.disk_usage('/').percent
                }
                
                # Se houver resultado recente, atualizar métricas
                try:
                    result = self.result_queue.get_nowait()
                    self.metrics.update(result, system_stats)
                except queue.Empty:
                    pass
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Erro na coleta de métricas: {e}")
                time.sleep(5)
    
    def get_metrics(self) -> Dict:
        """Retorna métricas atuais do sistema"""
        return self.metrics.get_current_metrics()
    
    def get_performance_data(self, last_n_minutes: int = 10) -> Dict:
        """Retorna dados para gráficos de performance"""
        try:
            # Dados dos últimos N minutos
            window_size = min(last_n_minutes * 60, len(self.metrics.fps_history))
            
            if window_size == 0:
                return {}
            
            # Preparar dados para gráficos
            timestamps = [time.time() - i for i in range(window_size, 0, -1)]
            
            performance_data = {
                'timestamps': timestamps,
                'fps': list(self.metrics.fps_history)[-window_size:],
                'processing_times': list(self.metrics.processing_times)[-window_size:],
                'anomaly_scores': list(self.metrics.anomaly_scores)[-window_size:],
                'memory_usage': list(self.metrics.memory_usage)[-window_size:],
                'cpu_usage': list(self.metrics.cpu_usage)[-window_size:],
                'summary': self.get_metrics()
            }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Erro ao obter dados de performance: {e}")
            return {}
    
    def export_metrics_to_json(self, filepath: str):
        """Exporta métricas para arquivo JSON"""
        try:
            metrics_data = {
                'export_timestamp': datetime.now().isoformat(),
                'system_config': self.config.get_all(),
                'performance_metrics': self.get_metrics(),
                'detailed_data': self.get_performance_data(60)  # Última hora
            }
            
            with open(filepath, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            logger.info(f"Métricas exportadas para {filepath}")
            
        except Exception as e:
            logger.error(f"Erro ao exportar métricas: {e}")
    
    # Métodos para registrar callbacks
    def add_result_callback(self, callback: Callable):
        """Adiciona callback para resultados"""
        self.result_callbacks.append(callback)
    
    def add_anomaly_callback(self, callback: Callable):
        """Adiciona callback para anomalias"""
        self.anomaly_callbacks.append(callback)