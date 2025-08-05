"""
Processing Engine - Motor de Processamento Principal
Sistema de Detecção de Anomalias em Tempo Real
"""

import threading
import time
import queue
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import Logger
from src.utils.config import Config
from src.detectors.optical_flow_detector import OpticalFlowDetector
from src.detectors.deep_learning_detector import DeepLearningDetector

class ProcessingState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    TRAINING = "training"
    STOPPING = "stopping"

@dataclass
class ProcessingResult:
    """Resultado do processamento de um frame"""
    frame_id: int
    timestamp: float
    anomaly_detected: bool
    anomaly_type: str
    confidence: float
    optical_flow_score: float
    deep_learning_score: float
    metadata: Dict[str, Any]

class ProcessingEngine:
    """
    Motor de processamento principal que coordena todos os detectores
    Implementa pipeline em cascata otimizado para hardware i5 11Gen
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger("ProcessingEngine")
        
        # Estado do sistema
        self.state = ProcessingState.IDLE
        self.is_running = False
        self.is_paused = False
        
        # Filas de processamento
        self.input_queue = queue.Queue(maxsize=30)
        self.output_queue = queue.Queue(maxsize=50)
        
        # Detectores
        self.optical_flow_detector = None
        self.deep_learning_detector = None
        
        # Threading
        self.processing_thread = None
        self.frame_counter = 0
        self.total_frames_processed = 0
        
        # Callbacks para resultados
        self.result_callbacks: List[Callable] = []
        
        # Estatísticas de performance
        self.stats = {
            'frames_processed': 0,
            'anomalies_detected': 0,
            'avg_processing_time': 0.0,
            'fps': 0.0,
            'optical_flow_detections': 0,
            'deep_learning_detections': 0,
            'false_positives': 0,
            'processing_times': []
        }
        
        # Buffer para análise temporal
        self.frame_buffer = []
        self.max_buffer_size = config.get('processing.frame_buffer_size', 10)
        
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Inicializa os detectores com configurações otimizadas"""
        try:
            # Optical Flow Detector (rápido - primeira camada)
            self.optical_flow_detector = OpticalFlowDetector(self.config)
            self.logger.info("OpticalFlowDetector inicializado")
            
            # Deep Learning Detector (lento - segunda camada)
            if self.config.get('processing.use_deep_learning', True):
                self.deep_learning_detector = DeepLearningDetector(self.config)
                self.logger.info("DeepLearningDetector inicializado")
            
            self.logger.info("Todos os detectores inicializados com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao inicializar detectores: {e}")
            raise
    
    def start(self):
        """Inicia o motor de processamento"""
        if self.is_running:
            self.logger.warning("Processing Engine já está rodando")
            return
        
        self.is_running = True
        self.state = ProcessingState.RUNNING
        
        # Thread principal de processamento
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="ProcessingEngine"
        )
        self.processing_thread.start()
        
        self.logger.info("Processing Engine iniciado")
    
    def stop(self):
        """Para o motor de processamento"""
        if not self.is_running:
            return
        
        self.state = ProcessingState.STOPPING
        self.is_running = False
        
        # Aguarda thread terminar
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        # Limpa filas
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        self.state = ProcessingState.IDLE
        self.logger.info("Processing Engine parado")
    
    def pause(self):
        """Pausa o processamento"""
        self.is_paused = True
        self.state = ProcessingState.PAUSED
        self.logger.info("Processing Engine pausado")
    
    def resume(self):
        """Retoma o processamento"""
        self.is_paused = False
        self.state = ProcessingState.RUNNING
        self.logger.info("Processing Engine retomado")
    
    def process_frame(self, frame: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        Adiciona frame para processamento
        
        Args:
            frame: Frame de vídeo
            metadata: Metadados do frame
            
        Returns:
            True se frame foi adicionado à fila
        """
        if not self.is_running:
            return False
        
        try:
            frame_data = {
                'frame': frame.copy(),
                'metadata': metadata or {},
                'timestamp': time.time(),
                'frame_id': self.frame_counter
            }
            
            # Adiciona à fila (não-bloqueante)
            self.input_queue.put_nowait(frame_data)
            self.frame_counter += 1
            return True
            
        except queue.Full:
            self.logger.warning("Fila de entrada cheia - descartando frame")
            return False
    
    def _processing_loop(self):
        """Loop principal de processamento"""
        self.logger.info("Iniciando loop de processamento")
        
        while self.is_running:
            try:
                # Verifica pausa
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Pega próximo frame
                try:
                    frame_data = self.input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Processa frame
                start_time = time.time()
                result = self._process_single_frame(frame_data)
                processing_time = time.time() - start_time
                
                # Atualiza estatísticas
                self._update_stats(processing_time, result)
                
                # Envia resultado
                if result:
                    self._send_result(result)
                
                # Marca como processado
                self.input_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Erro no loop de processamento: {e}")
                time.sleep(0.1)
        
        self.logger.info("Loop de processamento finalizado")
    
    def _process_single_frame(self, frame_data: Dict) -> Optional[ProcessingResult]:
        """
        Processa um único frame através do pipeline em cascata
        
        Pipeline:
        1. Optical Flow (rápido) - filtra frames sem movimento
        2. Deep Learning (lento) - só se movimento detectado
        """
        frame = frame_data['frame']
        metadata = frame_data['metadata']
        timestamp = frame_data['timestamp']
        frame_id = frame_data['frame_id']
        
        try:
            # ESTÁGIO 1: Optical Flow (sempre executa)
            optical_flow_result = self.optical_flow_detector.detect(frame)
            optical_flow_score = optical_flow_result.get('score', 0.0)
            
            # Verifica se há movimento significativo
            movement_threshold = self.config.get('processing.movement_threshold', 0.3)
            has_movement = optical_flow_score > movement_threshold
            
            anomaly_detected = False
            anomaly_type = "normal"
            confidence = 0.0
            deep_learning_score = 0.0
            
            # ESTÁGIO 2: Deep Learning (só se movimento detectado)
            if has_movement and self.deep_learning_detector:
                # Adiciona frame ao buffer temporal
                self._update_frame_buffer(frame)
                
                # Se buffer está cheio, faz análise com ConvLSTM
                if len(self.frame_buffer) >= self.max_buffer_size:
                    dl_result = self.deep_learning_detector.detect_sequence(
                        self.frame_buffer.copy()
                    )
                    
                    if dl_result['anomaly_detected']:
                        anomaly_detected = True
                        anomaly_type = dl_result['anomaly_type']
                        confidence = dl_result['confidence']
                        deep_learning_score = dl_result['score']
                
                # Análise frame único com CAE (sempre)
                cae_result = self.deep_learning_detector.detect_frame(frame)
                if cae_result['anomaly_detected'] and confidence < cae_result['confidence']:
                    anomaly_detected = True
                    anomaly_type = cae_result['anomaly_type']
                    confidence = cae_result['confidence']
                    deep_learning_score = max(deep_learning_score, cae_result['score'])
            
            # Cria resultado
            result = ProcessingResult(
                frame_id=frame_id,
                timestamp=timestamp,
                anomaly_detected=anomaly_detected,
                anomaly_type=anomaly_type,
                confidence=confidence,
                optical_flow_score=optical_flow_score,
                deep_learning_score=deep_learning_score,
                metadata={
                    **metadata,
                    'has_movement': has_movement,
                    'processing_stage': 'deep_learning' if has_movement else 'optical_flow',
                    'frame_shape': frame.shape
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao processar frame {frame_id}: {e}")
            return None
    
    def _update_frame_buffer(self, frame: np.ndarray):
        """Atualiza buffer circular de frames para análise temporal"""
        # Redimensiona frame para o tamanho padrão
        target_size = self.config.get('processing.frame_size', (64, 64))
        import cv2
        resized_frame = cv2.resize(frame, target_size)
        
        # Adiciona ao buffer
        self.frame_buffer.append(resized_frame)
        
        # Mantém tamanho máximo
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
    
    def _update_stats(self, processing_time: float, result: Optional[ProcessingResult]):
        """Atualiza estatísticas de performance"""
        self.stats['frames_processed'] += 1
        self.stats['processing_times'].append(processing_time)
        
        # Mantém apenas últimas 100 medições
        if len(self.stats['processing_times']) > 100:
            self.stats['processing_times'].pop(0)
        
        # Calcula médias
        self.stats['avg_processing_time'] = np.mean(self.stats['processing_times'])
        self.stats['fps'] = 1.0 / self.stats['avg_processing_time'] if self.stats['avg_processing_time'] > 0 else 0
        
        # Conta anomalias
        if result and result.anomaly_detected:
            self.stats['anomalies_detected'] += 1
            
            if result.optical_flow_score > result.deep_learning_score:
                self.stats['optical_flow_detections'] += 1
            else:
                self.stats['deep_learning_detections'] += 1
    
    def _send_result(self, result: ProcessingResult):
        """Envia resultado para callbacks registrados"""
        try:
            # Adiciona à fila de saída
            self.output_queue.put_nowait(result)
            
            # Chama callbacks
            for callback in self.result_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    self.logger.error(f"Erro em callback: {e}")
                    
        except queue.Full:
            self.logger.warning("Fila de saída cheia - descartando resultado")
    
    def get_result(self, timeout: float = 0.1) -> Optional[ProcessingResult]:
        """Pega próximo resultado da fila"""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def add_result_callback(self, callback: Callable[[ProcessingResult], None]):
        """Adiciona callback para receber resultados"""
        self.result_callbacks.append(callback)
        self.logger.info(f"Callback adicionado - total: {len(self.result_callbacks)}")
    
    def remove_result_callback(self, callback: Callable):
        """Remove callback"""
        if callback in self.result_callbacks:
            self.result_callbacks.remove(callback)
            self.logger.info(f"Callback removido - total: {len(self.result_callbacks)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas atuais"""
        return {
            **self.stats,
            'state': self.state.value,
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'queue_sizes': {
                'input': self.input_queue.qsize(),
                'output': self.output_queue.qsize()
            },
            'frame_buffer_size': len(self.frame_buffer)
        }
    
    def reset_stats(self):
        """Reseta estatísticas"""
        self.stats = {
            'frames_processed': 0,
            'anomalies_detected': 0,
            'avg_processing_time': 0.0,
            'fps': 0.0,
            'optical_flow_detections': 0,
            'deep_learning_detections': 0,
            'false_positives': 0,
            'processing_times': []
        }
        self.logger.info("Estatísticas resetadas")
    
    def set_training_mode(self, enabled: bool):
        """Ativa/desativa modo de treinamento"""
        if enabled:
            self.state = ProcessingState.TRAINING
            if self.deep_learning_detector:
                self.deep_learning_detector.set_training_mode(True)
        else:
            self.state = ProcessingState.RUNNING if self.is_running else ProcessingState.IDLE
            if self.deep_learning_detector:
                self.deep_learning_detector.set_training_mode(False)
        
        self.logger.info(f"Modo de treinamento: {'ativado' if enabled else 'desativado'}")
    
    def save_models(self, models_dir: str):
        """Salva modelos treinados"""
        if self.deep_learning_detector:
            self.deep_learning_detector.save_models(models_dir)
            self.logger.info(f"Modelos salvos em: {models_dir}")
    
    def load_models(self, models_dir: str):
        """Carrega modelos salvos"""
        if self.deep_learning_detector:
            self.deep_learning_detector.load_models(models_dir)
            self.logger.info(f"Modelos carregados de: {models_dir}")
    
    def cleanup(self):
        """Limpeza de recursos"""
        self.logger.info("Iniciando limpeza do ProcessingEngine")
        
        # Para processamento
        self.stop()
        
        # Reset detectores
        if self.optical_flow_detector:
            self.optical_flow_detector.reset()
            self.logger.info("OpticalFlowDetector resetado")
        
        if self.deep_learning_detector:
            self.deep_learning_detector.cleanup()
            self.logger.info("DeepLearningDetector limpo")
        
        # Limpa buffers
        self.frame_buffer.clear()
        self.result_callbacks.clear()
        
        self.logger.info("Detectores resetados")
        self.logger.info("ProcessingEngine limpo")
    
    def __del__(self):
        """Destrutor - garante limpeza"""
        try:
            self.cleanup()
        except:
            pass