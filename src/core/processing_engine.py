"""
Motor de processamento principal - coordena todos os detectores
Implementa pipeline em cascata: Optical Flow → CAE → ConvLSTM
Otimizado para arquitetura multi-thread i5 11Gen
"""

import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
from collections import deque
import cv2

from ..detectors.optical_flow_detector import OpticalFlowDetector
from ..detectors.deep_learning_detector import DeepLearningDetector
from ..utils.helpers import time_function, performance_monitor, movement_analyzer, alert_manager, frame_counter
from ..utils.logger import logger

class AnomalyClassifier:
    """
    Classificador que mapeia anomalias detectadas para tipos específicos
    baseado nas especificações de segurança e saúde
    """
    
    def __init__(self, config):
        self.config = config
        
        # Mapeamento de padrões para tipos de anomalia
        self.anomaly_patterns = {
            "security": {
                "intrusion": {
                    "optical_flow": ["sudden_movement", "large_area_movement"],
                    "spatial": ["high_reconstruction_error"],
                    "temporal": ["movement_increase_pattern"],
                    "confidence_threshold": 0.7
                },
                "break_in": {
                    "optical_flow": ["erratic_movement", "radial_convergence"],
                    "spatial": ["structural_anomaly"],
                    "temporal": ["irregular_sequence"],
                    "confidence_threshold": 0.8
                },
                "night_movement": {
                    "time_condition": "night",
                    "optical_flow": ["any_movement"],
                    "confidence_threshold": 0.6
                },
                "stranger_loitering": {
                    "temporal": ["prolonged_presence"],
                    "spatial": ["person_detection"],
                    "confidence_threshold": 0.7
                },
                "unknown_vehicle": {
                    "spatial": ["vehicle_shape"],
                    "temporal": ["stationary_object"],
                    "confidence_threshold": 0.6
                }
            },
            "health": {
                "fall": {
                    "optical_flow": ["radial_convergence", "sudden_movement"],
                    "temporal": ["fall_pattern"],
                    "confidence_threshold": 0.9
                },
                "collapse": {
                    "optical_flow": ["sudden_stop"],
                    "temporal": ["collapse_pattern"],
                    "confidence_threshold": 0.85
                },
                "immobility": {
                    "optical_flow": ["no_movement"],
                    "temporal": ["prolonged_stillness"],
                    "confidence_threshold": 0.8
                },
                "erratic_movement": {
                    "optical_flow": ["erratic_movement"],
                    "temporal": ["inconsistent_patterns"],
                    "confidence_threshold": 0.7
                },
                "no_movement": {
                    "optical_flow": ["no_movement"],
                    "temporal": ["absence_pattern"],
                    "confidence_threshold": 0.6
                }
            }
        }
    
    def classify_anomaly(self, detection_results: Dict) -> List[Dict]:
        """
        Classifica anomalias baseado nos resultados de detecção
        
        Args:
            detection_results: Resultados dos detectores
            
        Returns:
            Lista de anomalias classificadas
        """
        classified_anomalies = []
        
        # Extrair informações dos detectores
        optical_flow_anomalies = detection_results.get("optical_flow", {}).get("anomalies", [])
        dl_result = detection_results.get("deep_learning", {})
        final_decision = dl_result.get("final_decision", {})
        
        current_time = datetime.now()
        is_night = self._is_night_time(current_time)
        
        # Verificar cada padrão de anomalia
        for category, anomaly_types in self.anomaly_patterns.items():
            for anomaly_type, patterns in anomaly_types.items():
                confidence = self._calculate_pattern_confidence(
                    patterns, optical_flow_anomalies, dl_result, is_night
                )
                
                if confidence >= patterns.get("confidence_threshold", 0.5):
                    classified_anomaly = {
                        "category": category,
                        "type": anomaly_type,
                        "confidence": confidence,
                        "timestamp": current_time.isoformat(),
                        "description": self.config.get_alert_message(category, anomaly_type),
                        "supporting_evidence": self._get_supporting_evidence(
                            patterns, optical_flow_anomalies, dl_result
                        )
                    }
                    classified_anomalies.append(classified_anomaly)
        
        return classified_anomalies
    
    def _calculate_pattern_confidence(self, patterns: Dict, optical_flow_anomalies: List,
                                    dl_result: Dict, is_night: bool) -> float:
        """Calcula confiança baseada nos padrões detectados"""
        confidence_factors = []
        
        # Verificar condições temporais
        if "time_condition" in patterns:
            if patterns["time_condition"] == "night" and is_night:
                confidence_factors.append(0.3)
        
        # Verificar anomalias de optical flow
        if "optical_flow" in patterns:
            flow_patterns = patterns["optical_flow"]
            for anomaly in optical_flow_anomalies:
                if anomaly["type"] in flow_patterns or "any_movement" in flow_patterns:
                    confidence_factors.append(anomaly.get("confidence", 0.5))
        
        # Verificar anomalias espaciais (CAE)
        if "spatial" in patterns:
            cae_result = dl_result.get("cae_result", {})
            if cae_result.get("is_anomaly", False):
                confidence_factors.append(cae_result.get("confidence", 0.0))
        
        # Verificar anomalias temporais (ConvLSTM)
        if "temporal" in patterns:
            convlstm_result = dl_result.get("convlstm_result", {})
            if convlstm_result.get("is_anomaly", False):
                confidence_factors.append(convlstm_result.get("confidence", 0.0))
        
        # Calcular confiança final
        if confidence_factors:
            return min(np.mean(confidence_factors) * 1.2, 1.0)  # Boost na média
        
        return 0.0
    
    def _get_supporting_evidence(self, patterns: Dict, optical_flow_anomalies: List,
                               dl_result: Dict) -> Dict:
        """Coleta evidências que suportam a classificação"""
        evidence = {
            "optical_flow_detections": [],
            "spatial_anomaly": False,
            "temporal_anomaly": False,
            "confidence_breakdown": {}
        }
        
        # Evidências de optical flow
        for anomaly in optical_flow_anomalies:
            if "optical_flow" in patterns:
                if anomaly["type"] in patterns["optical_flow"]:
                    evidence["optical_flow_detections"].append(anomaly)
        
        # Evidências espaciais
        if "spatial" in patterns:
            cae_result = dl_result.get("cae_result", {})
            evidence["spatial_anomaly"] = cae_result.get("is_anomaly", False)
            evidence["confidence_breakdown"]["spatial"] = cae_result.get("confidence", 0.0)
        
        # Evidências temporais
        if "temporal" in patterns:
            convlstm_result = dl_result.get("convlstm_result", {})
            evidence["temporal_anomaly"] = convlstm_result.get("is_anomaly", False)
            evidence["confidence_breakdown"]["temporal"] = convlstm_result.get("confidence", 0.0)
        
        return evidence
    
    def _is_night_time(self, current_time: datetime) -> bool:
        """Determina se é período noturno (18:00 - 06:00)"""
        hour = current_time.hour
        return hour >= 18 or hour <= 6

class ProcessingEngine:
    """
    Motor de processamento principal que coordena todos os detectores
    Implementa pipeline otimizado e classificação inteligente
    """
    
    def __init__(self, config):
        self.config = config
        
        # Inicializar detectores
        self.optical_flow_detector = OpticalFlowDetector(config)
        self.deep_learning_detector = DeepLearningDetector(config)
        self.anomaly_classifier = AnomalyClassifier(config)
        
        # Estado do processamento
        self.is_processing = False
        self.processing_mode = "realtime"  # "realtime" ou "batch"
        
        # Threading para processamento paralelo
        self.processing_threads = []
        self.input_queue = queue.Queue(maxsize=config.system.queue_size)
        self.result_queue = queue.Queue()
        
        # Callbacks para resultados
        self.result_callbacks = []
        self.anomaly_callbacks = []
        
        # Histórico e estatísticas
        self.processing_history = deque(maxlen=1000)
        self.performance_stats = {
            "frames_processed": 0,
            "anomalies_detected": 0,
            "avg_processing_time": 0.0
        }
      # Controle de pipeline adaptativo
        self.adaptive_processing = True
        self.processing_load = 0.0
        self.skip_deep_learning = False
        
        logger.info("ProcessingEngine inicializado")
    
    def start_processing(self, num_threads: int = None) -> bool:
        """
        Inicia processamento em threads paralelas
        
        Args:
            num_threads: Número de threads (usa configuração se None)
        """
        if self.is_processing:
            logger.warning("Processamento já está ativo")
            return True
        
        if num_threads is None:
            num_threads = min(self.config.system.max_threads, 3)  # Máximo 3 para não sobrecarregar
        
        logger.info(f"Iniciando processamento com {num_threads} threads")
        
        self.is_processing = True
        
        # Criar threads de processamento
        for i in range(num_threads):
            thread = threading.Thread(target=self._processing_loop, args=(i,))
            thread.daemon = True
            thread.start()
            self.processing_threads.append(thread)
        
        # Thread para monitoramento de performance
        monitor_thread = threading.Thread(target=self._performance_monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        self.processing_threads.append(monitor_thread)
        
        logger.info("Processamento iniciado")
        return True
    
    def stop_processing(self):
        """Para processamento"""
        if not self.is_processing:
            return
        
        logger.info("Parando processamento...")
        self.is_processing = False
        
        # Aguardar threads terminarem
        for thread in self.processing_threads:
            thread.join(timeout=2.0)
        
        self.processing_threads.clear()
        
        # Limpar queues
        self._clear_queues()
        
        logger.info("Processamento parado")
    
    @time_function
    def process_frame(self, frame_data: Dict) -> Dict:
        """
        Processa um frame através do pipeline completo
        
        Args:
            frame_data: Dict com frame e metadados
            
        Returns:
            Dict com resultados completos da detecção
        """
        start_time = time.time()
        frame = frame_data["frame"]
        frame_id = frame_data.get("frame_id", frame_counter.increment())
        
        results = {
            "frame_id": frame_id,
            "timestamp": frame_data.get("timestamp", time.time()),
            "processing_start": start_time,
            "optical_flow": {},
            "deep_learning": {},
            "classified_anomalies": [],
            "final_assessment": {
                "has_anomaly": False,
                "risk_level": "low",
                "recommended_action": "none"
            }
        }
        
        try:
            # Fase 1: Optical Flow (sempre executado - rápido)
            logger.debug(f"Processando frame {frame_id} - Optical Flow")
            optical_flow_result = self.optical_flow_detector.detect(frame)
            results["optical_flow"] = optical_flow_result
            
            # Atualizar estatísticas
            if optical_flow_result.get("has_movement", False):
                self.performance_stats["optical_flow_detections"] += 1
            
            # Fase 2: Deep Learning (condicional baseado no optical flow)
            should_use_dl = self._should_use_deep_learning(optical_flow_result)
            
            if should_use_dl and not self.skip_deep_learning:
                logger.debug(f"Processando frame {frame_id} - Deep Learning")
                dl_result = self.deep_learning_detector.detect(frame)
                results["deep_learning"] = dl_result
                
                if dl_result.get("final_decision", {}).get("is_anomaly", False):
                    self.performance_stats["deep_learning_detections"] += 1
            else:
                logger.debug(f"Frame {frame_id} - Pulando Deep Learning")
                results["deep_learning"] = {"skipped": True, "reason": "no_movement_detected"}
            
            # Fase 3: Classificação de Anomalias
            if optical_flow_result.get("has_movement", False) or results["deep_learning"].get("final_decision", {}).get("is_anomaly", False):
                logger.debug(f"Processando frame {frame_id} - Classificação")
                classified_anomalies = self.anomaly_classifier.classify_anomaly(results)
                results["classified_anomalies"] = classified_anomalies
                
                if classified_anomalies:
                    self.performance_stats["anomalies_detected"] += 1
            
            # Fase 4: Avaliação Final e Recomendações
            results["final_assessment"] = self._make_final_assessment(results)
            
            # Processar resultados
            self._process_results(results, frame_data)
            
        except Exception as e:
            logger.error(f"Erro no processamento do frame {frame_id}: {str(e)}")
            results["error"] = str(e)
        
        # Atualizar estatísticas de performance
        processing_time = time.time() - start_time
        results["processing_time"] = processing_time
        self._update_performance_stats(processing_time)
        
        # Adicionar ao histórico
        self.processing_history.append(results)
        
        return results
    
    def _should_use_deep_learning(self, optical_flow_result: Dict) -> bool:
        """Decide se deve usar deep learning baseado no optical flow"""
        
        # Sempre usar se detectou movimento significativo
        if optical_flow_result.get("has_movement", False):
            magnitude = optical_flow_result.get("movement_magnitude", 0.0)
            if magnitude > self.config.model.movement_threshold:
                return True
        
        # Usar se detectou anomalias específicas no optical flow
        anomalies = optical_flow_result.get("anomalies", [])
        if anomalies:
            return True
        
        # Controle adaptativo baseado na carga de processamento
        if self.adaptive_processing and self.processing_load > 0.8:
            return False
        
        return False
    
    def _make_final_assessment(self, results: Dict) -> Dict:
        """Cria avaliação final e recomendações"""
        
        classified_anomalies = results.get("classified_anomalies", [])
        optical_flow_anomalies = results.get("optical_flow", {}).get("anomalies", [])
        dl_decision = results.get("deep_learning", {}).get("final_decision", {})
        
        assessment = {
            "has_anomaly": False,
            "risk_level": "low",
            "recommended_action": "none",
            "confidence": 0.0,
            "priority_anomalies": []
        }
        
        if classified_anomalies:
            # Determinar nível de risco baseado nas anomalias classificadas
            max_confidence = max(a["confidence"] for a in classified_anomalies)
            health_anomalies = [a for a in classified_anomalies if a["category"] == "health"]
            security_anomalies = [a for a in classified_anomalies if a["category"] == "security"]
            
            assessment["has_anomaly"] = True
            assessment["confidence"] = max_confidence
            
            # Determinar risco e ações
            if health_anomalies:
                # Anomalias de saúde têm alta prioridade
                fall_or_collapse = any(a["type"] in ["fall", "collapse"] for a in health_anomalies)
                
                if fall_or_collapse:
                    assessment["risk_level"] = "critical"
                    assessment["recommended_action"] = "immediate_response"
                elif any(a["type"] == "immobility" for a in health_anomalies):
                    assessment["risk_level"] = "high"
                    assessment["recommended_action"] = "check_person"
                else:
                    assessment["risk_level"] = "medium"
                    assessment["recommended_action"] = "monitor_closely"
            
            elif security_anomalies:
                # Anomalias de segurança
                intrusion_or_break = any(a["type"] in ["intrusion", "break_in"] for a in security_anomalies)
                
                if intrusion_or_break:
                    assessment["risk_level"] = "high"
                    assessment["recommended_action"] = "security_alert"
                else:
                    assessment["risk_level"] = "medium"
                    assessment["recommended_action"] = "investigate"
            
            # Anomalias prioritárias (high confidence + critical types)
            assessment["priority_anomalies"] = [
                a for a in classified_anomalies 
                if a["confidence"] > 0.8 and a["type"] in ["fall", "collapse", "intrusion", "break_in"]
            ]
        
        elif optical_flow_anomalies or dl_decision.get("is_anomaly", False):
            # Anomalia detectada mas não classificada
            assessment["has_anomaly"] = True
            assessment["risk_level"] = "low"
            assessment["recommended_action"] = "continue_monitoring"
            assessment["confidence"] = dl_decision.get("confidence", 0.3)
        
        return assessment
    
    def _process_results(self, results: Dict, frame_data: Dict):
        """Processa resultados e dispara callbacks/alertas"""
        
        # Chamar callbacks de resultado
        for callback in self.result_callbacks:
            try:
                callback(results, frame_data)
            except Exception as e:
                logger.error(f"Erro em callback de resultado: {e}")
        
        # Processar anomalias
        classified_anomalies = results.get("classified_anomalies", [])
        if classified_anomalies:
            for anomaly in classified_anomalies:
                self._handle_anomaly(anomaly, results, frame_data)
        
        # Adicionar resultados à queue
        try:
            self.result_queue.put_nowait({
                "results": results,
                "frame_data": frame_data
            })
        except queue.Full:
            # Queue cheia - remover resultado mais antigo
            try:
                self.result_queue.get_nowait()
                self.result_queue.put_nowait({
                    "results": results,
                    "frame_data": frame_data
                })
            except queue.Empty:
                pass
    
    def _handle_anomaly(self, anomaly: Dict, results: Dict, frame_data: Dict):
        """Trata anomalia detectada"""
        
        # Log da anomalia
        logger.log_anomaly(
            anomaly["category"],
            anomaly["type"],
            anomaly["confidence"],
            {
                "frame_id": results["frame_id"],
                "timestamp": results["timestamp"]
            },
            anomaly.get("supporting_evidence", {})
        )
        
        # Emitir alerta se passar pelo filtro
        alert_type = f"{anomaly['category']}_{anomaly['type']}"
        if alert_manager.emit_alert(
            alert_type,
            anomaly["description"],
            anomaly["confidence"],
            {
                "frame_id": results["frame_id"],
                "evidence": anomaly.get("supporting_evidence", {}),
                "risk_level": results.get("final_assessment", {}).get("risk_level", "unknown")
            }
        ):
            logger.info(f"Alerta emitido: {alert_type} (confiança: {anomaly['confidence']:.2f})")
            
            # Salvar frame se configurado
            if self.config.system.save_alert_frames:
                try:
                    from ..utils.helpers import FileManager
                    filename = f"alert_{alert_type}_{results['frame_id']}.jpg"
                    FileManager.save_frame(frame_data["frame"], filename, "data/alerts/")
                except Exception as e:
                    logger.error(f"Erro ao salvar frame de alerta: {e}")
        
        # Chamar callbacks de anomalia
        for callback in self.anomaly_callbacks:
            try:
                callback(anomaly, results, frame_data)
            except Exception as e:
                logger.error(f"Erro em callback de anomalia: {e}")
    
    def _processing_loop(self, thread_id: int):
        """Loop principal de processamento (executado em thread)"""
        logger.info(f"Thread de processamento {thread_id} iniciada")
        
        while self.is_processing:
            try:
                # Obter frame da queue
                frame_data = self.input_queue.get(timeout=1.0)
                
                # Processar frame
                results = self.process_frame(frame_data)
                
                # Marcar como processado
                self.input_queue.task_done()
                
            except queue.Empty:
                # Timeout - continuar loop
                continue
            except Exception as e:
                logger.error(f"Erro na thread de processamento {thread_id}: {e}")
                time.sleep(0.1)
        
        logger.info(f"Thread de processamento {thread_id} finalizada")
    
    def _performance_monitor_loop(self):
        """Loop de monitoramento de performance"""
        logger.info("Monitor de performance iniciado")
        
        while self.is_processing:
            try:
                # Calcular carga de processamento
                queue_size = self.input_queue.qsize()
                max_queue_size = self.config.system.queue_size
                self.processing_load = queue_size / max_queue_size
                
                # Controle adaptativo
                if self.adaptive_processing:
                    if self.processing_load > 0.9:
                        self.skip_deep_learning = True
                        logger.debug("Carga alta - pulando deep learning")
                    elif self.processing_load < 0.5:
                        self.skip_deep_learning = False
                
                # Atualizar estatísticas de performance
                performance_monitor.update_memory_usage()
                
                # Log periódico
                if self.performance_stats["frames_processed"] % 100 == 0:
                    logger.debug(f"Performance - Carga: {self.processing_load:.2f}, "
                               f"Frames: {self.performance_stats['frames_processed']}, "
                               f"Anomalias: {self.performance_stats['anomalies_detected']}")
                
                time.sleep(5.0)  # Monitor a cada 5 segundos
                
            except Exception as e:
                logger.error(f"Erro no monitor de performance: {e}")
                time.sleep(1.0)
        
        logger.info("Monitor de performance finalizado")
    
    def add_frame_to_queue(self, frame_data: Dict) -> bool:
        """
        Adiciona frame à queue de processamento
        
        Args:
            frame_data: Dict com frame e metadados
            
        Returns:
            True se adicionado com sucesso
        """
        try:
            self.input_queue.put_nowait(frame_data)
            return True
        except queue.Full:
            logger.warning("Queue de processamento cheia - frame descartado")
            return False
    
    def get_results(self, timeout: float = 1.0) -> Optional[Dict]:
        """Obtém próximo resultado da queue"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def add_result_callback(self, callback: Callable[[Dict, Dict], None]):
        """Adiciona callback para resultados de processamento"""
        self.result_callbacks.append(callback)
        logger.info(f"Callback de resultado adicionado - total: {len(self.result_callbacks)}")
    
    def add_anomaly_callback(self, callback: Callable[[Dict, Dict, Dict], None]):
        """Adiciona callback para anomalias detectadas"""
        self.anomaly_callbacks.append(callback)
        logger.info(f"Callback de anomalia adicionado - total: {len(self.anomaly_callbacks)}")
    
    def _update_performance_stats(self, processing_time: float):
        """Atualiza estatísticas de performance"""
        self.performance_stats["frames_processed"] += 1
        
        # Média móvel do tempo de processamento
        alpha = 0.1
        self.performance_stats["avg_processing_time"] = (
            alpha * processing_time + 
            (1 - alpha) * self.performance_stats["avg_processing_time"]
        )
    
    def _clear_queues(self):
        """Limpa todas as queues"""
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
    
    def get_performance_stats(self) -> Dict:
        """Retorna estatísticas de performance"""
        stats = self.performance_stats.copy()
        
        # Adicionar estatísticas dos detectores
        stats.update({
            "optical_flow_stats": self.optical_flow_detector.get_stats(),
            "deep_learning_info": self.deep_learning_detector.get_model_info(),
            "processing_load": self.processing_load,
            "input_queue_size": self.input_queue.qsize(),
            "result_queue_size": self.result_queue.qsize(),
            "is_processing": self.is_processing,
            "adaptive_processing": self.adaptive_processing,
            "skip_deep_learning": self.skip_deep_learning
        })
        
        return stats
    
    def start_training_mode(self):
        """Inicia modo de treinamento"""
        self.deep_learning_detector.start_training_mode()
        logger.info("Modo de treinamento iniciado no ProcessingEngine")
    
    def stop_training_mode(self):
        """Para modo de treinamento"""
        self.deep_learning_detector.stop_training_mode()
        logger.info("Modo de treinamento parado no ProcessingEngine")
    
    def train_models(self, external_data: Optional[np.ndarray] = None, 
                    epochs_cae: int = 50, epochs_convlstm: int = 30) -> Dict:
        """Treina modelos de deep learning"""
        return self.deep_learning_detector.train_models(external_data, epochs_cae, epochs_convlstm)
    
    def save_models(self, base_path: str = None):
        """Salva modelos treinados"""
        self.deep_learning_detector.save_models(base_path)
    
    def load_models(self, base_path: str = None) -> bool:
        """Carrega modelos treinados"""
        return self.deep_learning_detector.load_models(base_path)
    
    def reset_detectors(self):
        """Reset de todos os detectores"""
        self.optical_flow_detector.reset()
        # Deep learning detector não precisa reset (mantém modelos carregados)
        logger.info("Detectores resetados")
    
    def cleanup(self):
        """Limpeza completa do engine"""
        logger.info("Iniciando limpeza do ProcessingEngine")
        
        # Parar processamento
        self.stop_processing()
        
        # Limpar callbacks
        self.result_callbacks.clear()
        self.anomaly_callbacks.clear()
        
        # Limpar histórico
        self.processing_history.clear()
        
        # Reset detectores
        self.reset_detectors()
        
        logger.info("ProcessingEngine limpo")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()