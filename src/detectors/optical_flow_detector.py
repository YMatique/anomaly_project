"""
Detector de Optical Flow para detecção rápida de movimento
Primeira camada de filtragem antes dos modelos deep learning
Otimizado para i5 11Gen
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from collections import deque

from ..utils.helpers import time_function, performance_monitor, movement_analyzer
from ..utils.logger import logger

class OpticalFlowDetector:
    """
    Detector de Optical Flow otimizado para detecção de anomalias
    Usa Lucas-Kanade e Farneback para diferentes tipos de análise
    """
    
    def __init__(self, config):
        self.config = config
        self.method = config.model.optical_flow_method
        self.threshold = config.model.flow_threshold
        
        # Histórico de frames para análise temporal
        self.frame_history = deque(maxlen=10)
        self.flow_history = deque(maxlen=5)
        
        # Parâmetros para Lucas-Kanade
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Parâmetros para Farneback
        self.farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Pontos para tracking (Lucas-Kanade)
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        self.previous_frame = None
        self.tracking_points = None
        
        # Estatísticas
        self.stats = {
            "total_detections": 0,
            "movement_detections": 0,
            "avg_processing_time": 0.0
        }
        
        logger.info(f"OpticalFlowDetector inicializado - método: {self.method}")
    
    @time_function
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detecta movimento usando optical flow
        
        Args:
            frame: Frame atual (BGR)
            
        Returns:
            Dict com informações de movimento e possíveis anomalias
        """
        if frame is None:
            return self._empty_result()
        
        start_time = time.time()
        
        # Converter para grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Primeiro frame - inicializar
        if self.previous_frame is None:
            self.previous_frame = gray_frame
            return self._empty_result()
        
        # Calcular optical flow baseado no método
        if self.method == "lucas_kanade":
            result = self._lucas_kanade_flow(gray_frame, frame)
        else:  # farneback
            result = self._farneback_flow(gray_frame, frame)
        
        # Adicionar ao histórico
        self.frame_history.append({
            "timestamp": time.time(),
            "has_movement": result["has_movement"],
            "movement_magnitude": result["movement_magnitude"]
        })
        
        # Análise temporal
        temporal_analysis = self._analyze_temporal_patterns()
        result.update(temporal_analysis)
        
        # Atualizar frame anterior
        self.previous_frame = gray_frame
        
        # Atualizar estatísticas
        processing_time = time.time() - start_time
        self._update_stats(processing_time, result["has_movement"])
        
        return result
    
    def _lucas_kanade_flow(self, gray_frame: np.ndarray, color_frame: np.ndarray) -> Dict:
        """Optical flow usando Lucas-Kanade"""
        
        # Detectar pontos para tracking se necessário
        if self.tracking_points is None or len(self.tracking_points) < 50:
            corners = cv2.goodFeaturesToTrack(
                self.previous_frame, 
                **self.feature_params
            )
            if corners is not None:
                self.tracking_points = corners
        
        if self.tracking_points is None:
            return self._empty_result()
        
        # Calcular optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.previous_frame,
            gray_frame,
            self.tracking_points,
            None,
            **self.lk_params
        )
        
        # Filtrar pontos válidos
        if new_points is not None and status is not None:
            good_new = new_points[status == 1]
            good_old = self.tracking_points[status == 1]
            
            if len(good_new) > 0:
                # Calcular magnitude do movimento
                movement_vectors = good_new - good_old
                magnitudes = np.sqrt(movement_vectors[:, 0]**2 + movement_vectors[:, 1]**2)
                
                avg_magnitude = np.mean(magnitudes)
                max_magnitude = np.max(magnitudes)
                moving_points = np.sum(magnitudes > self.threshold)
                
                # Atualizar pontos de tracking
                self.tracking_points = good_new.reshape(-1, 1, 2)
                
                # Criar resultado
                result = {
                    "method": "lucas_kanade",
                    "movement_magnitude": float(avg_magnitude),
                    "max_magnitude": float(max_magnitude),
                    "moving_points": int(moving_points),
                    "total_points": len(good_new),
                    "has_movement": avg_magnitude > self.threshold,
                    "movement_vectors": movement_vectors,
                    "tracking_points": good_new,
                    "flow_field": None
                }
                
                # Detectar anomalias específicas
                anomalies = self._detect_movement_anomalies(magnitudes, movement_vectors)
                result["anomalies"] = anomalies
                
                return result
        
        # Reset pontos se perdeu tracking
        self.tracking_points = None
        return self._empty_result()
    
    def _farneback_flow(self, gray_frame: np.ndarray, color_frame: np.ndarray) -> Dict:
        """Optical flow usando Farneback (dense flow)"""
        
        # Calcular dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.previous_frame,
            gray_frame,
            None,
            **self.farneback_params
        )
        
        # Calcular magnitude e direção
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        direction = np.arctan2(flow[..., 1], flow[..., 0])
        
        # Estatísticas de movimento
        avg_magnitude = np.mean(magnitude)
        max_magnitude = np.max(magnitude)
        movement_pixels = np.sum(magnitude > self.threshold)
        total_pixels = magnitude.size
        
        # Adicionar ao histórico de flow
        self.flow_history.append(flow)
        
        result = {
            "method": "farneback",
            "movement_magnitude": float(avg_magnitude),
            "max_magnitude": float(max_magnitude),
            "movement_pixels": int(movement_pixels),
            "total_pixels": int(total_pixels),
            "movement_ratio": float(movement_pixels / total_pixels),
            "has_movement": avg_magnitude > self.threshold,
            "flow_field": flow,
            "magnitude_field": magnitude,
            "direction_field": direction
        }
        
        # Análise de padrões de movimento
        pattern_analysis = self._analyze_movement_patterns(magnitude, direction, flow)
        result.update(pattern_analysis)
        
        # Detectar anomalias específicas
        anomalies = self._detect_dense_flow_anomalies(magnitude, direction, flow)
        result["anomalies"] = anomalies
        
        return result
    
    def _analyze_movement_patterns(self, magnitude: np.ndarray, 
                                 direction: np.ndarray, flow: np.ndarray) -> Dict:
        """Analisa padrões específicos no movimento"""
        
        patterns = {
            "dominant_direction": float(np.mean(direction)),
            "direction_consistency": float(np.std(direction)),
            "movement_coherence": 0.0,
            "spatial_distribution": "uniform"
        }
        
        # Calcular coerência do movimento
        if np.mean(magnitude) > 0:
            # Calcular divergência do campo de fluxo
            fx = flow[..., 0]
            fy = flow[..., 1]
            
            # Gradientes para divergência
            fx_x = np.gradient(fx, axis=1)
            fy_y = np.gradient(fy, axis=0)
            divergence = fx_x + fy_y
            
            patterns["movement_coherence"] = float(np.std(divergence))
        
        # Analisar distribuição espacial do movimento
        h, w = magnitude.shape
        
        # Dividir frame em quadrantes
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            magnitude[:mid_h, :mid_w],      # Superior esquerdo
            magnitude[:mid_h, mid_w:],      # Superior direito  
            magnitude[mid_h:, :mid_w],      # Inferior esquerdo
            magnitude[mid_h:, mid_w:]       # Inferior direito
        ]
        
        quadrant_movements = [np.mean(q) for q in quadrants]
        movement_variance = np.var(quadrant_movements)
        
        if movement_variance > np.mean(quadrant_movements) * 0.5:
            patterns["spatial_distribution"] = "concentrated"
        else:
            patterns["spatial_distribution"] = "uniform"
        
        return {"patterns": patterns}
    
    def _detect_movement_anomalies(self, magnitudes: np.ndarray, 
                                 vectors: np.ndarray) -> List[Dict]:
        """Detecta anomalias específicas no movimento (Lucas-Kanade)"""
        anomalies = []
        
        if len(magnitudes) == 0:
            return anomalies
        
        avg_mag = np.mean(magnitudes)
        max_mag = np.max(magnitudes)
        
        # Movimento súbito (pico de velocidade)
        if max_mag > avg_mag * 3 and max_mag > 10:
            anomalies.append({
                "type": "sudden_movement",
                "confidence": min(max_mag / 20, 1.0),
                "description": "Movimento súbito detectado",
                "magnitude": float(max_mag)
            })
        
        # Movimento errático (muitas direções diferentes)
        if len(vectors) > 10:
            directions = np.arctan2(vectors[:, 1], vectors[:, 0])
            direction_std = np.std(directions)
            
            if direction_std > 2.0:  # Muito espalhado
                anomalies.append({
                    "type": "erratic_movement",
                    "confidence": min(direction_std / 3.0, 1.0),
                    "description": "Movimento errático detectado",
                    "direction_variance": float(direction_std)
                })
        
        return anomalies
    
    def _detect_dense_flow_anomalies(self, magnitude: np.ndarray, 
                                   direction: np.ndarray, flow: np.ndarray) -> List[Dict]:
        """Detecta anomalias no dense optical flow (Farneback)"""
        anomalies = []
        
        avg_magnitude = np.mean(magnitude)
        movement_pixels = np.sum(magnitude > self.threshold)
        total_pixels = magnitude.size
        movement_ratio = movement_pixels / total_pixels
        
        # Movimento repentino em grande área
        if movement_ratio > 0.3 and avg_magnitude > 5.0:
            anomalies.append({
                "type": "large_area_movement",
                "confidence": min(movement_ratio * 2, 1.0),
                "description": "Movimento em grande área detectado",
                "affected_area": float(movement_ratio)
            })
        
        # Ausência de movimento (possível imobilidade)
        if avg_magnitude < 0.5 and movement_ratio < 0.01:
            anomalies.append({
                "type": "no_movement",
                "confidence": 0.7,
                "description": "Ausência de movimento detectada",
                "movement_level": float(avg_magnitude)
            })
        
        # Movimento em padrão radial (possível queda)
        if len(self.flow_history) >= 3:
            # Analisar convergência/divergência
            center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
            y_coords, x_coords = np.mgrid[0:magnitude.shape[0], 0:magnitude.shape[1]]
            
            # Vetores do centro para cada pixel
            to_center_x = center_x - x_coords
            to_center_y = center_y - y_coords
            
            # Produto escalar com fluxo (convergência negativa, divergência positiva)
            convergence = (flow[..., 0] * to_center_x + flow[..., 1] * to_center_y)
            avg_convergence = np.mean(convergence[magnitude > self.threshold])
            
            if avg_convergence < -2.0:  # Movimento convergente forte
                anomalies.append({
                    "type": "radial_convergence",
                    "confidence": min(abs(avg_convergence) / 5.0, 1.0),
                    "description": "Movimento radial convergente (possível queda)",
                    "convergence_strength": float(avg_convergence)
                })
        
        return anomalies
    
    def _analyze_temporal_patterns(self) -> Dict:
        """Analisa padrões temporais no movimento"""
        if len(self.frame_history) < 5:
            return {"temporal_analysis": {}}
        
        recent_history = list(self.frame_history)[-5:]
        magnitudes = [h["movement_magnitude"] for h in recent_history]
        
        temporal_data = {
            "movement_trend": "stable",
            "trend_strength": 0.0,
            "consistency": 0.0
        }
        
        # Calcular tendência
        x = np.arange(len(magnitudes))
        if len(magnitudes) > 1:
            slope = np.polyfit(x, magnitudes, 1)[0]
            
            if slope > 0.5:
                temporal_data["movement_trend"] = "increasing"
            elif slope < -0.5:
                temporal_data["movement_trend"] = "decreasing"
            
            temporal_data["trend_strength"] = float(abs(slope))
        
        # Calcular consistência
        if len(magnitudes) > 1:
            temporal_data["consistency"] = float(1.0 / (1.0 + np.std(magnitudes)))
        
        return {"temporal_analysis": temporal_data}
    
    def _empty_result(self) -> Dict:
        """Retorna resultado vazio quando não há movimento"""
        return {
            "method": self.method,
            "movement_magnitude": 0.0,
            "max_magnitude": 0.0,
            "has_movement": False,
            "anomalies": [],
            "patterns": {},
            "temporal_analysis": {}
        }
    
    def _update_stats(self, processing_time: float, has_movement: bool):
        """Atualiza estatísticas do detector"""
        self.stats["total_detections"] += 1
        
        if has_movement:
            self.stats["movement_detections"] += 1
        
        # Média móvel do tempo de processamento
        alpha = 0.1
        self.stats["avg_processing_time"] = (
            alpha * processing_time + 
            (1 - alpha) * self.stats["avg_processing_time"]
        )
    
    def get_visualization(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Cria visualização do optical flow no frame
        
        Args:
            frame: Frame original
            result: Resultado da detecção
            
        Returns:
            Frame com visualização do flow
        """
        if frame is None or not result["has_movement"]:
            return frame
        
        vis_frame = frame.copy()
        
        if result["method"] == "lucas_kanade" and "tracking_points" in result:
            # Desenhar pontos e vetores
            points = result["tracking_points"]
            vectors = result.get("movement_vectors", [])
            
            if len(points) > 0 and len(vectors) > 0:
                for i, (point, vector) in enumerate(zip(points, vectors)):
                    x, y = int(point[0]), int(point[1])
                    dx, dy = int(vector[0] * 5), int(vector[1] * 5)  # Escalar para visualização
                    
                    # Desenhar ponto
                    cv2.circle(vis_frame, (x, y), 3, (0, 255, 0), -1)
                    
                    # Desenhar vetor
                    if abs(dx) > 1 or abs(dy) > 1:
                        cv2.arrowedLine(vis_frame, (x, y), (x + dx, y + dy), (0, 0, 255), 2)
        
        elif result["method"] == "farneback" and "flow_field" in result:
            # Visualizar dense flow
            flow = result["flow_field"]
            magnitude = result["magnitude_field"]
            
            # Criar visualização HSV
            h, w = flow.shape[:2]
            hsv = np.zeros((h, w, 3), dtype=np.uint8)
            hsv[..., 1] = 255
            
            # Direção -> Hue, Magnitude -> Value
            direction = np.arctan2(flow[..., 1], flow[..., 0])
            hsv[..., 0] = (direction + np.pi) * 180 / (2 * np.pi)
            hsv[..., 2] = np.clip(magnitude * 10, 0, 255)
            
            # Converter para BGR
            flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Combinar com frame original (overlay)
            vis_frame = cv2.addWeighted(vis_frame, 0.7, flow_vis, 0.3, 0)
        
        # Adicionar informações de texto
        info_text = [
            f"Método: {result['method']}",
            f"Magnitude: {result['movement_magnitude']:.2f}",
            f"Movimento: {'Sim' if result['has_movement'] else 'Não'}"
        ]
        
        if result["anomalies"]:
            info_text.append(f"Anomalias: {len(result['anomalies'])}")
        
        # Desenhar texto
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(vis_frame, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame
    
    def reset(self):
        """Reset do detector"""
        self.previous_frame = None
        self.tracking_points = None
        self.frame_history.clear()
        self.flow_history.clear()
        logger.info("OpticalFlowDetector resetado")
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do detector"""
        stats = self.stats.copy()
        
        if stats["total_detections"] > 0:
            stats["movement_detection_rate"] = (
                stats["movement_detections"] / stats["total_detections"]
            )
        else:
            stats["movement_detection_rate"] = 0.0
        
        return stats