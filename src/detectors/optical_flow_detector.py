"""
Detector de Optical Flow para detecção de movimento e anomalias
Primeira camada de detecção - rápida e eficiente
VERSÃO CORRIGIDA - Retorna estrutura compatível
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import time

from ..utils.logger import logger

class OpticalFlowDetector:
    """
    Detector de anomalias baseado em Optical Flow
    Detecta movimento anômalo comparando frames consecutivos
    """
    
    def __init__(self, method: str = "farneback"):
        self.method = method
        self.prev_frame = None
        self.prev_gray = None
        
        # Configurações
        self.motion_threshold = 0.05  # Threshold para detectar movimento
        self.anomaly_threshold = 0.3  # Threshold para detectar anomalia
        
        # Background subtractor para movimento
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )
        
        # Parâmetros do Optical Flow
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        logger.info(f"OpticalFlowDetector inicializado - método: {method}")
    
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detecta anomalias usando optical flow
        
        Args:
            frame: Frame atual [H, W, C]
            
        Returns:
            Dict com estrutura padronizada de resultados
        """
        if frame is None:
            return self._create_empty_result("Frame inválido")
        
        try:
            # Converter para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Resultado padrão
            result = {
                "is_anomaly": False,
                "confidence": 0.0,
                "motion_detected": False,
                "motion_intensity": 0.0,
                "flow_magnitude": 0.0,
                "anomaly_type": "none",
                "timestamp": time.time()
            }
            
            # Background subtraction para detectar movimento
            fg_mask = self.bg_subtractor.apply(frame)
            motion_ratio = np.sum(fg_mask > 0) / fg_mask.size
            
            result["motion_detected"] = motion_ratio > self.motion_threshold
            result["motion_intensity"] = float(motion_ratio)
            
            # Optical Flow se tem frame anterior
            if self.prev_gray is not None:
                # Calcular optical flow
                flow_magnitude = self._calculate_optical_flow(self.prev_gray, gray)
                result["flow_magnitude"] = float(flow_magnitude)
                
                # Detectar anomalia baseada em movimento
                if flow_magnitude > self.anomaly_threshold:
                    result["is_anomaly"] = True
                    result["confidence"] = min(flow_magnitude / self.anomaly_threshold, 1.0)
                    result["anomaly_type"] = "motion"
                
                # Detectar movimento súbito (anomalia)
                if motion_ratio > self.motion_threshold * 3:  # 3x o threshold normal
                    result["is_anomaly"] = True
                    result["confidence"] = max(result["confidence"], motion_ratio)
                    result["anomaly_type"] = "sudden_movement"
            
            # Salvar frame atual para próxima iteração
            self.prev_gray = gray.copy()
            self.prev_frame = frame.copy()
            
            return result
            
        except Exception as e:
            logger.error(f"Erro no optical flow detector: {e}")
            return self._create_empty_result(f"Erro: {str(e)}")
    
    def _calculate_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """Calcula magnitude do optical flow"""
        try:
            if self.method == "farneback":
                # Farneback optical flow
                flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, None, None, **self.flow_params)
                
                if len(flow) >= 2 and flow[0] is not None:
                    # Calcular magnitude média
                    magnitude = np.sqrt(flow[0][:, :, 0]**2 + flow[0][:, :, 1]**2)
                    return float(np.mean(magnitude))
                else:
                    return 0.0
            
            elif self.method == "lucas_kanade":
                # Lucas-Kanade optical flow
                # Detectar features para rastrear
                corners = cv2.goodFeaturesToTrack(
                    prev_gray, maxCorners=100, qualityLevel=0.01, 
                    minDistance=10, blockSize=3
                )
                
                if corners is not None and len(corners) > 0:
                    # Calcular optical flow
                    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        prev_gray, curr_gray, corners, None
                    )
                    
                    # Filtrar pontos válidos
                    good_old = corners[status == 1]
                    good_new = next_pts[status == 1]
                    
                    if len(good_old) > 0:
                        # Calcular magnitude média do movimento
                        displacement = good_new - good_old
                        magnitude = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
                        return float(np.mean(magnitude))
                
                return 0.0
            
            else:
                logger.warning(f"Método {self.method} não implementado, usando farneback")
                return self._calculate_optical_flow(prev_gray, curr_gray)
                
        except Exception as e:
            logger.error(f"Erro no cálculo optical flow: {e}")
            return 0.0
    
    def _create_empty_result(self, error_msg: str = None) -> Dict:
        """Cria resultado vazio com estrutura padrão"""
        return {
            "is_anomaly": False,
            "confidence": 0.0,
            "motion_detected": False,
            "motion_intensity": 0.0,
            "flow_magnitude": 0.0,
            "anomaly_type": "none",
            "timestamp": time.time(),
            "error": error_msg if error_msg else None
        }
    
    def reset(self):
        """Reseta o detector"""
        self.prev_frame = None
        self.prev_gray = None
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )
        logger.info("OpticalFlowDetector resetado")
    
    def get_info(self) -> Dict:
        """Retorna informações do detector"""
        return {
            "method": self.method,
            "motion_threshold": self.motion_threshold,
            "anomaly_threshold": self.anomaly_threshold,
            "has_previous_frame": self.prev_frame is not None,
            "ready": True  # Optical flow sempre está pronto
        }
    
    def set_thresholds(self, motion_threshold: float = None, anomaly_threshold: float = None):
        """Ajusta thresholds do detector"""
        if motion_threshold is not None:
            self.motion_threshold = motion_threshold
            logger.info(f"Motion threshold ajustado para: {motion_threshold}")
        
        if anomaly_threshold is not None:
            self.anomaly_threshold = anomaly_threshold
            logger.info(f"Anomaly threshold ajustado para: {anomaly_threshold}")

# ===== TESTE DO OPTICAL FLOW =====
def test_optical_flow():
    """Função para testar o optical flow detector"""
    
    print("🧪 Testando Optical Flow Detector...")
    
    detector = OpticalFlowDetector()
    
    # Criar frames de teste
    frame1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Testar primeiro frame
    result1 = detector.detect(frame1)
    print(f"Frame 1: {result1}")
    
    # Testar segundo frame
    result2 = detector.detect(frame2)
    print(f"Frame 2: {result2}")
    
    # Verificar estrutura
    expected_keys = ['is_anomaly', 'confidence', 'motion_detected', 'motion_intensity']
    for key in expected_keys:
        if key in result2:
            print(f"✅ {key}: {result2[key]}")
        else:
            print(f"❌ {key}: FALTANDO")
    
    print("✅ Teste concluído")

if __name__ == "__main__":
    test_optical_flow()