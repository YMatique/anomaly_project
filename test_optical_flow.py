# test_optical_flow.py
import sys
import os
sys.path.append('./src')

# Simular o logger
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")

# Substituir imports
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import time

logger = SimpleLogger()

class OpticalFlowDetector:
    """Detector de anomalias baseado em Optical Flow - VERSÃO TESTE"""
    
    def __init__(self, method: str = "farneback"):
        self.method = method
        self.prev_frame = None
        self.prev_gray = None
        
        # Configurações
        self.motion_threshold = 0.05
        self.anomaly_threshold = 0.3
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=16,
            history=500
        )
        
        logger.info(f"OpticalFlowDetector inicializado - método: {method}")
    
    def detect(self, frame: np.ndarray) -> Dict:
        """Detecta anomalias usando optical flow"""
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
            
            # Background subtraction
            fg_mask = self.bg_subtractor.apply(frame)
            motion_ratio = np.sum(fg_mask > 0) / fg_mask.size
            
            result["motion_detected"] = motion_ratio > self.motion_threshold
            result["motion_intensity"] = float(motion_ratio)
            
            # Optical Flow se tem frame anterior
            if self.prev_gray is not None:
                flow_magnitude = self._calculate_optical_flow(self.prev_gray, gray)
                result["flow_magnitude"] = float(flow_magnitude)
                
                # Detectar anomalia
                if flow_magnitude > self.anomaly_threshold:
                    result["is_anomaly"] = True
                    result["confidence"] = min(flow_magnitude / self.anomaly_threshold, 1.0)
                    result["anomaly_type"] = "motion"
            
            # Salvar frame atual
            self.prev_gray = gray.copy()
            self.prev_frame = frame.copy()
            
            return result
            
        except Exception as e:
            logger.error(f"Erro no optical flow detector: {e}")
            return self._create_empty_result(f"Erro: {str(e)}")
    
    def _calculate_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """Calcula magnitude do optical flow"""
        try:
            # Detectar features para rastrear
            corners = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=100, qualityLevel=0.01, 
                minDistance=10, blockSize=3
            )
            
            if corners is not None and len(corners) > 0:
                # Calcular optical flow Lucas-Kanade
                next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_gray, curr_gray, corners, None
                )
                
                # Filtrar pontos válidos
                good_old = corners[status == 1]
                good_new = next_pts[status == 1]
                
                if len(good_old) > 0:
                    # Calcular magnitude média
                    displacement = good_new - good_old
                    magnitude = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
                    return float(np.mean(magnitude))
            
            return 0.0
                
        except Exception as e:
            logger.error(f"Erro no cálculo optical flow: {e}")
            return 0.0
    
    def _create_empty_result(self, error_msg: str = None) -> Dict:
        """Cria resultado vazio"""
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

def test_optical_flow():
    """Teste do optical flow detector"""
    
    print("🧪 Testando Optical Flow Detector...")
    
    detector = OpticalFlowDetector()
    
    # Criar frames de teste realistas
    print("Criando frames de teste...")
    
    # Frame 1 - fundo estático
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame1, (100, 100), (200, 200), (255, 255, 255), -1)
    
    # Frame 2 - objeto moveu
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame2, (110, 110), (210, 210), (255, 255, 255), -1)  # Moveu 10px
    
    # Testar primeiro frame
    print("\n📊 Testando Frame 1...")
    result1 = detector.detect(frame1)
    print(f"Resultado 1: {result1}")
    
    # Testar segundo frame (com movimento)
    print("\n📊 Testando Frame 2 (com movimento)...")
    result2 = detector.detect(frame2)
    print(f"Resultado 2: {result2}")
    
    # Verificar estrutura
    print("\n🔍 Verificando estrutura dos dados...")
    expected_keys = [
        'is_anomaly', 'confidence', 'motion_detected', 
        'motion_intensity', 'flow_magnitude', 'anomaly_type'
    ]
    
    for key in expected_keys:
        if key in result2:
            print(f"✅ {key}: {result2[key]}")
        else:
            print(f"❌ {key}: FALTANDO")
    
    # Teste com frame real de vídeo se existir
    video_path = "./data/videos/normal/h-6.mp4"
    if os.path.exists(video_path):
        print(f"\n🎬 Testando com vídeo real: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret1, real_frame1 = cap.read()
            ret2, real_frame2 = cap.read()
            
            if ret1 and ret2:
                result_real = detector.detect(real_frame1)
                result_real2 = detector.detect(real_frame2)
                
                print(f"Frame real 1: motion={result_real.get('motion_detected')}")
                print(f"Frame real 2: motion={result_real2.get('motion_detected')}, flow={result_real2.get('flow_magnitude'):.4f}")
            
            cap.release()
    
    print("\n✅ Teste do Optical Flow concluído!")

if __name__ == "__main__":
    test_optical_flow()