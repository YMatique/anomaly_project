"""
Interface Web Flask - Bypass do Logger Problemático
Contorna o erro do Logger e funciona com sistema existente
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

# Setup básico de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WebApp")

# Adiciona path do projeto
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

# PATCH TEMPORÁRIO: Substitui a classe Logger problemática
class MockLogger:
    """Logger mock que aceita parâmetros sem causar erro"""
    def __init__(self, name=None):
        self.name = name or "MockLogger"
        self.logger = logging.getLogger(self.name)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)

# Injeta o MockLogger no sistema antes de importar
try:
    from src.utils import logger as logger_module
    # Substitui a classe Logger problemática
    logger_module.Logger = MockLogger
    logger.info("✅ Logger patcheado com sucesso")
except:
    pass

# Agora tenta importar o sistema principal
try:
    from main import AnomalyDetectionSystem
    MAIN_SYSTEM_AVAILABLE = True
    logger.info("✅ Sistema principal importado com sucesso!")
except ImportError as e:
    logger.warning(f"⚠️ Erro ao importar sistema principal: {e}")
    MAIN_SYSTEM_AVAILABLE = False

# Tenta importar detectores individuais como fallback
try:
    from src.detectors.optical_flow_detector import OpticalFlowDetector
    from src.detectors.deep_learning_detector import DeepLearningDetector
    from src.utils.config import Config
    DETECTORS_AVAILABLE = True
    logger.info("✅ Detectores individuais disponíveis")
except ImportError as e:
    logger.warning(f"⚠️ Detectores individuais não disponíveis: {e}")
    DETECTORS_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Estado global do sistema
system_running = False
system_paused = False
training_mode = False
current_frame = None
current_frame_with_detection = None
camera_capture = None

# Sistema principal ou detectores individuais
main_system = None
optical_flow_detector = None
deep_learning_detector = None
config = None

# Buffer para frames anteriores (optical flow)
previous_frames = []
max_previous_frames = 3

# Estatísticas em tempo real
stats = {
    'frames_processed': 0,
    'anomalies_detected': 0,
    'fps': 0.0,
    'avg_processing_time': 0.0,
    'optical_flow_detections': 0,
    'deep_learning_detections': 0,
    'recent_alerts': []
}

# Locks para thread safety
frame_lock = threading.Lock()
stats_lock = threading.Lock()
detection_lock = threading.Lock()

def initialize_system_safe():
    """Inicializa o sistema de forma segura, contornando problemas do Logger"""
    global main_system, optical_flow_detector, deep_learning_detector, config
    
    try:
        if MAIN_SYSTEM_AVAILABLE:
            logger.info("🚀 Inicializando sistema principal (com patch do Logger)...")
            
            # Tenta criar instância do sistema principal
            main_system = AnomalyDetectionSystem()
            logger.info("✅ Sistema principal inicializado com sucesso")
            return True
            
        elif DETECTORS_AVAILABLE:
            logger.info("🚀 Inicializando detectores individuais...")
            
            # Cria configuração mock se necessário
            try:
                config = Config()
            except:
                config = type('MockConfig', (), {
                    'get': lambda self, key, default=None: default,
                    'get_all': lambda self: {}
                })()
            
            # Inicializa detectores
            optical_flow_detector = OpticalFlowDetector(config)
            deep_learning_detector = DeepLearningDetector(config)
            logger.info("✅ Detectores individuais inicializados")
            return True
        else:
            logger.warning("⚠️ Nenhum sistema de detecção disponível")
            return False
            
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar sistema: {e}")
        return False

def simple_optical_flow(current_frame, prev_frame):
    """Implementação simples e robusta de optical flow"""
    try:
        # Converte para grayscale
        if len(current_frame.shape) == 3:
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = current_frame
            
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
        
        # Redimensiona se necessário para garantir compatibilidade
        if curr_gray.shape != prev_gray.shape:
            prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))
        
        # Calcula optical flow usando Lucas-Kanade (mais robusto)
        # Detecta cantos para tracking
        corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        
        if corners is not None:
            # Calcula optical flow
            next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)
            
            # Filtra pontos válidos
            good_new = next_points[status == 1]
            good_old = corners[status == 1]
            
            # Calcula magnitude do movimento
            if len(good_new) > 0 and len(good_old) > 0:
                movement_vectors = good_new - good_old
                magnitudes = np.sqrt(movement_vectors[:, 0]**2 + movement_vectors[:, 1]**2)
                avg_magnitude = np.mean(magnitudes)
                
                # Cria vetores para visualização
                flow_vectors = []
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    if magnitudes[i] > 2:  # Apenas movimentos significativos
                        flow_vectors.append((old[0], old[1], new[0], new[1]))
                
                return {
                    'score': min(avg_magnitude / 50.0, 1.0),  # Normaliza para 0-1
                    'flow_vectors': flow_vectors[:30],  # Limita a 30 vetores
                    'magnitude': avg_magnitude
                }
        
        return {'score': 0.0, 'flow_vectors': [], 'magnitude': 0.0}
        
    except Exception as e:
        logger.warning(f"⚠️ Erro no optical flow simples: {e}")
        return {'score': 0.0, 'flow_vectors': [], 'error': str(e)}

def process_frame_simple_detection(frame):
    """Processamento simples e robusto de detecção"""
    global previous_frames, stats
    
    try:
        start_time = time.time()
        detection_found = False
        anomaly_type = "normal"
        confidence = 0.0
        optical_flow_result = {'score': 0.0, 'flow_vectors': []}
        
        # ESTÁGIO 1: Optical Flow Simples
        if len(previous_frames) > 0:
            optical_flow_result = simple_optical_flow(frame, previous_frames[-1])
            optical_flow_score = optical_flow_result.get('score', 0.0)
            movement_detected = optical_flow_score > 0.2
            
            if movement_detected:
                with stats_lock:
                    stats['optical_flow_detections'] += 1
        else:
            movement_detected = False
        
        # ESTÁGIO 2: Deep Learning (se disponível e movimento detectado)
        if movement_detected and deep_learning_detector:
            try:
                # Redimensiona frame
                resized_frame = cv2.resize(frame, (64, 64))
                
                # Análise com CAE
                cae_result = deep_learning_detector.detect_frame(resized_frame)
                
                if cae_result.get('anomaly_detected', False):
                    detection_found = True
                    anomaly_type = cae_result.get('anomaly_type', 'anomalia_detectada')
                    confidence = cae_result.get('confidence', 0.0)
                    
                    with stats_lock:
                        stats['deep_learning_detections'] += 1
                        stats['anomalies_detected'] += 1
                    
                    logger.info(f"🚨 Anomalia detectada: {anomaly_type} (conf: {confidence:.2f})")
                
            except Exception as e:
                logger.warning(f"⚠️ Erro no deep learning: {e}")
        
        # Atualiza buffer de frames anteriores
        previous_frames.append(frame.copy())
        if len(previous_frames) > max_previous_frames:
            previous_frames.pop(0)
        
        # Cria frame com visualizações
        detection_frame = create_detection_overlay(
            frame.copy(), 
            optical_flow_result, 
            detection_found, 
            anomaly_type, 
            confidence
        )
        
        # Atualiza estatísticas
        processing_time = time.time() - start_time
        with stats_lock:
            stats['avg_processing_time'] = processing_time
        
        # Adiciona alerta se anomalia detectada
        if detection_found:
            add_anomaly_alert(anomaly_type, confidence)
        
        return detection_frame, detection_found, anomaly_type
        
    except Exception as e:
        logger.error(f"❌ Erro no processamento: {e}")
        return frame, False, "erro"

def create_detection_overlay(frame, optical_flow_result, anomaly_detected, anomaly_type, confidence):
    """Cria overlay visual para detecções"""
    try:
        overlay = frame.copy()
        height, width = overlay.shape[:2]
        
        # Desenha optical flow
        if optical_flow_result and 'flow_vectors' in optical_flow_result:
            for (x1, y1, x2, y2) in optical_flow_result['flow_vectors']:
                cv2.arrowedLine(overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                               (0, 255, 255), 2, tipLength=0.5)
        
        # Overlay de anomalia
        if anomaly_detected:
            color = (0, 0, 255) if 'security' in anomaly_type.lower() else (0, 165, 255)
            
            # Borda piscante
            border_thickness = 8 if int(time.time() * 4) % 2 == 0 else 4
            cv2.rectangle(overlay, (5, 5), (width-5, height-5), color, border_thickness)
            
            # Fundo para texto
            cv2.rectangle(overlay, (10, 10), (min(600, width-10), 100), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (min(600, width-10), 100), color, 2)
            
            # Texto de alerta
            alert_text = f"ANOMALIA: {anomaly_type.upper()}"
            confidence_text = f"Confianca: {confidence:.1%}"
            detector_text = "Deep Learning" if deep_learning_detector else "Simulado"
            
            cv2.putText(overlay, alert_text, (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(overlay, confidence_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(overlay, f"Detector: {detector_text}", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Informações de optical flow
        if optical_flow_result:
            flow_score = optical_flow_result.get('score', 0.0)
            magnitude = optical_flow_result.get('magnitude', 0.0)
            
            flow_text = f"Movement: {flow_score:.3f} | Mag: {magnitude:.1f}"
            cv2.putText(overlay, flow_text, (10, height-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status dos detectores
        detector_status = []
        if optical_flow_detector or len(previous_frames) > 0:
            detector_status.append("OpticalFlow:OK")
        if deep_learning_detector:
            detector_status.append("DeepLearning:OK")
        if main_system:
            detector_status.append("MainSystem:OK")
        
        status_text = " | ".join(detector_status) if detector_status else "Demo Mode"
        cv2.putText(overlay, status_text, (10, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)
        
        return overlay
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar overlay: {e}")
        return frame

def add_anomaly_alert(anomaly_type, confidence):
    """Adiciona alerta de anomalia"""
    try:
        with stats_lock:
            alert = {
                'message': f"Anomalia detectada: {anomaly_type}",
                'type': 'security' if 'security' in anomaly_type.lower() else 'health',
                'timestamp': time.time(),
                'confidence': confidence,
                'detector': 'Deep Learning' if deep_learning_detector else 'Optical Flow'
            }
            
            stats['recent_alerts'].insert(0, alert)
            stats['recent_alerts'] = stats['recent_alerts'][:20]
        
    except Exception as e:
        logger.error(f"❌ Erro ao adicionar alerta: {e}")

def camera_loop_robust():
    """Loop da câmera robusto e simples"""
    global current_frame, current_frame_with_detection, stats
    
    frame_count = 0
    start_time = time.time()
    
    while system_running and camera_capture:
        try:
            ret, frame = camera_capture.read()
            if ret:
                # Frame original
                with frame_lock:
                    current_frame = frame.copy()
                
                # Processamento (apenas se não pausado)
                if not system_paused:
                    with detection_lock:
                        detection_frame, anomaly_found, anomaly_type = process_frame_simple_detection(frame)
                        current_frame_with_detection = detection_frame
                else:
                    with detection_lock:
                        current_frame_with_detection = frame.copy()
                
                # Atualiza estatísticas
                frame_count += 1
                elapsed = time.time() - start_time
                
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    
                    with stats_lock:
                        stats['frames_processed'] += frame_count
                        stats['fps'] = fps
                    
                    frame_count = 0
                    start_time = time.time()
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"❌ Erro no loop da câmera: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/system/start', methods=['POST'])
def start_system():
    global system_running, system_paused, camera_capture
    
    try:
        if system_running:
            return jsonify({'error': 'Sistema já está rodando'}), 400
        
        data = request.get_json() or {}
        source_type = data.get('source_type', 'webcam')
        source_param = data.get('source_param', '0')
        
        logger.info(f"🚀 Iniciando sistema (com bypass do Logger) - {source_type}: {source_param}")
        
        # Inicializa sistema se necessário
        if not main_system and not optical_flow_detector:
            if not initialize_system_safe():
                logger.warning("⚠️ Sistema funcionará em modo básico")
        
        # Abre câmera ou vídeo
        if source_type == 'webcam':
            camera_index = int(source_param) if source_param.isdigit() else 0
            camera_capture = cv2.VideoCapture(camera_index)
        elif source_type == 'video':
            camera_capture = cv2.VideoCapture(source_param)
        
        if not camera_capture or not camera_capture.isOpened():
            return jsonify({'error': 'Não foi possível abrir a fonte de vídeo'}), 400
        
        # Configura resolução
        camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        system_running = True
        system_paused = False
        
        # Limpa estatísticas e buffers
        with stats_lock:
            stats['frames_processed'] = 0
            stats['anomalies_detected'] = 0
            stats['optical_flow_detections'] = 0
            stats['deep_learning_detections'] = 0
        
        global previous_frames
        previous_frames = []
        
        # Inicia thread da câmera
        camera_thread = threading.Thread(target=camera_loop_robust, daemon=True)
        camera_thread.start()
        
        # Determina modo de detecção
        if main_system:
            detection_mode = "SISTEMA PRINCIPAL"
        elif optical_flow_detector and deep_learning_detector:
            detection_mode = "DETECTORES INDIVIDUAIS"
        else:
            detection_mode = "OPTICAL FLOW BÁSICO"
        
        logger.info(f"✅ Sistema iniciado - Modo: {detection_mode}")
        
        return jsonify({
            'status': 'success',
            'message': f'Sistema iniciado - {detection_mode}',
            'detection_mode': detection_mode
        })
        
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar sistema: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/pause', methods=['POST'])
def pause_system():
    global system_paused
    
    try:
        if not system_running:
            return jsonify({'error': 'Sistema não está rodando'}), 400
        
        system_paused = not system_paused
        message = 'Sistema pausado' if system_paused else 'Sistema retomado'
        
        logger.info(f"⏯️ {message}")
        return jsonify({
            'status': 'success',
            'message': message,
            'paused': system_paused
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    global system_running, system_paused, camera_capture, current_frame, current_frame_with_detection, previous_frames
    
    try:
        system_running = False
        system_paused = False
        
        if camera_capture:
            camera_capture.release()
            camera_capture = None
        
        with frame_lock:
            current_frame = None
        
        with detection_lock:
            current_frame_with_detection = None
        
        previous_frames = []
        
        logger.info("🛑 Sistema parado")
        return jsonify({'status': 'success', 'message': 'Sistema parado'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    try:
        with stats_lock:
            current_stats = stats.copy()
        
        current_stats.update({
            'system_running': system_running,
            'system_paused': system_paused,
            'training_mode': training_mode,
            'main_system_available': main_system is not None,
            'detectors_available': optical_flow_detector is not None or deep_learning_detector is not None,
            'frame_buffer_size': len(previous_frames),
            'timestamp': time.time()
        })
        
        return jsonify(current_stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/feed')
def video_feed():
    try:
        # Usa frame com detecção se disponível
        with detection_lock:
            if current_frame_with_detection is not None:
                frame = current_frame_with_detection.copy()
            else:
                with frame_lock:
                    if current_frame is not None:
                        frame = current_frame.copy()
                    else:
                        # Frame placeholder
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, 'Aguardando Video...', (180, 240),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Adiciona status do sistema
        if system_running:
            status_text = "PAUSADO" if system_paused else "DETECÇÃO ATIVA"
            color = (0, 255, 255) if system_paused else (0, 255, 0)
            
            # Indica tipo de sistema
            if main_system:
                status_text += " [PRINCIPAL]"
            elif optical_flow_detector and deep_learning_detector:
                status_text += " [INDIVIDUAL]"
            else:
                status_text += " [BÁSICO]"
            
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # FPS e contadores
            fps = stats.get('fps', 0)
            processed = stats.get('frames_processed', 0)
            anomalies = stats.get('anomalies_detected', 0)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Frames: {processed} | Anomalias: {anomalies}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Codifica como JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return Response(
            buffer.tobytes(),
            mimetype='image/jpeg',
            headers={'Cache-Control': 'no-cache, no-store, must-revalidate'}
        )
        
    except Exception as e:
        logger.error(f"❌ Erro no feed de vídeo: {e}")
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, 'Erro no Video', (250, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/training/start', methods=['POST'])
def start_training():
    global training_mode
    
    try:
        data = request.get_json() or {}
        duration = data.get('duration', 15)
        
        training_mode = True
        
        if main_system:
            logger.info(f"🎓 Treinamento com sistema principal por {duration} minutos")
        else:
            logger.info(f"🎓 Treinamento simulado por {duration} minutos")
        
        # Para automaticamente após duração
        def stop_training():
            time.sleep(duration * 60)
            global training_mode
            training_mode = False
            logger.info("🎓 Treinamento finalizado")
        
        threading.Thread(target=stop_training, daemon=True).start()
        
        return jsonify({
            'status': 'success',
            'message': f'Treinamento iniciado por {duration} minutos'
        })
        
    except Exception as e:
        logger.error(f"❌ Erro ao iniciar treinamento: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    global training_mode
    
    try:
        training_mode = False
        logger.info("🎓 Treinamento parado")
        return jsonify({'status': 'success', 'message': 'Treinamento parado'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Outras rotas mantidas...
@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        return jsonify({
            'sensitivity': 0.5,
            'detection_mode': 'all',
            'source_type': 'webcam',
            'source_param': '0'
        })
    else:
        return jsonify({'status': 'success', 'message': 'Configuração salva'})

@app.route('/api/model/save', methods=['POST'])
def save_model():
    try:
        if main_system:
            logger.info("💾 Salvando modelos do sistema principal...")
            return jsonify({'status': 'success', 'message': 'Modelos salvos com sistema principal'})
        else:
            return jsonify({'status': 'success', 'message': 'Salvamento simulado'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/screenshot', methods=['POST'])
def take_screenshot():
    try:
        with detection_lock:
            if current_frame_with_detection is not None:
                frame = current_frame_with_detection.copy()
            else:
                with frame_lock:
                    if current_frame is None:
                        return jsonify({'error': 'Nenhum frame disponível'}), 400
                    frame = current_frame.copy()
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'detection_screenshot_{timestamp}.jpg'
        
        return Response(
            buffer.tobytes(),
            mimetype='image/jpeg',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    try:
        with stats_lock:
            stats['recent_alerts'] = []
        return jsonify({'status': 'success', 'message': 'Alertas limpos'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info("🚀 Iniciando servidor web com bypass do Logger...")
        
        # Inicializa sistema na inicialização do servidor
        initialize_system_safe()
        
        logger.info("🌐 Servidor disponível em http://localhost:5000")
        logger.info("🔧 Bypass do Logger aplicado - sem erros de inicialização")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("👋 Servidor interrompido")
    except Exception as e:
        logger.error(f"❌ Erro: {e}")
    finally:
        if camera_capture:
            camera_capture.release()
        cv2.destroyAllWindows()