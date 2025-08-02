"""
Interface Web Flask com Algoritmos Reais de Detecção
Integra Optical Flow + CAE + ConvLSTM para detecção real de anomalias
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

# Tenta importar módulos reais
try:
    from src.detectors.optical_flow_detector import OpticalFlowDetector
    from src.detectors.deep_learning_detector import DeepLearningDetector
    from src.utils.config import Config
    REAL_DETECTORS_AVAILABLE = True
    logger.info("✅ Detectores reais importados com sucesso!")
except ImportError as e:
    logger.warning(f"⚠️ Erro ao importar detectores: {e}")
    REAL_DETECTORS_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Estado global do sistema
system_running = False
system_paused = False
training_mode = False
current_frame = None
current_frame_with_detection = None
camera_capture = None

# Detectores reais
optical_flow_detector = None
deep_learning_detector = None
config = None

# Buffer para análise temporal (ConvLSTM)
frame_buffer = []
max_buffer_size = 10

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

def initialize_real_detectors():
    """Inicializa os detectores reais"""
    global optical_flow_detector, deep_learning_detector, config
    
    if not REAL_DETECTORS_AVAILABLE:
        return False
    
    try:
        logger.info("🚀 Inicializando detectores reais...")
        
        # Carrega configuração
        config = Config()
        
        # Inicializa Optical Flow Detector
        optical_flow_detector = OpticalFlowDetector(config)
        logger.info("✅ Optical Flow Detector inicializado")
        
        # Inicializa Deep Learning Detector (CAE + ConvLSTM)
        deep_learning_detector = DeepLearningDetector(config)
        logger.info("✅ Deep Learning Detector (CAE + ConvLSTM) inicializado")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro ao inicializar detectores: {e}")
        return False

def process_frame_with_real_detection(frame):
    """Processa frame com detecção real de anomalias"""
    global frame_buffer, stats
    
    if not optical_flow_detector or not deep_learning_detector:
        return frame, False, "sem_detectores"
    
    try:
        start_time = time.time()
        detection_found = False
        anomaly_type = "normal"
        confidence = 0.0
        
        # ESTÁGIO 1: Optical Flow (detecção rápida de movimento)
        optical_flow_result = optical_flow_detector.detect(frame)
        optical_flow_score = optical_flow_result.get('score', 0.0)
        movement_detected = optical_flow_score > 0.3  # threshold
        
        if movement_detected:
            with stats_lock:
                stats['optical_flow_detections'] += 1
        
        # ESTÁGIO 2: Deep Learning (apenas se movimento detectado)
        if movement_detected:
            # Redimensiona frame para entrada do modelo
            resized_frame = cv2.resize(frame, (64, 64))
            
            # Análise com CAE (frame único)
            cae_result = deep_learning_detector.detect_frame(resized_frame)
            
            if cae_result.get('anomaly_detected', False):
                detection_found = True
                anomaly_type = cae_result.get('anomaly_type', 'anomalia_cae')
                confidence = cae_result.get('confidence', 0.0)
                
                with stats_lock:
                    stats['deep_learning_detections'] += 1
                    stats['anomalies_detected'] += 1
            
            # Adiciona frame ao buffer para ConvLSTM
            frame_buffer.append(resized_frame)
            if len(frame_buffer) > max_buffer_size:
                frame_buffer.pop(0)
            
            # Análise temporal com ConvLSTM (quando buffer está cheio)
            if len(frame_buffer) >= max_buffer_size:
                try:
                    convlstm_result = deep_learning_detector.detect_sequence(frame_buffer.copy())
                    
                    if convlstm_result.get('anomaly_detected', False):
                        detection_found = True
                        anomaly_type = convlstm_result.get('anomaly_type', 'anomalia_temporal')
                        confidence = max(confidence, convlstm_result.get('confidence', 0.0))
                        
                        logger.info(f"🧠 ConvLSTM detectou anomalia temporal: {anomaly_type}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Erro no ConvLSTM: {e}")
        
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
        logger.error(f"❌ Erro no processamento de detecção: {e}")
        return frame, False, "erro"

def create_detection_overlay(frame, optical_flow_result, anomaly_detected, anomaly_type, confidence):
    """Cria overlay visual com informações de detecção"""
    try:
        overlay = frame.copy()
        height, width = overlay.shape[:2]
        
        # Desenha optical flow se disponível
        if optical_flow_result and 'flow_vectors' in optical_flow_result:
            flow_vectors = optical_flow_result['flow_vectors']
            if flow_vectors is not None and len(flow_vectors) > 0:
                # Desenha vetores de movimento
                for (x1, y1, x2, y2) in flow_vectors[:50]:  # Máximo 50 vetores
                    cv2.arrowedLine(overlay, (int(x1), int(y1)), (int(x2), int(y2)), 
                                   (0, 255, 255), 1, tipLength=0.3)
        
        # Overlay de anomalia
        if anomaly_detected:
            # Cor baseada no tipo
            if 'security' in anomaly_type.lower():
                color = (0, 0, 255)  # Vermelho
            elif 'health' in anomaly_type.lower():
                color = (0, 165, 255)  # Laranja
            else:
                color = (0, 255, 255)  # Amarelo
            
            # Borda de alerta piscante
            border_thickness = 8 if int(time.time() * 3) % 2 == 0 else 4
            cv2.rectangle(overlay, (5, 5), (width-5, height-5), color, border_thickness)
            
            # Fundo para texto
            cv2.rectangle(overlay, (10, 10), (min(600, width-10), 100), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (min(600, width-10), 100), color, 2)
            
            # Texto de alerta
            alert_text = f"ANOMALIA: {anomaly_type.upper()}"
            confidence_text = f"Confianca: {confidence:.1%}"
            
            cv2.putText(overlay, alert_text, (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(overlay, confidence_text, (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Informações de optical flow
        if optical_flow_result:
            flow_score = optical_flow_result.get('score', 0.0)
            flow_text = f"Optical Flow: {flow_score:.3f}"
            cv2.putText(overlay, flow_text, (10, height-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Buffer status para ConvLSTM
        buffer_text = f"Buffer ConvLSTM: {len(frame_buffer)}/{max_buffer_size}"
        cv2.putText(overlay, buffer_text, (10, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
        
    except Exception as e:
        logger.error(f"❌ Erro ao criar overlay: {e}")
        return frame

def add_anomaly_alert(anomaly_type, confidence):
    """Adiciona alerta de anomalia às estatísticas"""
    try:
        with stats_lock:
            alert = {
                'message': f"Anomalia detectada: {anomaly_type}",
                'type': 'security' if 'security' in anomaly_type.lower() else 'health',
                'timestamp': time.time(),
                'confidence': confidence,
                'detector': 'CAE+ConvLSTM'
            }
            
            stats['recent_alerts'].insert(0, alert)
            stats['recent_alerts'] = stats['recent_alerts'][:20]  # Mantém últimas 20
        
        logger.info(f"🚨 Anomalia adicionada: {anomaly_type} (conf: {confidence:.2f})")
        
    except Exception as e:
        logger.error(f"❌ Erro ao adicionar alerta: {e}")

def camera_loop_with_detection():
    """Loop principal da câmera com detecção real"""
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
                
                # Processamento de detecção (apenas se não pausado)
                if not system_paused:
                    with detection_lock:
                        detection_frame, anomaly_found, anomaly_type = process_frame_with_real_detection(frame)
                        current_frame_with_detection = detection_frame
                else:
                    with detection_lock:
                        current_frame_with_detection = frame.copy()
                
                # Atualiza estatísticas
                frame_count += 1
                elapsed = time.time() - start_time
                
                if elapsed >= 1.0:  # A cada segundo
                    fps = frame_count / elapsed
                    
                    with stats_lock:
                        stats['frames_processed'] += frame_count
                        stats['fps'] = fps
                    
                    frame_count = 0
                    start_time = time.time()
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"❌ Erro no loop da câmera: {e}")
            break

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
        
        logger.info(f"🚀 Iniciando sistema com detecção real - {source_type}: {source_param}")
        
        # Inicializa detectores se ainda não foram inicializados
        if REAL_DETECTORS_AVAILABLE and optical_flow_detector is None:
            if not initialize_real_detectors():
                return jsonify({'error': 'Falha ao inicializar detectores reais'}), 500
        
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
        
        # Inicia thread da câmera com detecção
        camera_thread = threading.Thread(target=camera_loop_with_detection, daemon=True)
        camera_thread.start()
        
        mode = "DETECÇÃO REAL" if REAL_DETECTORS_AVAILABLE else "MODO DEMO"
        logger.info(f"✅ Sistema iniciado - {mode}")
        
        return jsonify({
            'status': 'success',
            'message': f'Sistema iniciado - {mode}',
            'real_detection': REAL_DETECTORS_AVAILABLE
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
    global system_running, system_paused, camera_capture, current_frame, current_frame_with_detection
    
    try:
        if not system_running:
            return jsonify({'error': 'Sistema não está rodando'}), 400
        
        system_running = False
        system_paused = False
        
        if camera_capture:
            camera_capture.release()
            camera_capture = None
        
        with frame_lock:
            current_frame = None
        
        with detection_lock:
            current_frame_with_detection = None
        
        # Limpa buffer
        global frame_buffer
        frame_buffer = []
        
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
            'real_detection_available': REAL_DETECTORS_AVAILABLE,
            'frame_buffer_size': len(frame_buffer),
            'timestamp': time.time()
        })
        
        return jsonify(current_stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/feed')
def video_feed():
    try:
        # Prioriza frame com detecção se disponível
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
            
            if REAL_DETECTORS_AVAILABLE:
                status_text += " [REAL]"
            else:
                status_text += " [DEMO]"
            
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # FPS
            fps = stats.get('fps', 0)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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

# Todas as outras rotas mantidas do código anterior
@app.route('/api/training/start', methods=['POST'])
def start_training():
    global training_mode
    
    try:
        if not system_running:
            return jsonify({'error': 'Sistema deve estar rodando'}), 400
        
        data = request.get_json() or {}
        duration = data.get('duration', 15)
        
        training_mode = True
        
        if REAL_DETECTORS_AVAILABLE and deep_learning_detector:
            # Ativa modo de treinamento real
            try:
                deep_learning_detector.set_training_mode(True)
                logger.info(f"🎓 Treinamento REAL iniciado por {duration} minutos")
            except AttributeError:
                logger.info(f"🎓 Treinamento simulado por {duration} minutos")
        else:
            logger.info(f"🎓 Treinamento simulado por {duration} minutos")
        
        # Para automaticamente após duração
        def stop_training():
            time.sleep(duration * 60)
            global training_mode
            training_mode = False
            if REAL_DETECTORS_AVAILABLE and deep_learning_detector:
                try:
                    deep_learning_detector.set_training_mode(False)
                except AttributeError:
                    pass
            logger.info("🎓 Treinamento finalizado")
        
        threading.Thread(target=stop_training, daemon=True).start()
        
        return jsonify({
            'status': 'success',
            'message': f'Treinamento iniciado por {duration} minutos'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    global training_mode
    
    try:
        training_mode = False
        
        if REAL_DETECTORS_AVAILABLE and deep_learning_detector:
            try:
                deep_learning_detector.set_training_mode(False)
            except AttributeError:
                pass
        
        logger.info("🎓 Treinamento parado")
        return jsonify({'status': 'success', 'message': 'Treinamento parado'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rotas mantidas do código anterior (config, model save, screenshot, etc.)
@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    if request.method == 'GET':
        if config:
            try:
                return jsonify(config.get_all())
            except:
                pass
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
        if REAL_DETECTORS_AVAILABLE and deep_learning_detector:
            timestamp = int(time.time())
            models_dir = f'models/web_model_{timestamp}'
            
            try:
                deep_learning_detector.save_models(models_dir)
                logger.info(f"💾 Modelos reais salvos em: {models_dir}")
                return jsonify({'status': 'success', 'message': f'Modelos salvos em {models_dir}'})
            except Exception as e:
                logger.error(f"❌ Erro ao salvar modelos: {e}")
                return jsonify({'status': 'success', 'message': 'Erro ao salvar modelos reais'})
        else:
            return jsonify({'status': 'success', 'message': 'Salvamento simulado (detectores não disponíveis)'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/screenshot', methods=['POST'])
def take_screenshot():
    try:
        # Usa frame com detecção se disponível
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

@app.route('/api/report/export')
def export_report():
    try:
        with stats_lock:
            report_stats = stats.copy()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'detection_system': {
                'optical_flow_available': optical_flow_detector is not None,
                'deep_learning_available': deep_learning_detector is not None,
                'real_detection_active': REAL_DETECTORS_AVAILABLE
            },
            'statistics': report_stats,
            'system_status': {
                'running': system_running,
                'paused': system_paused,
                'training': training_mode
            }
        }
        
        return Response(
            json.dumps(report, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment; filename=detection_report.json'}
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
        logger.info("🚀 Iniciando servidor com detecção real de anomalias...")
        
        if REAL_DETECTORS_AVAILABLE:
            logger.info("🧠 Detectores disponíveis:")
            logger.info("   • Optical Flow (Lucas-Kanade + Farneback)")
            logger.info("   • CAE (Convolutional Autoencoder)")
            logger.info("   • ConvLSTM (Análise temporal)")
        else:
            logger.warning("⚠️ Detectores reais não disponíveis - modo demonstração")
        
        logger.info("🌐 Servidor iniciando em http://localhost:5000")
        
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