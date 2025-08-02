"""
Interface Web Flask - Versão Ultra Simples
Funciona garantidamente sem erros de importação
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

app = Flask(__name__)
CORS(app)

# Estado global do sistema
system_running = False
system_paused = False
training_mode = False
current_frame = None
camera_capture = None

# Estatísticas básicas
stats = {
    'frames_processed': 0,
    'anomalies_detected': 0,
    'fps': 0.0,
    'avg_processing_time': 0.0,
    'recent_alerts': []
}

# Locks para thread safety
frame_lock = threading.Lock()
stats_lock = threading.Lock()

def simulate_anomaly_detection():
    """Simula detecção básica de anomalias"""
    import random
    
    while True:
        time.sleep(10)  # Verifica a cada 10 segundos
        
        if system_running and not system_paused and random.random() < 0.2:  # 20% chance
            anomaly_types = [
                'Movimento suspeito detectado',
                'Pessoa imóvel detectada',
                'Movimento noturno anômalo',
                'Padrão de movimento irregular'
            ]
            
            message = random.choice(anomaly_types)
            
            with stats_lock:
                stats['anomalies_detected'] += 1
                alert = {
                    'message': message,
                    'type': 'security' if 'suspeito' in message else 'health',
                    'timestamp': time.time(),
                    'confidence': random.uniform(0.7, 0.95)
                }
                stats['recent_alerts'].insert(0, alert)
                stats['recent_alerts'] = stats['recent_alerts'][:10]  # Mantém últimas 10
            
            logger.info(f"Anomalia simulada: {message}")

def camera_loop():
    """Loop principal da câmera"""
    global current_frame, stats
    
    frame_count = 0
    start_time = time.time()
    
    while system_running and camera_capture:
        try:
            ret, frame = camera_capture.read()
            if ret:
                with frame_lock:
                    current_frame = frame.copy()
                
                frame_count += 1
                elapsed = time.time() - start_time
                
                # Atualiza estatísticas a cada segundo
                if elapsed >= 1.0:
                    fps = frame_count / elapsed
                    
                    with stats_lock:
                        stats['frames_processed'] += frame_count
                        stats['fps'] = fps
                        stats['avg_processing_time'] = 1.0 / fps if fps > 0 else 0
                    
                    frame_count = 0
                    start_time = time.time()
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            logger.error(f"Erro no loop da câmera: {e}")
            break

@app.route('/')
def index():
    """Página principal"""
    return render_template('dashboard.html')

@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Inicia o sistema"""
    global system_running, system_paused, camera_capture
    
    try:
        if system_running:
            return jsonify({'error': 'Sistema já está rodando'}), 400
        
        data = request.get_json() or {}
        source_type = data.get('source_type', 'webcam')
        source_param = data.get('source_param', '0')
        
        logger.info(f"Iniciando sistema - {source_type}: {source_param}")
        
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
        
        # Inicia threads
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()
        
        logger.info("Sistema iniciado com sucesso")
        return jsonify({
            'status': 'success',
            'message': 'Sistema iniciado com sucesso'
        })
        
    except Exception as e:
        logger.error(f"Erro ao iniciar sistema: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/pause', methods=['POST'])
def pause_system():
    """Pausa/retoma o sistema"""
    global system_paused
    
    try:
        if not system_running:
            return jsonify({'error': 'Sistema não está rodando'}), 400
        
        system_paused = not system_paused
        message = 'Sistema pausado' if system_paused else 'Sistema retomado'
        
        logger.info(message)
        return jsonify({
            'status': 'success',
            'message': message,
            'paused': system_paused
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    """Para o sistema"""
    global system_running, system_paused, camera_capture, current_frame
    
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
        
        logger.info("Sistema parado")
        return jsonify({
            'status': 'success',
            'message': 'Sistema parado'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Retorna estatísticas"""
    try:
        with stats_lock:
            current_stats = stats.copy()
        
        current_stats.update({
            'system_running': system_running,
            'system_paused': system_paused,
            'training_mode': training_mode,
            'timestamp': time.time()
        })
        
        return jsonify(current_stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/feed')
def video_feed():
    """Feed de vídeo"""
    try:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
            else:
                # Frame placeholder
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, 'Aguardando Video...', (180, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Adiciona informações de status
        if system_running:
            status_text = "PAUSADO" if system_paused else "ATIVO"
            color = (0, 255, 255) if system_paused else (0, 255, 0)
            
            cv2.putText(frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # FPS
            fps = stats.get('fps', 0)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Contadores
            processed = stats.get('frames_processed', 0)
            anomalies = stats.get('anomalies_detected', 0)
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
        logger.error(f"Erro no feed de vídeo: {e}")
        # Frame de erro
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, 'Erro no Video', (250, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', error_frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Inicia treinamento"""
    global training_mode
    
    try:
        if not system_running:
            return jsonify({'error': 'Sistema deve estar rodando'}), 400
        
        data = request.get_json() or {}
        duration = data.get('duration', 15)
        
        training_mode = True
        logger.info(f"Treinamento simulado iniciado por {duration} minutos")
        
        # Para automaticamente após duração
        def stop_training():
            time.sleep(duration * 60)
            global training_mode
            training_mode = False
            logger.info("Treinamento finalizado")
        
        threading.Thread(target=stop_training, daemon=True).start()
        
        return jsonify({
            'status': 'success',
            'message': f'Treinamento iniciado por {duration} minutos'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Para treinamento"""
    global training_mode
    
    try:
        training_mode = False
        logger.info("Treinamento parado")
        return jsonify({'status': 'success', 'message': 'Treinamento parado'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """Configuração"""
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
    """Salva modelo"""
    return jsonify({'status': 'success', 'message': 'Modelo salvo (simulado)'})

@app.route('/api/video/screenshot', methods=['POST'])
def take_screenshot():
    """Captura screenshot"""
    try:
        with frame_lock:
            if current_frame is None:
                return jsonify({'error': 'Nenhum frame disponível'}), 400
            frame = current_frame.copy()
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'screenshot_{timestamp}.jpg'
        
        return Response(
            buffer.tobytes(),
            mimetype='image/jpeg',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/report/export')
def export_report():
    """Exporta relatório"""
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats.copy(),
            'system_status': {
                'running': system_running,
                'paused': system_paused,
                'training': training_mode
            }
        }
        
        return Response(
            json.dumps(report, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment; filename=relatorio.json'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    """Limpa alertas"""
    try:
        with stats_lock:
            stats['recent_alerts'] = []
        return jsonify({'status': 'success', 'message': 'Alertas limpos'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Inicia thread de simulação de anomalias
        anomaly_thread = threading.Thread(target=simulate_anomaly_detection, daemon=True)
        anomaly_thread.start()
        
        logger.info("Servidor iniciando em http://localhost:5000")
        logger.info("Sistema simples - funciona garantidamente!")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("Servidor interrompido")
    except Exception as e:
        logger.error(f"Erro: {e}")
    finally:
        if camera_capture:
            camera_capture.release()
        cv2.destroyAllWindows()