"""
Interface Web Flask - Sistema de Detecção de Anomalias
Versão independente que funciona sem os outros módulos
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime
from io import BytesIO
import base64
import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request, Response, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuração de logging simples
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("WebApp")

# Configuração global simulada
default_config = {
    "input": {
        "camera_index": 0,
        "resolution": [640, 480],
        "fps": 30
    },
    "processing": {
        "movement_threshold": 0.3,
        "anomaly_threshold": 0.7
    }
}

# Estado do sistema
system_running = False
system_paused = False
current_frame = None
camera_capture = None
latest_stats = {
    'frames_processed': 0,
    'anomalies_detected': 0,
    'avg_processing_time': 0.0,
    'fps': 0.0,
    'optical_flow_detections': 0,
    'deep_learning_detections': 0,
    'recent_alerts': []
}

# Thread-safe locks
frame_lock = threading.Lock()
stats_lock = threading.Lock()

def create_placeholder_frame(text="Aguardando Video..."):
    """Cria frame placeholder"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, text, (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

def update_stats():
    """Atualiza estatísticas simuladas"""
    global latest_stats
    
    with stats_lock:
        if system_running and not system_paused:
            latest_stats['frames_processed'] += 1
            latest_stats['fps'] = min(30, latest_stats['frames_processed'] % 25 + 5)
            latest_stats['avg_processing_time'] = 0.05 + (latest_stats['frames_processed'] % 10) * 0.01

def camera_loop():
    """Loop de captura da câmera"""
    global current_frame, camera_capture
    
    while system_running:
        if camera_capture and not system_paused:
            ret, frame = camera_capture.read()
            if ret:
                with frame_lock:
                    current_frame = frame.copy()
                update_stats()
            else:
                logger.warning("Falha ao capturar frame da câmera")
        
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def index():
    """Página principal do dashboard"""
    return render_template('dashboard.html')

@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Inicia o sistema de detecção"""
    global system_running, system_paused, camera_capture
    
    try:
        data = request.get_json() or {}
        source_type = data.get('source_type', 'webcam')
        source_param = data.get('source_param', '0')
        
        if system_running:
            return jsonify({'error': 'Sistema já está rodando'}), 400
        
        logger.info(f"Iniciando sistema - Fonte: {source_type}, Parâmetro: {source_param}")
        
        # Configura fonte de entrada
        if source_type == 'webcam':
            camera_index = int(source_param) if source_param.isdigit() else 0
            camera_capture = cv2.VideoCapture(camera_index)
            
            if not camera_capture.isOpened():
                return jsonify({'error': f'Não foi possível abrir câmera {camera_index}'}), 400
            
            # Configura resolução
            camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
        elif source_type == 'video':
            camera_capture = cv2.VideoCapture(source_param)
            if not camera_capture.isOpened():
                return jsonify({'error': f'Não foi possível abrir vídeo: {source_param}'}), 400
        
        # Inicia sistema
        system_running = True
        system_paused = False
        
        # Inicia thread da câmera
        camera_thread = threading.Thread(target=camera_loop, daemon=True)
        camera_thread.start()
        
        logger.info("Sistema iniciado com sucesso")
        return jsonify({'status': 'success', 'message': 'Sistema iniciado'})
        
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
        return jsonify({'status': 'success', 'message': message, 'paused': system_paused})
        
    except Exception as e:
        logger.error(f"Erro ao pausar/retomar sistema: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    """Para o sistema"""
    global system_running, system_paused, camera_capture
    
    try:
        if not system_running:
            return jsonify({'error': 'Sistema não está rodando'}), 400
        
        logger.info("Parando sistema...")
        
        # Para sistema
        system_running = False
        system_paused = False
        
        # Libera câmera
        if camera_capture:
            camera_capture.release()
            camera_capture = None
        
        # Limpa frame atual
        with frame_lock:
            current_frame = None
        
        logger.info("Sistema parado com sucesso")
        return jsonify({'status': 'success', 'message': 'Sistema parado'})
        
    except Exception as e:
        logger.error(f"Erro ao parar sistema: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Retorna estatísticas do sistema"""
    try:
        with stats_lock:
            stats = latest_stats.copy()
        
        # Adiciona informações do sistema
        stats.update({
            'system_running': system_running,
            'system_paused': system_paused,
            'timestamp': time.time()
        })
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Erro ao obter estatísticas: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/feed')
def video_feed():
    """Retorna frame atual como imagem"""
    try:
        with frame_lock:
            if current_frame is None:
                frame = create_placeholder_frame("Aguardando Video...")
            else:
                frame = current_frame.copy()
        
        # Adiciona overlay de informações
        if system_running:
            status_text = "PAUSADO" if system_paused else "ATIVO"
            color = (0, 255, 255) if system_paused else (0, 255, 0)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Adiciona FPS
            fps_text = f"FPS: {latest_stats.get('fps', 0):.1f}"
            cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Codifica frame como JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        return Response(
            buffer.tobytes(),
            mimetype='image/jpeg',
            headers={'Cache-Control': 'no-cache, no-store, must-revalidate'}
        )
        
    except Exception as e:
        logger.error(f"Erro ao obter feed de vídeo: {e}")
        # Retorna imagem de erro
        error_frame = create_placeholder_frame("Erro no Video")
        _, buffer = cv2.imencode('.jpg', error_frame)
        return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/api/video/screenshot', methods=['POST'])
def take_screenshot():
    """Captura screenshot do frame atual"""
    try:
        with frame_lock:
            if current_frame is None:
                return jsonify({'error': 'Nenhum frame disponível'}), 400
            
            frame = current_frame.copy()
        
        # Codifica como JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'screenshot_{timestamp}.jpg'
        
        return Response(
            buffer.tobytes(),
            mimetype='image/jpeg',
            headers={
                'Content-Disposition': f'attachment; filename={filename}'
            }
        )
        
    except Exception as e:
        logger.error(f"Erro ao capturar screenshot: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def handle_config():
    """GET: Retorna configuração atual, POST: Salva configuração"""
    try:
        if request.method == 'GET':
            return jsonify(default_config)
        
        elif request.method == 'POST':
            new_config = request.get_json()
            default_config.update(new_config)
            
            logger.info("Configuração atualizada via web interface")
            return jsonify({'status': 'success', 'message': 'Configuração salva'})
            
    except Exception as e:
        logger.error(f"Erro ao gerenciar configuração: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Inicia modo de treinamento (simulado)"""
    try:
        data = request.get_json() or {}
        duration = data.get('duration', 15)
        
        if not system_running:
            return jsonify({'error': 'Sistema deve estar rodando para treinar'}), 400
        
        logger.info(f"Modo de treinamento simulado iniciado por {duration} minutos")
        
        # Adiciona alerta de treinamento
        with stats_lock:
            alert = {
                'message': f'Treinamento iniciado por {duration} minutos',
                'type': 'normal',
                'timestamp': time.time()
            }
            latest_stats['recent_alerts'].insert(0, alert)
        
        return jsonify({
            'status': 'success', 
            'message': f'Treinamento iniciado por {duration} minutos'
        })
        
    except Exception as e:
        logger.error(f"Erro ao iniciar treinamento: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Para modo de treinamento"""
    try:
        logger.info("Modo de treinamento parado")
        
        with stats_lock:
            alert = {
                'message': 'Treinamento interrompido',
                'type': 'normal',
                'timestamp': time.time()
            }
            latest_stats['recent_alerts'].insert(0, alert)
        
        return jsonify({'status': 'success', 'message': 'Treinamento parado'})
        
    except Exception as e:
        logger.error(f"Erro ao parar treinamento: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/save', methods=['POST'])
def save_model():
    """Salva modelos (simulado)"""
    try:
        timestamp = int(time.time())
        model_name = f'model_{timestamp}'
        
        logger.info(f"Modelo simulado salvo: {model_name}")
        
        with stats_lock:
            alert = {
                'message': f'Modelo salvo: {model_name}',
                'type': 'normal',
                'timestamp': time.time()
            }
            latest_stats['recent_alerts'].insert(0, alert)
        
        return jsonify({
            'status': 'success', 
            'message': f'Modelo salvo: {model_name}'
        })
        
    except Exception as e:
        logger.error(f"Erro ao salvar modelo: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/report/export')
def export_report():
    """Exporta relatório do sistema"""
    try:
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'system_status': {
                'running': system_running,
                'paused': system_paused
            },
            'statistics': latest_stats.copy(),
            'configuration': default_config.copy()
        }
        
        report_json = json.dumps(report_data, indent=2, ensure_ascii=False)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'relatorio_anomalias_{timestamp}.json'
        
        return Response(
            report_json,
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename={filename}'
            }
        )
        
    except Exception as e:
        logger.error(f"Erro ao exportar relatório: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/clear', methods=['POST'])
def clear_alerts():
    """Limpa alertas recentes"""
    try:
        with stats_lock:
            latest_stats['recent_alerts'] = []
        
        logger.info("Alertas limpos via web interface")
        return jsonify({'status': 'success', 'message': 'Alertas limpos'})
        
    except Exception as e:
        logger.error(f"Erro ao limpar alertas: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system/status')
def system_status():
    """Retorna status detalhado do sistema"""
    try:
        status = {
            'running': system_running,
            'paused': system_paused,
            'camera_available': camera_capture is not None and camera_capture.isOpened() if camera_capture else False,
            'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Erro ao obter status do sistema: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handler para rotas não encontradas"""
    return jsonify({'error': 'Endpoint não encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handler para erros internos"""
    return jsonify({'error': 'Erro interno do servidor'}), 500

def cleanup_system():
    """Limpa recursos do sistema"""
    global system_running, camera_capture
    
    try:
        logger.info("Limpando recursos do sistema...")
        
        system_running = False
        
        if camera_capture:
            camera_capture.release()
            camera_capture = None
        
        cv2.destroyAllWindows()
        logger.info("Recursos limpos com sucesso")
        
    except Exception as e:
        logger.error(f"Erro na limpeza: {e}")

# Simula detecção de anomalias periódica
def simulate_anomaly_detection():
    """Simula detecção de anomalias"""
    import random
    
    while True:
        time.sleep(30)  # A cada 30 segundos
        
        if system_running and not system_paused and random.random() < 0.3:  # 30% chance
            anomaly_types = [
                ('Movimento suspeito detectado', 'security'),
                ('Pessoa imóvel por longo período', 'health'),
                ('Movimento noturno detectado', 'security'),
                ('Padrão de movimento anômalo', 'health')
            ]
            
            message, alert_type = random.choice(anomaly_types)
            
            with stats_lock:
                latest_stats['anomalies_detected'] += 1
                alert = {
                    'message': message,
                    'type': alert_type,
                    'timestamp': time.time()
                }
                latest_stats['recent_alerts'].insert(0, alert)
                
                # Mantém apenas últimos 10 alertas
                latest_stats['recent_alerts'] = latest_stats['recent_alerts'][:10]
            
            logger.info(f"Anomalia simulada: {message}")

if __name__ == '__main__':
    try:
        # Marca tempo de início
        app.start_time = time.time()
        
        # Inicia thread de simulação de anomalias
        anomaly_thread = threading.Thread(target=simulate_anomaly_detection, daemon=True)
        anomaly_thread.start()
        
        logger.info("Iniciando servidor web em http://localhost:5000")
        logger.info("Interface independente - funciona sem os outros módulos")
        
        # Inicia servidor Flask
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.info("Servidor interrompido pelo usuário")
    except Exception as e:
        logger.error(f"Erro ao iniciar servidor: {e}")
    finally:
        cleanup_system()