"""
Interface Web para Sistema de Detecção de Anomalias
Dashboard intuitivo e moderno com controles em tempo real
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit
import cv2
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
import base64
import numpy as np

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.input_manager import InputManager
from src.core.processing_engine import ProcessingEngine
from src.core.output_manager import OutputManager
from src.utils.config import Config
from src.utils.logger import logger

app = Flask(__name__)
app.config['SECRET_KEY'] = 'anomaly_detection_system_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Sistema global
system_instance = None
system_thread = None
system_running = False

class WebSystemManager:
    """Gerenciador do sistema para interface web"""
    
    def __init__(self):
        self.config = Config()
        self.input_manager = InputManager(self.config)
        self.processing_engine = ProcessingEngine(self.config)
        self.output_manager = OutputManager(self.config)
        
        self.current_frame = None
        self.current_results = {}
        self.system_stats = {}
        
        # Configurar callbacks
        self._setup_callbacks()
        
        logger.info("WebSystemManager inicializado")
    
    def _setup_callbacks(self):
        """Configura callbacks para interface web"""
        
        def frame_callback(frame_data):
            self.processing_engine.add_frame_to_queue(frame_data)
        
        def result_callback(results, frame_data):
            self.current_frame = frame_data["frame"]
            self.current_results = results
            
            # Enviar dados para clientes web
            self._emit_update(results, frame_data)
        
        def anomaly_callback(anomaly, results, frame_data):
            # Enviar alerta para interface web
            alert_data = {
                "type": "anomaly_alert",
                "anomaly": anomaly,
                "timestamp": datetime.now().isoformat(),
                "frame_id": results.get("frame_id")
            }
            socketio.emit('anomaly_alert', alert_data)
        
        self.input_manager.add_frame_callback(frame_callback)
        self.processing_engine.add_result_callback(result_callback)
        self.processing_engine.add_anomaly_callback(anomaly_callback)
    
    def _emit_update(self, results, frame_data):
        """Emite atualização para clientes web"""
        update_data = {
            "timestamp": datetime.now().isoformat(),
            "frame_id": results.get("frame_id"),
            "has_movement": results.get("optical_flow", {}).get("has_movement", False),
            "anomalies_count": len(results.get("classified_anomalies", [])),
            "risk_level": results.get("final_assessment", {}).get("risk_level", "low"),
            "processing_time": results.get("processing_time", 0),
            "stats": self.get_system_stats()
        }
        
        socketio.emit('system_update', update_data)
    
    def start_webcam(self, camera_index=0):
        """Inicia detecção por webcam"""
        try:
            if self.input_manager.initialize_webcam(camera_index):
                self.input_manager.start_capture()
                self.processing_engine.start_processing()
                logger.info(f"Webcam {camera_index} iniciada via web")
                return True
        except Exception as e:
            logger.error(f"Erro ao iniciar webcam via web: {e}")
        return False
    
    def start_video_file(self, file_path, loop=False):
        """Inicia análise de arquivo"""
        try:
            if self.input_manager.initialize_video_file(file_path, loop):
                self.input_manager.start_capture()
                self.processing_engine.start_processing()
                logger.info(f"Arquivo {file_path} iniciado via web")
                return True
        except Exception as e:
            logger.error(f"Erro ao iniciar arquivo via web: {e}")
        return False
    
    def stop_system(self):
        """Para o sistema"""
        try:
            self.processing_engine.stop_processing()
            self.input_manager.stop_capture()
            logger.info("Sistema parado via web")
            return True
        except Exception as e:
            logger.error(f"Erro ao parar sistema via web: {e}")
        return False
    
    def get_current_frame_encoded(self):
        """Retorna frame atual codificado para web"""
        if self.current_frame is not None:
            try:
                # Redimensionar para web
                frame_resized = cv2.resize(self.current_frame, (640, 480))
                
                # Codificar como JPEG
                _, buffer = cv2.imencode('.jpg', frame_resized, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Converter para base64
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                return f"data:image/jpeg;base64,{frame_base64}"
            except Exception as e:
                logger.error(f"Erro ao codificar frame: {e}")
        return None
    
    def get_system_stats(self):
        """Retorna estatísticas do sistema"""
        return {
            "input": self.input_manager.get_stats(),
            "processing": self.processing_engine.get_performance_stats(),
            "output": self.output_manager.get_display_stats()
        }
    
    def get_recent_alerts(self, hours=24):
        """Retorna alertas recentes"""
        return self.output_manager.get_recent_alerts(hours)

# Rotas principais
@app.route('/')
def index():
    """Página principal"""
    return render_template('dashboard.html')

@app.route('/api/system/status')
def system_status():
    """Status do sistema"""
    if system_instance:
        return jsonify({
            "running": system_running,
            "stats": system_instance.get_system_stats(),
            "timestamp": datetime.now().isoformat()
        })
    return jsonify({"running": False})

@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Iniciar sistema"""
    global system_instance, system_running
    
    try:
        data = request.get_json()
        source_type = data.get('source_type', 'webcam')
        
        if not system_instance:
            system_instance = WebSystemManager()
        
        if source_type == 'webcam':
            camera_index = data.get('camera_index', 0)
            success = system_instance.start_webcam(camera_index)
        elif source_type == 'video':
            file_path = data.get('file_path', '')
            loop = data.get('loop', False)
            success = system_instance.start_video_file(file_path, loop)
        else:
            return jsonify({"success": False, "error": "Tipo de fonte inválido"})
        
        system_running = success
        return jsonify({"success": success})
        
    except Exception as e:
        logger.error(f"Erro ao iniciar sistema via API: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    """Parar sistema"""
    global system_running
    
    try:
        if system_instance:
            success = system_instance.stop_system()
            system_running = False
            return jsonify({"success": success})
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/frame/current')
def current_frame():
    """Frame atual"""
    if system_instance:
        frame_data = system_instance.get_current_frame_encoded()
        if frame_data:
            return jsonify({"frame": frame_data})
    return jsonify({"frame": None})

@app.route('/api/alerts/recent')
def recent_alerts():
    """Alertas recentes"""
    hours = request.args.get('hours', 24, type=int)
    if system_instance:
        alerts = system_instance.get_recent_alerts(hours)
        return jsonify({"alerts": alerts})
    return jsonify({"alerts": []})

@app.route('/api/config/get')
def get_config():
    """Obter configuração atual"""
    config = Config()
    return jsonify({
        "video": config.video.__dict__,
        "model": config.model.__dict__,
        "system": config.system.__dict__
    })

@app.route('/api/config/update', methods=['POST'])
def update_config():
    """Atualizar configuração"""
    try:
        data = request.get_json()
        config = Config()
        
        # Atualizar thresholds se fornecidos
        if 'thresholds' in data:
            config.update_thresholds(**data['thresholds'])
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# Eventos WebSocket
@socketio.on('connect')
def handle_connect():
    """Cliente conectado"""
    logger.info("Cliente web conectado")
    emit('connected', {"message": "Conectado ao sistema de detecção"})

@socketio.on('disconnect')
def handle_disconnect():
    """Cliente desconectado"""
    logger.info("Cliente web desconectado")

@socketio.on('request_frame')
def handle_frame_request():
    """Solicitação de frame"""
    if system_instance:
        frame_data = system_instance.get_current_frame_encoded()
        if frame_data:
            emit('frame_update', {"frame": frame_data})

@socketio.on('request_stats')
def handle_stats_request():
    """Solicitação de estatísticas"""
    if system_instance:
        stats = system_instance.get_system_stats()
        emit('stats_update', {"stats": stats})

def run_web_server(host='127.0.0.1', port=5000, debug=False):
    """Executa servidor web"""
    logger.info(f"Iniciando servidor web em http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Verificar se templates existem
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
        logger.warning("Diretório templates criado - adicione dashboard.html")
    
    run_web_server(debug=True)