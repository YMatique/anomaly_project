"""
Gerenciador de saída - alertas, visualizações e interface
Coordena todas as formas de comunicação com o usuário
"""

import cv2
import numpy as np
import threading
import queue
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Callable, Union
from datetime import datetime
from collections import deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

from ..utils.helpers import FileManager, alert_manager, performance_monitor
from ..utils.logger import logger

class VisualizationRenderer:
    """
    Renderizador de visualizações para o sistema
    Cria overlays informativos nos frames
    """
    
    def __init__(self, config):
        self.config = config
        
        # Configurações de cores
        self.colors = {
            "normal": (0, 255, 0),      # Verde
            "warning": (0, 255, 255),   # Amarelo
            "danger": (0, 0, 255),      # Vermelho
            "info": (255, 255, 255),    # Branco
            "background": (0, 0, 0),    # Preto
            "overlay": (50, 50, 50)     # Cinza escuro
        }
        
        # Configurações de texto
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        # Estado da visualização
        self.show_info_panel = True
        self.show_flow_overlay = True
        self.show_anomaly_markers = True
        self.show_statistics = True
        
    def render_frame(self, frame: np.ndarray, results: Dict, 
                    frame_data: Dict) -> np.ndarray:
        """
        Renderiza frame com todas as visualizações
        
        Args:
            frame: Frame original
            results: Resultados do processamento
            frame_data: Metadados do frame
            
        Returns:
            Frame com visualizações
        """
        if frame is None:
            return None
        
        vis_frame = frame.copy()
        
        # Renderizar optical flow
        if self.show_flow_overlay:
            vis_frame = self._render_optical_flow(vis_frame, results.get("optical_flow", {}))
        
        # Renderizar marcadores de anomalia
        if self.show_anomaly_markers:
            vis_frame = self._render_anomaly_markers(vis_frame, results)
        
        # Renderizar painel de informações
        if self.show_info_panel:
            vis_frame = self._render_info_panel(vis_frame, results, frame_data)
        
        # Renderizar estatísticas
        if self.show_statistics:
            vis_frame = self._render_statistics(vis_frame)
        
        return vis_frame
    
    def _render_optical_flow(self, frame: np.ndarray, optical_flow_result: Dict) -> np.ndarray:
        """Renderiza visualização do optical flow"""
        
        if not optical_flow_result.get("has_movement", False):
            return frame
        
        method = optical_flow_result.get("method", "")
        
        if method == "lucas_kanade" and "tracking_points" in optical_flow_result:
            # Renderizar pontos de tracking
            points = optical_flow_result["tracking_points"]
            vectors = optical_flow_result.get("movement_vectors", [])
            
            if len(points) > 0:
                for i, point in enumerate(points):
                    x, y = int(point[0]), int(point[1])
                    
                    # Cor baseada na magnitude do movimento
                    if i < len(vectors):
                        magnitude = np.sqrt(vectors[i][0]**2 + vectors[i][1]**2)
                        if magnitude > 10:
                            color = self.colors["danger"]
                        elif magnitude > 5:
                            color = self.colors["warning"]
                        else:
                            color = self.colors["normal"]
                    else:
                        color = self.colors["normal"]
                    
                    # Desenhar ponto
                    cv2.circle(frame, (x, y), 3, color, -1)
                    
                    # Desenhar vetor se significativo
                    if i < len(vectors):
                        dx, dy = int(vectors[i][0] * 3), int(vectors[i][1] * 3)
                        if abs(dx) > 2 or abs(dy) > 2:
                            cv2.arrowedLine(frame, (x, y), (x + dx, y + dy), color, 2)
        
        elif method == "farneback" and "magnitude_field" in optical_flow_result:
            # Renderizar dense flow como heatmap
            magnitude = optical_flow_result["magnitude_field"]
            
            # Normalizar magnitude para visualização
            norm_magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Aplicar colormap
            heatmap = cv2.applyColorMap(norm_magnitude, cv2.COLORMAP_JET)
            
            # Criar máscara para pixels com movimento significativo
            movement_mask = magnitude > optical_flow_result.get("threshold", 2.0)
            
            # Aplicar overlay apenas onde há movimento
            frame[movement_mask] = cv2.addWeighted(
                frame[movement_mask], 0.7, heatmap[movement_mask], 0.3, 0
            )
        
        return frame
    
    def _render_anomaly_markers(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """Renderiza marcadores de anomalias detectadas"""
        
        classified_anomalies = results.get("classified_anomalies", [])
        optical_flow_anomalies = results.get("optical_flow", {}).get("anomalies", [])
        final_assessment = results.get("final_assessment", {})
        
        # Renderizar anomalias classificadas
        if classified_anomalies:
            # Banner superior para anomalias críticas
            critical_anomalies = [a for a in classified_anomalies if a["confidence"] > 0.8]
            
            if critical_anomalies:
                self._draw_alert_banner(frame, critical_anomalies[0])
            
            # Marcadores laterais para todas as anomalias
            y_offset = 100
            for anomaly in classified_anomalies:
                self._draw_anomaly_marker(frame, anomaly, y_offset)
                y_offset += 40
        
        # Indicador de movimento geral
        if results.get("optical_flow", {}).get("has_movement", False):
            movement_magnitude = results.get("optical_flow", {}).get("movement_magnitude", 0.0)
            self._draw_movement_indicator(frame, movement_magnitude)
        
        # Status geral do sistema
        self._draw_system_status(frame, final_assessment)
        
        return frame
    
    def _draw_alert_banner(self, frame: np.ndarray, anomaly: Dict):
        """Desenha banner de alerta no topo"""
        h, w = frame.shape[:2]
        
        # Cor baseada na categoria
        if anomaly["category"] == "health":
            bg_color = (0, 0, 200)  # Vermelho escuro
            text_color = (255, 255, 255)
        else:  # security
            bg_color = (0, 100, 200)  # Laranja escuro
            text_color = (255, 255, 255)
        
        # Desenhar fundo do banner
        cv2.rectangle(frame, (0, 0), (w, 60), bg_color, -1)
        
        # Texto do alerta
        alert_text = f"⚠ ALERTA: {anomaly['description'].upper()}"
        confidence_text = f"Confiança: {anomaly['confidence']:.0%}"
        
        # Desenhar textos
        cv2.putText(frame, alert_text, (10, 25), self.font, 0.8, text_color, 2)
        cv2.putText(frame, confidence_text, (10, 50), self.font, 0.6, text_color, 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (w - 100, 25), self.font, 0.6, text_color, 2)
    
    def _draw_anomaly_marker(self, frame: np.ndarray, anomaly: Dict, y_pos: int):
        """Desenha marcador lateral para anomalia"""
        h, w = frame.shape[:2]
        
        # Cor baseada na confiança
        confidence = anomaly["confidence"]
        if confidence > 0.8:
            color = self.colors["danger"]
        elif confidence > 0.6:
            color = self.colors["warning"]
        else:
            color = self.colors["info"]
        
        # Desenhar marcador
        cv2.circle(frame, (w - 30, y_pos), 8, color, -1)
        cv2.circle(frame, (w - 30, y_pos), 8, (0, 0, 0), 2)
        
        # Texto da anomalia
        anomaly_text = f"{anomaly['type']} ({confidence:.0%})"
        text_size = cv2.getTextSize(anomaly_text, self.font, 0.5, 1)[0]
        
        # Fundo do texto
        cv2.rectangle(frame, 
                     (w - text_size[0] - 50, y_pos - 12), 
                     (w - 40, y_pos + 8), 
                     self.colors["overlay"], -1)
        
        # Texto
        cv2.putText(frame, anomaly_text, (w - text_size[0] - 45, y_pos + 3), 
                   self.font, 0.5, color, 1)
    
    def _draw_movement_indicator(self, frame: np.ndarray, magnitude: float):
        """Desenha indicador de movimento"""
        h, w = frame.shape[:2]
        
        # Posição do indicador
        center = (50, h - 50)
        radius = 20
        
        # Cor baseada na magnitude
        if magnitude > 10:
            color = self.colors["danger"]
        elif magnitude > 5:
            color = self.colors["warning"]
        else:
            color = self.colors["normal"]
        
        # Desenhar círculo de movimento
        cv2.circle(frame, center, radius, color, 3)
        
        # Barra de magnitude
        bar_height = int((magnitude / 20.0) * 30)  # Max 30 pixels
        bar_y = center[1] + radius + 10
        cv2.rectangle(frame, 
                     (center[0] - 10, bar_y), 
                     (center[0] + 10, bar_y + bar_height), 
                     color, -1)
        
        # Texto
        cv2.putText(frame, f"{magnitude:.1f}", 
                   (center[0] - 15, center[1] + radius + 50), 
                   self.font, 0.4, color, 1)
    
    def _draw_system_status(self, frame: np.ndarray, final_assessment: Dict):
        """Desenha status geral do sistema"""
        h, w = frame.shape[:2]
        
        risk_level = final_assessment.get("risk_level", "low")
        has_anomaly = final_assessment.get("has_anomaly", False)
        
        # Cor do status
        if risk_level == "critical":
            status_color = self.colors["danger"]
            status_text = "CRÍTICO"
        elif risk_level == "high":
            status_color = (0, 165, 255)  # Laranja
            status_text = "ALTO"
        elif risk_level == "medium":
            status_color = self.colors["warning"]
            status_text = "MÉDIO"
        else:
            status_color = self.colors["normal"]
            status_text = "NORMAL"
        
        # Desenhar status
        status_pos = (10, h - 30)
        cv2.putText(frame, f"Status: {status_text}", status_pos, 
                   self.font, 0.6, status_color, 2)
    
    def _render_info_panel(self, frame: np.ndarray, results: Dict, frame_data: Dict) -> np.ndarray:
        """Renderiza painel de informações"""
        h, w = frame.shape[:2]
        
        # Posição do painel (canto superior direito)
        panel_width = 300
        panel_height = 150
        panel_x = w - panel_width - 10
        panel_y = 70
        
        # Fundo do painel
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     self.colors["overlay"], -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Borda do painel
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     self.colors["info"], 2)
        
        # Informações do frame
        info_lines = [
            f"Frame ID: {results.get('frame_id', 'N/A')}",
            f"Timestamp: {datetime.fromtimestamp(results.get('timestamp', 0)).strftime('%H:%M:%S')}",
            f"Processamento: {results.get('processing_time', 0):.3f}s",
            f"FPS: {performance_monitor.get_stats()['fps']['current']:.1f}",
            f"Movimento: {'Sim' if results.get('optical_flow', {}).get('has_movement') else 'Não'}",
            f"Anomalias: {len(results.get('classified_anomalies', []))}"
        ]
        
        # Desenhar informações
        y_offset = panel_y + 20
        for line in info_lines:
            cv2.putText(frame, line, (panel_x + 10, y_offset), 
                       self.font, 0.4, self.colors["info"], 1)
            y_offset += 20
        
        return frame
    
    def _render_statistics(self, frame: np.ndarray) -> np.ndarray:
        """Renderiza estatísticas do sistema"""
        h, w = frame.shape[:2]
        
        # Posição das estatísticas (canto inferior direito)
        stats_x = w - 200
        stats_y = h - 100
        
        # Obter estatísticas
        perf_stats = performance_monitor.get_stats()
        
        stats_lines = [
            f"FPS Médio: {perf_stats['fps']['average']:.1f}",
            f"Memória: {perf_stats['memory']['current']:.1f}%",
            f"Proc.Time: {perf_stats['processing_time']['average']:.3f}s"
        ]
        
        # Desenhar estatísticas
        y_offset = stats_y
        for line in stats_lines:
            # Fundo do texto
            text_size = cv2.getTextSize(line, self.font, 0.4, 1)[0]
            cv2.rectangle(frame, (stats_x - 5, y_offset - 15), 
                         (stats_x + text_size[0] + 5, y_offset + 5), 
                         self.colors["overlay"], -1)
            
            # Texto
            cv2.putText(frame, line, (stats_x, y_offset), 
                       self.font, 0.4, self.colors["info"], 1)
            y_offset += 20
        
        return frame
    
    def toggle_info_panel(self):
        """Alterna exibição do painel de informações"""
        self.show_info_panel = not self.show_info_panel
        logger.info(f"Painel de informações: {'ON' if self.show_info_panel else 'OFF'}")
    
    def toggle_flow_overlay(self):
        """Alterna overlay de optical flow"""
        self.show_flow_overlay = not self.show_flow_overlay
        logger.info(f"Overlay de flow: {'ON' if self.show_flow_overlay else 'OFF'}")
    
    def toggle_anomaly_markers(self):
        """Alterna marcadores de anomalia"""  
        self.show_anomaly_markers = not self.show_anomaly_markers
        logger.info(f"Marcadores de anomalia: {'ON' if self.show_anomaly_markers else 'OFF'}")

class AlertSystem:
    """
    Sistema de alertas multi-canal
    Suporta console, email, webhook e arquivo
    """
    
    def __init__(self, config):
        self.config = config
        self.alert_channels = []
        
        # Configurações de email (se disponível)
        self.smtp_config = {
            "enabled": False,
            "server": "",
            "port": 587,
            "username": "",
            "password": "",
            "recipients": []
        }
        
        # Configurações de webhook (se disponível)
        self.webhook_config = {
            "enabled": False,
            "url": "",
            "headers": {}
        }
        
        # Histórico de alertas
        self.alert_history = deque(maxlen=1000)
        
        logger.info("AlertSystem inicializado")
    
    def configure_email(self, server: str, port: int, username: str, 
                       password: str, recipients: List[str]):
        """Configura alertas por email"""
        self.smtp_config.update({
            "enabled": True,
            "server": server,
            "port": port,
            "username": username,
            "password": password,
            "recipients": recipients
        })
        logger.info(f"Email configurado - {len(recipients)} destinatários")
    
    def configure_webhook(self, url: str, headers: Dict[str, str] = None):
        """Configura alertas por webhook"""
        self.webhook_config.update({
            "enabled": True,
            "url": url,
            "headers": headers or {}
        })
        logger.info(f"Webhook configurado: {url}")
    
    def send_alert(self, anomaly: Dict, results: Dict, frame_data: Dict, 
                  frame_path: str = None):
        """
        Envia alerta através de todos os canais configurados
        
        Args:
            anomaly: Dados da anomalia
            results: Resultados do processamento
            frame_data: Metadados do frame
            frame_path: Caminho para imagem do frame (opcional)
        """
        alert_data = {
            "timestamp": datetime.now().isoformat(),
            "anomaly": anomaly,
            "frame_id": results.get("frame_id"),
            "risk_level": results.get("final_assessment", {}).get("risk_level", "unknown"),
            "frame_path": frame_path
        }
        
        # Adicionar ao histórico
        self.alert_history.append(alert_data)
        
        # Enviar por todos os canais
        if self.smtp_config["enabled"]:
            self._send_email_alert(alert_data)
        
        if self.webhook_config["enabled"]:
            self._send_webhook_alert(alert_data)
        
        # Log local
        logger.warning(f"ALERTA ENVIADO: {anomaly['description']} - "
                      f"Confiança: {anomaly['confidence']:.0%} - "
                      f"Risco: {alert_data['risk_level'].upper()}")
    
    def _send_email_alert(self, alert_data: Dict):
        """Envia alerta por email"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.smtp_config["username"]
            msg['To'] = ", ".join(self.smtp_config["recipients"])
            msg['Subject'] = f"🚨 Alerta de Segurança - {alert_data['anomaly']['type'].title()}"
            
            # Corpo do email
            body = f"""
ALERTA DE ANOMALIA DETECTADA

Tipo: {alert_data['anomaly']['description']}
Categoria: {alert_data['anomaly']['category'].title()}
Confiança: {alert_data['anomaly']['confidence']:.0%}
Nivel de Risco: {alert_data['risk_level'].upper()}
Timestamp: {alert_data['timestamp']}
Frame ID: {alert_data['frame_id']}

Evidências:
{json.dumps(alert_data['anomaly'].get('supporting_evidence', {}), indent=2)}

---
Sistema de Detecção de Anomalias
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Anexar imagem se disponível
            if alert_data.get('frame_path') and os.path.exists(alert_data['frame_path']):
                with open(alert_data['frame_path'], 'rb') as f:
                    img_data = f.read()
                    image = MIMEImage(img_data)
                    image.add_header('Content-Disposition', 'attachment', filename='alert_frame.jpg')
                    msg.attach(image)
            
            # Enviar email
            server = smtplib.SMTP(self.smtp_config["server"], self.smtp_config["port"])
            server.starttls()
            server.login(self.smtp_config["username"], self.smtp_config["password"])
            
            text = msg.as_string()
            server.sendmail(self.smtp_config["username"], 
                          self.smtp_config["recipients"], text)
            server.quit()
            
            logger.info("Email de alerta enviado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao enviar email de alerta: {e}")
    
    def _send_webhook_alert(self, alert_data: Dict):
        """Envia alerta por webhook"""
        try:
            import requests
            
            payload = {
                "type": "anomaly_alert",
                "timestamp": alert_data["timestamp"],
                "anomaly_type": alert_data["anomaly"]["type"],
                "category": alert_data["anomaly"]["category"],
                "confidence": alert_data["anomaly"]["confidence"],
                "risk_level": alert_data["risk_level"],
                "description": alert_data["anomaly"]["description"],
                "frame_id": alert_data["frame_id"]
            }
            
            response = requests.post(
                self.webhook_config["url"],
                json=payload,
                headers=self.webhook_config["headers"],
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Webhook de alerta enviado com sucesso")
            else:
                logger.warning(f"Webhook retornou status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Erro ao enviar webhook de alerta: {e}")
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Retorna alertas recentes"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        recent_alerts = []
        for alert in self.alert_history:
            alert_time = datetime.fromisoformat(alert["timestamp"]).timestamp()
            if alert_time > cutoff_time:
                recent_alerts.append(alert)
        
        return recent_alerts

class OutputManager:
    """
    Gerenciador principal de saída
    Coordena visualizações, alertas e interface
    """
    
    def __init__(self, config):
        self.config = config
        
        # Componentes
        self.visualization = VisualizationRenderer(config)
        self.alert_system = AlertSystem(config)
        
        # Estado da interface
        self.display_active = False
        self.recording_active = False
        self.current_frame = None
        
        # Controle de display
        self.window_name = "Sistema de Detecção de Anomalias"
        self.window_created = False
        
        # Threading para display
        self.display_thread = None
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Gravação de vídeo
        self.video_writer = None
        self.recording_path = None
        
        # Callbacks
        self.key_callbacks = {}  # tecla -> callback
        self.mouse_callbacks = []
        
        # Estatísticas
        self.display_stats = {
            "frames_displayed": 0,
            "frames_saved": 0,
            "alerts_sent": 0
        }
        
        logger.info("OutputManager inicializado")
    
    def start_display(self, window_size: Tuple[int, int] = None) -> bool:
        """
        Inicia display visual
        
        Args:
            window_size: Tamanho da janela (width, height)
        """
        if self.display_active:
            logger.warning("Display já está ativo")
            return True
        
        logger.info("Iniciando display visual")
        
        # Criar janela
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        if window_size:
            cv2.resizeWindow(self.window_name, window_size[0], window_size[1])
        
        # Configurar callbacks de mouse
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        self.window_created = True
        self.display_active = True
        
        # Iniciar thread de display
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
        # Registrar teclas padrão
        self._setup_default_keys()
        
        logger.info("Display visual iniciado")
        return True
    
    def stop_display(self):
        """Para display visual"""
        if not self.display_active:
            return
        
        logger.info("Parando display visual")
        
        self.display_active = False
        
        # Aguardar thread terminar
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
        
        # Fechar janela
        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False
        
        # Parar gravação se ativa
        self.stop_recording()
        
        logger.info("Display visual parado")
    
    def display_frame(self, frame: np.ndarray, results: Dict, frame_data: Dict):
        """
        Adiciona frame para display
        
        Args:
            frame: Frame original
            results: Resultados do processamento
            frame_data: Metadados do frame
        """
        if not self.display_active:
            return
        
        # Renderizar visualizações
        vis_frame = self.visualization.render_frame(frame, results, frame_data)
        
        if vis_frame is not None:
            self.current_frame = vis_frame.copy()
            
            # Adicionar à queue de display
            try:
                display_data = {
                    "frame": vis_frame,
                    "results": results,
                    "frame_data": frame_data
                }
                self.frame_queue.put_nowait(display_data)
            except queue.Full:
                # Queue cheia - descartar frame mais antigo
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(display_data)
                except queue.Empty:
                    pass
    
    def _display_loop(self):
        """Loop principal de display (thread separada)"""
        logger.info("Thread de display iniciada")
        
        while self.display_active:
            try:
                # Obter próximo frame
                display_data = self.frame_queue.get(timeout=1.0)
                
                vis_frame = display_data["frame"]
                results = display_data["results"]
                
                # Exibir frame
                cv2.imshow(self.window_name, vis_frame)
                
                # Processar eventos de teclado
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Alguma tecla foi pressionada
                    self._handle_key_press(key, display_data)
                
                # Gravar se recording ativo
                if self.recording_active and self.video_writer:
                    self.video_writer.write(vis_frame)
                
                # Processar alertas
                self._process_alerts(results, display_data)
                
                # Atualizar estatísticas
                self.display_stats["frames_displayed"] += 1
                
            except queue.Empty:
                # Timeout - continuar
                continue
            except Exception as e:
                logger.error(f"Erro no loop de display: {e}")
                time.sleep(0.1)
        
        logger.info("Thread de display finalizada")
    
    def _setup_default_keys(self):
        """Configura teclas padrão"""
        self.key_callbacks.update({
            ord('q'): lambda data: self.stop_display(),
            ord('s'): lambda data: self._save_current_frame(),
            ord('r'): lambda data: self._toggle_recording(),
            ord('i'): lambda data: self.visualization.toggle_info_panel(),
            ord('f'): lambda data: self.visualization.toggle_flow_overlay(),
            ord('a'): lambda data: self.visualization.toggle_anomaly_markers(),
            27: lambda data: self.stop_display()  # ESC
        })
    
    def _handle_key_press(self, key: int, display_data: Dict):
        """Processa tecla pressionada"""
        if key in self.key_callbacks:
            try:
                self.key_callbacks[key](display_data)
            except Exception as e:
                logger.error(f"Erro no callback de tecla {key}: {e}")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Callback para eventos de mouse"""
        for callback in self.mouse_callbacks:
            try:
                callback(event, x, y, flags, param)
            except Exception as e:
                logger.error(f"Erro no callback de mouse: {e}")
    
    def _save_current_frame(self):
        """Salva frame atual"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"manual_capture_{timestamp}.jpg"
            save_path = FileManager.save_frame(self.current_frame, filename, "data/captures/")
            
            if save_path:
                logger.info(f"Frame salvo: {save_path}")
                self.display_stats["frames_saved"] += 1
            else:
                logger.error("Erro ao salvar frame")
    
    def _toggle_recording(self):
        """Alterna gravação de vídeo"""
        if self.recording_active:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self, output_path: str = None) -> bool:
        """
        Inicia gravação de vídeo
        
        Args:
            output_path: Caminho de saída (gera automaticamente se None)
        """
        if self.recording_active:
            logger.warning("Gravação já está ativa")
            return True
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/recordings/recording_{timestamp}.mp4"
        
        # Garantir que diretório existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Configurar codec e writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30.0
        
        # Obter dimensões do frame atual
        if self.current_frame is not None:
            height, width = self.current_frame.shape[:2]
        else:
            width, height = 640, 480  # Padrão
        
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if self.video_writer.isOpened():
            self.recording_active = True
            self.recording_path = output_path
            logger.info(f"Gravação iniciada: {output_path}")
            return True
        else:
            logger.error("Falha ao iniciar gravação")
            return False
    
    def stop_recording(self):
        """Para gravação de vídeo"""
        if not self.recording_active:
            return
        
        self.recording_active = False
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        logger.info(f"Gravação parada: {self.recording_path}")
        self.recording_path = None
    
    def _process_alerts(self, results: Dict, display_data: Dict):
        """Processa alertas dos resultados"""
        classified_anomalies = results.get("classified_anomalies", [])
        
        for anomaly in classified_anomalies:
            # Verificar se deve emitir alerta
            alert_type = f"{anomaly['category']}_{anomaly['type']}"
            
            if alert_manager.should_alert(alert_type):
                # Salvar frame de alerta
                frame_path = None
                if self.config.system.save_alert_frames:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"alert_{alert_type}_{timestamp}.jpg"
                    frame_path = FileManager.save_frame(
                        display_data["frame"], filename, "data/alerts/"
                    )
                
                # Enviar alerta
                self.alert_system.send_alert(anomaly, results, display_data["frame_data"], frame_path)
                self.display_stats["alerts_sent"] += 1
    
    def add_key_callback(self, key: Union[int, str], callback: Callable):
        """
        Adiciona callback para tecla
        
        Args:
            key: Código da tecla ou caractere
            callback: Função callback que recebe display_data
        """
        if isinstance(key, str):
            key = ord(key.lower())
        
        self.key_callbacks[key] = callback
        logger.info(f"Callback de tecla adicionado: {key}")
    
    def add_mouse_callback(self, callback: Callable):
        """Adiciona callback para mouse"""
        self.mouse_callbacks.append(callback)
        logger.info(f"Callback de mouse adicionado - total: {len(self.mouse_callbacks)}")
    
    def configure_email_alerts(self, server: str, port: int, username: str, 
                             password: str, recipients: List[str]):
        """Configura alertas por email"""
        self.alert_system.configure_email(server, port, username, password, recipients)
    
    def configure_webhook_alerts(self, url: str, headers: Dict[str, str] = None):
        """Configura alertas por webhook"""
        self.alert_system.configure_webhook(url, headers)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Retorna alertas recentes"""
        return self.alert_system.get_recent_alerts(hours)
    
    def export_session_report(self, output_path: str = None) -> str:
        """
        Exporta relatório da sessão
        
        Args:
            output_path: Caminho de saída (gera automaticamente se None)
            
        Returns:
            Caminho do arquivo gerado
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/reports/session_report_{timestamp}.json"
        
        # Garantir que diretório existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Coletar dados do relatório
        report_data = {
            "session_info": {
                "timestamp": datetime.now().isoformat(),
                "duration": time.time() - (getattr(self, 'session_start_time', time.time())),
                "display_active": self.display_active,
                "recording_active": self.recording_active
            },
            "statistics": {
                "display_stats": self.display_stats,
                "performance_stats": performance_monitor.get_stats()
            },
            "recent_alerts": self.get_recent_alerts(24),
            "anomaly_summary": self._generate_anomaly_summary(),
            "system_config": {
                "video_resolution": self.config.video.realtime_resolution,
                "fps": self.config.video.fps,
                "detection_thresholds": {
                    "optical_flow": self.config.model.flow_threshold,
                    "anomaly": self.config.model.anomaly_threshold
                }
            }
        }
        
        # Salvar relatório
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Relatório da sessão exportado: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Erro ao exportar relatório: {e}")
            return None
    
    def _generate_anomaly_summary(self) -> Dict:
        """Gera resumo das anomalias detectadas"""
        recent_alerts = self.get_recent_alerts(24)
        
        summary = {
            "total_anomalies": len(recent_alerts),
            "by_category": {},
            "by_type": {},
            "by_risk_level": {},
            "hourly_distribution": {},
            "average_confidence": 0.0
        }
        
        if recent_alerts:
            # Análise por categoria
            for alert in recent_alerts:
                category = alert["anomaly"]["category"]
                anomaly_type = alert["anomaly"]["type"]
                risk_level = alert["risk_level"]
                
                summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
                summary["by_type"][anomaly_type] = summary["by_type"].get(anomaly_type, 0) + 1
                summary["by_risk_level"][risk_level] = summary["by_risk_level"].get(risk_level, 0) + 1
                
                # Distribuição por hora
                alert_time = datetime.fromisoformat(alert["timestamp"])
                hour_key = alert_time.strftime("%H:00")
                summary["hourly_distribution"][hour_key] = summary["hourly_distribution"].get(hour_key, 0) + 1
            
            # Confiança média
            total_confidence = sum(alert["anomaly"]["confidence"] for alert in recent_alerts)
            summary["average_confidence"] = total_confidence / len(recent_alerts)
        
        return summary
    
    def create_dashboard_data(self) -> Dict:
        """Cria dados para dashboard web"""
        return {
            "system_status": {
                "display_active": self.display_active,
                "recording_active": self.recording_active,
                "current_time": datetime.now().isoformat()
            },
            "statistics": self.display_stats,
            "performance": performance_monitor.get_stats(),
            "recent_alerts": self.get_recent_alerts(1),  # Última hora
            "anomaly_summary": self._generate_anomaly_summary(),
            "visualization_settings": {
                "show_info_panel": self.visualization.show_info_panel,
                "show_flow_overlay": self.visualization.show_flow_overlay,
                "show_anomaly_markers": self.visualization.show_anomaly_markers,
                "show_statistics": self.visualization.show_statistics
            }
        }
    
    def set_visualization_settings(self, settings: Dict):
        """Atualiza configurações de visualização"""
        if "show_info_panel" in settings:
            self.visualization.show_info_panel = settings["show_info_panel"]
        
        if "show_flow_overlay" in settings:
            self.visualization.show_flow_overlay = settings["show_flow_overlay"]
        
        if "show_anomaly_markers" in settings:
            self.visualization.show_anomaly_markers = settings["show_anomaly_markers"]
        
        if "show_statistics" in settings:
            self.visualization.show_statistics = settings["show_statistics"]
        
        logger.info("Configurações de visualização atualizadas")
    
    def get_display_stats(self) -> Dict:
        """Retorna estatísticas do display"""
        stats = self.display_stats.copy()
        stats.update({
            "display_active": self.display_active,
            "recording_active": self.recording_active,
            "queue_size": self.frame_queue.qsize(),
            "recording_path": self.recording_path
        })
        return stats
    
    def reset_stats(self):
        """Reset das estatísticas"""
        self.display_stats = {
            "frames_displayed": 0,
            "frames_saved": 0,
            "alerts_sent": 0
        }
        logger.info("Estatísticas do OutputManager resetadas")
    
    def cleanup(self):
        """Limpeza completa do manager"""
        logger.info("Iniciando limpeza do OutputManager")
        
        # Parar display
        self.stop_display()
        
        # Parar gravação
        self.stop_recording()
        
        # Limpar callbacks
        self.key_callbacks.clear()
        self.mouse_callbacks.clear()
        
        # Limpar queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("OutputManager limpo")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

# Classe auxiliar para integração com interface web
class WebIntegration:
    """
    Integração com interface web Flask
    Fornece endpoints e dados para dashboard
    """
    
    def __init__(self, output_manager: OutputManager):
        self.output_manager = output_manager
        self.websocket_clients = []
        
    def get_dashboard_data(self) -> Dict:
        """Dados para dashboard web"""
        return self.output_manager.create_dashboard_data()
    
    def get_live_frame(self) -> Optional[bytes]:
        """Obtém frame atual codificado para web"""
        if self.output_manager.current_frame is not None:
            # Codificar frame como JPEG
            _, buffer = cv2.imencode('.jpg', self.output_manager.current_frame)
            return buffer.tobytes()
        return None
    
    def send_websocket_update(self, data: Dict):
        """Envia atualização via websocket"""
        # Implementação seria feita na aplicação web
        pass
    
    def register_websocket_client(self, client_id: str):
        """Registra cliente websocket"""
        if client_id not in self.websocket_clients:
            self.websocket_clients.append(client_id)
    
    def unregister_websocket_client(self, client_id: str):
        """Remove cliente websocket"""
        if client_id in self.websocket_clients:
            self.websocket_clients.remove(client_id)