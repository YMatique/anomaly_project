"""
Gerenciador de entrada de vídeo otimizado
Suporta webcam, arquivos de vídeo e streaming
Otimizado para i5 11Gen com threading eficiente
"""

import cv2
import numpy as np
import threading
import queue
import time
import os
from typing import Dict, Optional, Tuple, Union, Callable
from datetime import datetime

from ..utils.helpers import time_function, performance_monitor, VideoProcessor, FileManager
from ..utils.logger import logger

class VideoSource:
    """Classe base para fontes de vídeo"""
    
    def __init__(self, source_id: str):
        self.source_id = source_id
        self.is_active = False
        self.frame_count = 0
        self.start_time = None
        
    def initialize(self) -> bool:
        """Inicializa a fonte de vídeo"""
        raise NotImplementedError
        
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """Obtém próximo frame"""
        raise NotImplementedError
        
    def release(self):
        """Libera recursos"""
        raise NotImplementedError
        
    def get_info(self) -> Dict:
        """Retorna informações da fonte"""
        raise NotImplementedError

class WebcamSource(VideoSource):
    """Fonte de vídeo para webcam"""
    
    def __init__(self, camera_index: int = 0, resolution: Tuple[int, int] = (640, 480)):
        super().__init__(f"webcam_{camera_index}")
        self.camera_index = camera_index
        self.resolution = resolution
        self.cap = None
        
    def initialize(self) -> bool:
        """Inicializa captura da webcam"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                logger.error(f"Não foi possível abrir webcam {self.camera_index}")
                return False
            
            # Configurar resolução
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Configurar buffer para reduzir latência
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Testar captura de frame
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Não foi possível capturar frame da webcam")
                return False
            
            self.is_active = True
            self.start_time = time.time()
            logger.info(f"Webcam {self.camera_index} inicializada - {self.resolution}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar webcam: {e}")
            return False
    
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """Captura frame da webcam"""
        if not self.is_active or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            return True, frame
        
        return False, None
    
    def release(self):
        """Libera recursos da webcam"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.is_active = False
        logger.info(f"Webcam {self.camera_index} liberada")
    
    def get_info(self) -> Dict:
        """Informações da webcam"""
        info = {
            "source_type": "webcam",
            "camera_index": self.camera_index,
            "resolution": self.resolution,
            "is_active": self.is_active,
            "frame_count": self.frame_count
        }
        
        if self.cap is not None and self.is_active:
            info.update({
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "actual_width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "actual_height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            })
        
        return info

class VideoFileSource(VideoSource):
    """Fonte de vídeo para arquivos"""
    
    def __init__(self, file_path: str, loop: bool = False):
        super().__init__(f"file_{os.path.basename(file_path)}")
        self.file_path = file_path
        self.loop = loop
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame = 0
        
    def initialize(self) -> bool:
        """Inicializa captura do arquivo"""
        try:
            if not os.path.exists(self.file_path):
                logger.error(f"Arquivo não encontrado: {self.file_path}")
                return False
            
            self.cap = cv2.VideoCapture(self.file_path)
            
            if not self.cap.isOpened():
                logger.error(f"Não foi possível abrir arquivo: {self.file_path}")
                return False
            
            # Obter informações do vídeo
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            if self.fps <= 0:
                self.fps = 30.0  # FPS padrão se não conseguir detectar
            
            # Testar captura
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Não foi possível ler frame do arquivo")
                return False
            
            # Voltar ao início
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self.is_active = True
            self.start_time = time.time()
            logger.info(f"Arquivo de vídeo carregado: {self.file_path} ({self.total_frames} frames, {self.fps} fps)")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar arquivo: {e}")
            return False
    
    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """Lê próximo frame do arquivo"""
        if not self.is_active or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.frame_count += 1
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            return True, frame
        else:
            # Fim do arquivo
            if self.loop:
                # Reiniciar vídeo
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                ret, frame = self.cap.read()
                if ret:
                    self.frame_count += 1
                    return True, frame
            
            # Arquivo terminou e não está em loop
            self.is_active = False
            return False, None
    
    def seek_frame(self, frame_number: int) -> bool:
        """Vai para frame específico"""
        if self.cap is not None and 0 <= frame_number < self.total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            return True
        return False
    
    def get_progress(self) -> float:
        """Retorna progresso de reprodução (0-1)"""
        if self.total_frames > 0:
            return self.current_frame / self.total_frames
        return 0.0
    
    def release(self):
        """Libera recursos do arquivo"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.is_active = False
        logger.info(f"Arquivo de vídeo liberado: {self.file_path}")
    
    def get_info(self) -> Dict:
        """Informações do arquivo de vídeo"""
        info = {
            "source_type": "file",
            "file_path": self.file_path,
            "is_active": self.is_active,
            "frame_count": self.frame_count,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame,
            "fps": self.fps,
            "progress": self.get_progress(),
            "loop": self.loop
        }
        
        if self.cap is not None and self.is_active:
            info.update({
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": self.total_frames / self.fps if self.fps > 0 else 0
            })
        
        return info

class InputManager:
    """
    Gerenciador principal de entrada de vídeo
    Suporta múltiplas fontes e processamento em threading
    """
    
    def __init__(self, config):
        self.config = config
        self.current_source = None
        self.is_running = False
        
        # Threading para captura assíncrona
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=config.system.queue_size)
        self.frame_callbacks = []
        
        # Controle de FPS
        self.target_fps = config.video.fps
        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = 0
        
        # Estatísticas
        self.stats = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "avg_fps": 0.0,
            "capture_errors": 0
        }
        
        # Frame skip para otimização
        self.frame_skip = config.video.frame_skip
        self.skip_counter = 0
        
        logger.info("InputManager inicializado")
    
    def initialize_webcam(self, camera_index: int = 0, 
                         resolution: Optional[Tuple[int, int]] = None) -> bool:
        """
        Inicializa captura da webcam
        
        Args:
            camera_index: Índice da câmera
            resolution: Resolução desejada (usa configuração se None)
        """
        if resolution is None:
            resolution = self.config.video.realtime_resolution
        
        logger.info(f"Inicializando webcam {camera_index} - resolução: {resolution}")
        
        # Liberar fonte atual se existir
        self._release_current_source()
        
        # Criar nova fonte
        webcam_source = WebcamSource(camera_index, resolution)
        
        if webcam_source.initialize():
            self.current_source = webcam_source
            logger.info("Webcam inicializada com sucesso")
            return True
        else:
            logger.error("Falha ao inicializar webcam")
            return False
    
    def initialize_video_file(self, file_path: str, loop: bool = False) -> bool:
        """
        Inicializa captura de arquivo de vídeo
        
        Args:
            file_path: Caminho para o arquivo
            loop: Se deve repetir o vídeo
        """
        logger.info(f"Inicializando arquivo: {file_path}")
        
        # Liberar fonte atual
        self._release_current_source()
        
        # Criar nova fonte
        file_source = VideoFileSource(file_path, loop)
        
        if file_source.initialize():
            self.current_source = file_source
            logger.info("Arquivo de vídeo inicializado com sucesso")
            return True
        else:
            logger.error("Falha ao inicializar arquivo de vídeo")
            return False
    
    def start_capture(self) -> bool:
        """Inicia captura de vídeo em thread separada"""
        if self.current_source is None:
            logger.error("Nenhuma fonte de vídeo inicializada")
            return False
        
        if self.is_running:
            logger.warning("Captura já está em execução")
            return True
        
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        logger.info("Captura de vídeo iniciada")
        return True
    
    def stop_capture(self):
        """Para captura de vídeo"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
        
        # Limpar queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Captura de vídeo parada")
    
    def _capture_loop(self):
        """Loop principal de captura (executado em thread separada)"""
        logger.info("Thread de captura iniciada")
        
        while self.is_running and self.current_source is not None:
            try:
                current_time = time.time()
                
                # Controle de FPS
                time_since_last = current_time - self.last_frame_time
                if time_since_last < self.frame_interval:
                    sleep_time = self.frame_interval - time_since_last
                    time.sleep(sleep_time)
                    continue
                
                # Capturar frame
                ret, frame = self.current_source.get_frame()
                
                if not ret:
                    if isinstance(self.current_source, VideoFileSource):
                        # Arquivo terminou
                        logger.info("Fim do arquivo de vídeo")
                        break
                    else:
                        # Erro de captura
                        self.stats["capture_errors"] += 1
                        logger.warning("Erro na captura de frame")
                        time.sleep(0.1)  # Pequena pausa antes de tentar novamente
                        continue
                
                # Frame skip para otimização
                self.skip_counter += 1
                if self.skip_counter < self.frame_skip:
                    continue
                self.skip_counter = 0
                
                # Atualizar estatísticas
                self.stats["frames_captured"] += 1
                self.last_frame_time = current_time
                
                # Atualizar FPS
                performance_monitor.update_fps()
                
                # Adicionar frame à queue (não bloqueante)
                frame_data = {
                    "frame": frame,
                    "timestamp": current_time,
                    "frame_id": self.stats["frames_captured"],
                    "source_info": self.current_source.get_info()
                }
                
                try:
                    self.frame_queue.put_nowait(frame_data)
                except queue.Full:
                    # Queue cheia - descartar frame mais antigo
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                        self.stats["frames_dropped"] += 1
                    except queue.Empty:
                        pass
                
                # Chamar callbacks
                self._call_frame_callbacks(frame_data)
                
            except Exception as e:
                logger.error(f"Erro no loop de captura: {e}")
                self.stats["capture_errors"] += 1
                time.sleep(0.1)
        
        logger.info("Thread de captura finalizada")
    
    def get_frame(self, timeout: float = 1.0) -> Optional[Dict]:
        """
        Obtém próximo frame da queue
        
        Args:
            timeout: Timeout em segundos
            
        Returns:
            Dict com dados do frame ou None se timeout
        """
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_frame_nowait(self) -> Optional[Dict]:
        """Obtém frame sem esperar (não bloqueante)"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def add_frame_callback(self, callback: Callable[[Dict], None]):
        """
        Adiciona callback para ser chamado quando novo frame é capturado
        
        Args:
            callback: Função que recebe dict com dados do frame
        """
        self.frame_callbacks.append(callback)
        logger.info(f"Callback adicionado - total: {len(self.frame_callbacks)}")
    
    def remove_frame_callback(self, callback: Callable[[Dict], None]):
        """Remove callback"""
        if callback in self.frame_callbacks:
            self.frame_callbacks.remove(callback)
            logger.info(f"Callback removido - total: {len(self.frame_callbacks)}")
    
    def _call_frame_callbacks(self, frame_data: Dict):
        """Chama todos os callbacks registrados"""
        for callback in self.frame_callbacks:
            try:
                callback(frame_data)
            except Exception as e:
                logger.error(f"Erro em callback: {e}")
    
    def _release_current_source(self):
        """Libera fonte atual se existir"""
        if self.current_source is not None:
            self.current_source.release()
            self.current_source = None
    
    def get_source_info(self) -> Optional[Dict]:
        """Retorna informações da fonte atual"""
        if self.current_source is not None:
            return self.current_source.get_info()
        return None
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas de captura"""
        stats = self.stats.copy()
        
        # Adicionar estatísticas de performance
        perf_stats = performance_monitor.get_stats()
        stats.update({
            "current_fps": perf_stats["fps"]["current"],
            "avg_fps": perf_stats["fps"]["average"],
            "queue_size": self.frame_queue.qsize(),
            "is_running": self.is_running
        })
        
        return stats
    
    def save_current_frame(self, directory: str = "data/captures/") -> Optional[str]:
        """Salva frame atual em arquivo"""
        frame_data = self.get_frame_nowait()
        if frame_data is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"capture_{timestamp}.jpg"
            return FileManager.save_frame(frame_data["frame"], filename, directory)
        return None
    
    def set_fps(self, fps: float):
        """Ajusta FPS de captura"""
        self.target_fps = max(1.0, min(fps, 60.0))  # Limitar entre 1-60 FPS
        self.frame_interval = 1.0 / self.target_fps
        logger.info(f"FPS ajustado para: {self.target_fps}")
    
    def set_frame_skip(self, skip: int):
        """Ajusta frame skip para otimização"""
        self.frame_skip = max(1, skip)
        logger.info(f"Frame skip ajustado para: {self.frame_skip}")
    
    def reset_stats(self):
        """Reseta estatísticas"""
        self.stats = {
            "frames_captured": 0,
            "frames_dropped": 0,
            "avg_fps": 0.0,
            "capture_errors": 0
        }
        logger.info("Estatísticas resetadas")
    
    def cleanup(self):
        """Limpeza completa do manager"""
        logger.info("Iniciando limpeza do InputManager")
        
        # Parar captura
        self.stop_capture()
        
        # Liberar fonte
        self._release_current_source()
        
        # Limpar callbacks
        self.frame_callbacks.clear()
        
        # Limpar queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logger.info("InputManager limpo")
    
    def __del__(self):
        """Destructor - garante limpeza"""
        self.cleanup()