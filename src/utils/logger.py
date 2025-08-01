"""
Sistema de logging avançado para detecção de anomalias
Com diferentes níveis e formatação específica para o projeto
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
import json

class ColoredFormatter(logging.Formatter):
    """Formatter com cores para diferentes níveis de log"""
    
    # Códigos de cores ANSI
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Adicionar cor baseada no nível
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class AnomalyLogger:
    """Sistema de logging especializado para detecção de anomalias"""
    
    def __init__(self, name: str = "AnomalyDetection", log_file: str = "data/logs/anomaly_detection.log"):
        self.name = name
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        
        # Evitar duplicação de handlers
        if not self.logger.handlers:
            self._setup_logger()
        
        # Arquivo para logs de anomalias específicas
        self.anomaly_log_file = "data/logs/anomalies.json"
        self.anomaly_logs = []
    
    def _setup_logger(self):
        """Configura o logger com handlers de console e arquivo"""
        self.logger.setLevel(logging.DEBUG)
        
        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Formatter para logs detalhados
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Formatter simples para console
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Handler para arquivo
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Adicionar handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, **kwargs):
        """Log de debug com informações extras"""
        extra_info = f" | {kwargs}" if kwargs else ""
        self.logger.debug(f"{message}{extra_info}")
    
    def info(self, message: str, **kwargs):
        """Log de informação"""
        extra_info = f" | {kwargs}" if kwargs else ""
        self.logger.info(f"{message}{extra_info}")
    
    def warning(self, message: str, **kwargs):
        """Log de aviso"""
        extra_info = f" | {kwargs}" if kwargs else ""
        self.logger.warning(f"{message}{extra_info}")
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log de erro com informações da exceção"""
        extra_info = f" | {kwargs}" if kwargs else ""
        error_info = f" | Exception: {str(error)}" if error else ""
        self.logger.error(f"{message}{extra_info}{error_info}")
    
    def critical(self, message: str, **kwargs):
        """Log crítico"""
        extra_info = f" | {kwargs}" if kwargs else ""
        self.logger.critical(f"{message}{extra_info}")
    
    def log_anomaly(self, anomaly_type: str, subtype: str, confidence: float, 
                   frame_info: dict, additional_data: dict = None):
        """
        Log específico para anomalias detectadas
        
        Args:
            anomaly_type: Tipo principal (security/health)
            subtype: Subtipo específico (fall, intrusion, etc.)
            confidence: Nível de confiança da detecção
            frame_info: Informações do frame (timestamp, frame_id, etc.)
            additional_data: Dados adicionais da detecção
        """
        anomaly_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": anomaly_type,
            "subtype": subtype,
            "confidence": confidence,
            "frame_info": frame_info,
            "additional_data": additional_data or {}
        }
        
        # Adicionar à lista de anomalias
        self.anomaly_logs.append(anomaly_entry)
        
        # Log normal
        self.warning(
            f"ANOMALIA DETECTADA: {anomaly_type.upper()} - {subtype}",
            confidence=f"{confidence:.2f}",
            frame=frame_info.get('frame_id', 'N/A')
        )
        
        # Salvar anomalias em arquivo JSON (últimas 1000)
        self._save_anomaly_logs()
    
    def _save_anomaly_logs(self):
        """Salva logs de anomalias em arquivo JSON"""
        try:
            # Manter apenas as últimas 1000 anomalias
            if len(self.anomaly_logs) > 1000:
                self.anomaly_logs = self.anomaly_logs[-1000:]
            
            # Garantir que o diretório existe
            os.makedirs(os.path.dirname(self.anomaly_log_file), exist_ok=True)
            
            with open(self.anomaly_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.anomaly_logs, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.error("Erro ao salvar logs de anomalias", error=e)
    
    def log_performance(self, operation: str, duration: float, **metrics):
        """Log de performance do sistema"""
        self.info(
            f"PERFORMANCE: {operation}",
            duration=f"{duration:.3f}s",
            **metrics
        )
    
    def log_training(self, epoch: int, loss: float, accuracy: float = None, **metrics):
        """Log específico para treinamento de modelos"""
        log_data = {
            "epoch": epoch,
            "loss": f"{loss:.4f}"
        }
        
        if accuracy is not None:
            log_data["accuracy"] = f"{accuracy:.4f}"
        
        log_data.update({k: f"{v:.4f}" if isinstance(v, float) else v 
                        for k, v in metrics.items()})
        
        self.info(f"TRAINING: Epoch {epoch}", **log_data)
    
    def log_system_info(self):
        """Log informações do sistema no início"""
        import psutil
        import platform
        
        system_info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "memory_gb": f"{psutil.virtual_memory().total / (1024**3):.1f}",
            "timestamp": datetime.now().isoformat()
        }
        
        self.info("SISTEMA INICIADO", **system_info)
    
    def get_anomaly_stats(self, hours: int = 24) -> dict:
        """Retorna estatísticas das anomalias das últimas N horas"""
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_anomalies = [
            a for a in self.anomaly_logs 
            if datetime.fromisoformat(a['timestamp']) > cutoff_time
        ]
        
        stats = {
            "total_anomalies": len(recent_anomalies),
            "by_type": {},
            "by_subtype": {},
            "average_confidence": 0.0
        }
        
        if recent_anomalies:
            # Estatísticas por tipo
            for anomaly in recent_anomalies:
                anomaly_type = anomaly['type']
                subtype = anomaly['subtype']
                
                stats["by_type"][anomaly_type] = stats["by_type"].get(anomaly_type, 0) + 1
                stats["by_subtype"][subtype] = stats["by_subtype"].get(subtype, 0) + 1
            
            # Confiança média
            total_confidence = sum(a['confidence'] for a in recent_anomalies)
            stats["average_confidence"] = total_confidence / len(recent_anomalies)
        
        return stats

# Instância global do logger
logger = AnomalyLogger()

class Logger:
    """Classe wrapper para compatibilidade"""
    def __init__(self):
        self.logger = logger
    
    def __getattr__(self, name):
        return getattr(self.logger, name)