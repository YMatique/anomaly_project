#!/usr/bin/env python3
"""
Sistema de Detecção de Anomalias em Tempo Real
Usando Optical Flow, CAE (Convolutional Autoencoder) e ConvLSTM

Versão Completa com Análise Quantitativa
"""

import os
import sys
import time
import argparse
import signal
import threading
from typing import Optional, Dict, Union, List
import cv2
import json
from datetime import datetime

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.input_manager import InputManager
from src.core.processing_engine import ProcessingEngine
from src.core.output_manager import OutputManager
from src.utils.config import Config
from src.utils.logger import logger
from src.utils.helpers import create_directories, check_system_requirements

class AnomalyDetectionSystem:
    """
    Sistema principal de detecção de anomalias
    Integra todos os componentes: entrada, processamento e saída
    """
    
    def __init__(self, config_file: str = "config.json"):
        """Inicializa o sistema"""
        
        print("🚀 Inicializando Sistema de Detecção de Anomalias...")
        
        # Verificar requisitos do sistema
        check_system_requirements()
        
        # Criar diretórios necessários
        create_directories()
        
        # Configuração
        self.config = Config(config_file)
        self.config.optimize_for_hardware()
        
        # Componentes principais
        self.input_manager = None
        self.processing_engine = None
        self.output_manager = None
        
        # Estado do sistema
        self.running = False
        self.paused = False
        self.start_time = None
        
        # Métricas da sessão
        self.session_metrics = {
            'frames_processed': 0,
            'anomalies_detected': 0,
            'start_time': None,
            'performance_data': []
        }
        
        # Configurar manipuladores de sinal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.log_system_info()
        logger.info("Sistema de Detecção de Anomalias inicializado")
        print("✅ Sistema inicializado com sucesso!")
    
    def _signal_handler(self, signum, frame):
        """Manipulador de sinais para parada graceful"""
        print(f"\n⚠️  Sinal {signum} recebido. Parando sistema...")
        self.stop()
        print("👋 Sistema finalizado graciosamente")
        sys.exit(0)
    
    def initialize_components(self):
        """Inicializa os componentes do sistema"""
        try:
            print("🔧 Inicializando componentes...")
            
            # Input Manager
            if not self.input_manager:
                self.input_manager = InputManager(self.config)
                print("  ✅ Input Manager inicializado")
            
            # Processing Engine
            if not self.processing_engine:
                self.processing_engine = ProcessingEngine(self.config)
                self._setup_callbacks()
                print("  ✅ Processing Engine inicializado")
            
            # Output Manager
            if not self.output_manager:
                self.output_manager = OutputManager(self.config)
                print("  ✅ Output Manager inicializado")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar componentes: {e}")
            print(f"❌ Erro na inicialização: {e}")
            return False
    
    def _setup_callbacks(self):
        """Configura callbacks entre componentes"""
        
        # Input Manager -> Processing Engine
        def frame_callback(frame_data):
            if not self.paused and self.running:
                success = self.processing_engine.add_frame_to_queue(frame_data)
                if success:
                    self.session_metrics['frames_processed'] += 1
        
        # Processing Engine -> Output Manager
        def result_callback(result, frame_data):
            if self.output_manager:
                self.output_manager.handle_result(result, frame_data)
            
            # Coletar métricas para análise
            self._collect_performance_metrics(result)
        
        def anomaly_callback(result, frame_data):
            if self.output_manager:
                self.output_manager.handle_anomaly(result, frame_data)
            
            self.session_metrics['anomalies_detected'] += 1
            logger.warning(f"Anomalia detectada: {result.anomaly_type} (score: {result.final_anomaly_score:.3f})")
        
        # Registrar callbacks
        if self.input_manager:
            self.input_manager.add_frame_callback(frame_callback)
        
        if self.processing_engine:
            self.processing_engine.add_result_callback(result_callback)
            self.processing_engine.add_anomaly_callback(anomaly_callback)
    
    def _collect_performance_metrics(self, result):
        """Coleta métricas de performance para análise posterior"""
        try:
            # Obter métricas do sistema
            system_metrics = self.processing_engine.get_metrics() if self.processing_engine else {}
            
            # Criar registro de performance
            performance_record = {
                'timestamp': result.timestamp.timestamp(),
                'frame_id': result.frame_id,
                'fps': system_metrics.get('fps', {}).get('current', 0),
                'processing_time': result.processing_time,
                'anomaly_score': result.final_anomaly_score,
                'optical_flow_score': result.optical_flow_score,
                'cae_score': result.cae_score,
                'convlstm_score': result.convlstm_score,
                'cpu_usage': system_metrics.get('system', {}).get('cpu_usage', 0),
                'memory_usage': system_metrics.get('system', {}).get('memory_usage', 0),
                'is_anomaly': result.is_anomaly,
                'anomaly_type': result.anomaly_type
            }
            
            # Adicionar às métricas da sessão
            self.session_metrics['performance_data'].append(performance_record)
            
            # Manter apenas os últimos 10000 registros para evitar uso excessivo de memória
            if len(self.session_metrics['performance_data']) > 10000:
                self.session_metrics['performance_data'] = self.session_metrics['performance_data'][-5000:]
                
        except Exception as e:
            logger.error(f"Erro ao coletar métricas: {e}")
    
    def start(self, mode: str = "webcam", **kwargs):
        """Inicia o sistema de detecção"""
        try:
            if self.running:
                print("⚠️  Sistema já está em execução")
                return False
            
            print(f"🎬 Iniciando sistema em modo: {mode}")
            
            # Inicializar componentes
            if not self.initialize_components():
                return False
            
            # Configurar fonte de entrada baseada no modo
            if mode == "webcam":
                camera_id = kwargs.get('camera_id', 0)
                success = self.input_manager.start_webcam(camera_id)
                if not success:
                    print(f"❌ Falha ao conectar com câmera {camera_id}")
                    return False
                print(f"📹 Câmera {camera_id} conectada")
                
            elif mode == "video":
                video_path = kwargs.get('video_path')
                if not video_path or not os.path.exists(video_path):
                    print(f"❌ Arquivo de vídeo não encontrado: {video_path}")
                    return False
                success = self.input_manager.start_video_file(video_path)
                if not success:
                    print(f"❌ Falha ao abrir vídeo: {video_path}")
                    return False
                print(f"🎥 Vídeo carregado: {video_path}")
                
            elif mode == "train":
                # Modo de treinamento
                duration = kwargs.get('duration_minutes', 10)
                print(f"🎓 Modo treinamento ativado por {duration} minutos")
                camera_id = kwargs.get('camera_id', 0)
                success = self.input_manager.start_webcam(camera_id)
                if not success:
                    print(f"❌ Falha ao conectar com câmera para treinamento")
                    return False
                
                # Configurar modo de treinamento
                self.config.set('training.mode', True)
                self.config.set('training.duration_minutes', duration)
                
            else:
                print(f"❌ Modo desconhecido: {mode}")
                return False
            
            # Iniciar componentes
            self.processing_engine.start()
            
            # Configurar pipeline
            self._setup_callbacks()
            
            # Marcar como iniciado
            self.running = True
            self.paused = False
            self.start_time = datetime.now()
            self.session_metrics['start_time'] = self.start_time
            
            logger.info(f"Sistema iniciado em modo {mode}")
            print("✅ Sistema iniciado com sucesso!")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao iniciar sistema: {e}")
            print(f"❌ Erro ao iniciar: {e}")
            return False
    
    def pause(self):
        """Pausa o sistema"""
        if not self.running:
            print("⚠️  Sistema não está em execução")
            return
        
        self.paused = True
        if self.processing_engine:
            self.processing_engine.pause()
        
        print("⏸️  Sistema pausado")
        logger.info("Sistema pausado")
    
    def resume(self):
        """Resume o sistema"""
        if not self.running:
            print("⚠️  Sistema não está em execução")
            return
        
        self.paused = False
        if self.processing_engine:
            self.processing_engine.resume()
        
        print("▶️  Sistema resumido")
        logger.info("Sistema resumido")
    
    def stop(self):
        """Para o sistema"""
        try:
            if not self.running:
                return
            
            print("🛑 Parando sistema...")
            
            # Parar componentes
            if self.input_manager:
                self.input_manager.stop()
                print("  ✅ Input Manager parado")
            
            if self.processing_engine:
                self.processing_engine.stop()
                print("  ✅ Processing Engine parado")
            
            if self.output_manager:
                self.output_manager.stop()
                print("  ✅ Output Manager parado")
            
            # Marcar como parado
            self.running = False
            self.paused = False
            
            # Salvar métricas da sessão
            self._save_session_metrics()
            
            logger.info("Sistema parado")
            print("✅ Sistema parado com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao parar sistema: {e}")
            print(f"❌ Erro ao parar: {e}")
    
    def _save_session_metrics(self):
        """Salva métricas da sessão atual"""
        try:
            if not self.session_metrics['performance_data']:
                return
            
            # Criar diretório de métricas
            metrics_dir = os.path.join('data', 'metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            
            # Preparar dados para salvamento  
            session_data = {
                'session_info': {
                    'start_time': self.session_metrics['start_time'].isoformat() if self.session_metrics['start_time'] else None,
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': (datetime.now() - self.session_metrics['start_time']).total_seconds() if self.session_metrics['start_time'] else 0,
                    'frames_processed': self.session_metrics['frames_processed'],
                    'anomalies_detected': self.session_metrics['anomalies_detected']
                },
                'detailed_data': {
                    'timestamps': [record['timestamp'] for record in self.session_metrics['performance_data']],
                    'fps': [record['fps'] for record in self.session_metrics['performance_data']],
                    'processing_times': [record['processing_time'] for record in self.session_metrics['performance_data']],
                    'anomaly_scores': [record['anomaly_score'] for record in self.session_metrics['performance_data']],
                    'memory_usage': [record['memory_usage'] for record in self.session_metrics['performance_data']],
                    'cpu_usage': [record['cpu_usage'] for record in self.session_metrics['performance_data']]
                },
                'summary': self.processing_engine.get_metrics() if self.processing_engine else {}
            }
            
            # Salvar arquivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_metrics_{timestamp}.json"
            filepath = os.path.join(metrics_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"📊 Métricas salvas em: {filepath}")
            logger.info(f"Métricas da sessão salvas: {filepath}")
            
            return filepath
            
        except Exception as e:
            logger.error(f"Erro ao salvar métricas: {e}")
            print(f"❌ Erro ao salvar métricas: {e}")
            return None
    
    def run_interactive(self):
        """Executa o sistema em modo interativo"""
        print("\n🎮 Modo Interativo Ativado")
        print("Controles disponíveis:")
        print("  [ESPAÇO] - Pausar/Resumir")
        print("  [S] - Mostrar estatísticas")
        print("  [R] - Exportar relatório")
        print("  [Q/ESC] - Sair")
        print("  [H] - Mostrar esta ajuda")
        
        try:
            while self.running:
                # Verificar input do usuário (não bloqueante)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q ou ESC
                    break
                elif key == ord(' '):  # ESPAÇO
                    if self.paused:
                        self.resume()
                    else:
                        self.pause()
                elif key == ord('s'):  # S - Estatísticas
                    self._show_statistics()
                elif key == ord('r'):  # R - Relatório
                    self._export_report()
                elif key == ord('h'):  # H - Ajuda
                    self._show_help()
                
                # Mostrar informações na tela se houver frame atual
                if self.output_manager and hasattr(self.output_manager, 'current_frame'):
                    frame = self.output_manager.current_frame
                    if frame is not None:
                        # Adicionar overlay com informações
                        self._add_info_overlay(frame)
                        cv2.imshow('Sistema de Detecção de Anomalias', frame)
                
                time.sleep(0.01)  # Pequena pausa para evitar uso excessivo de CPU
                
        except KeyboardInterrupt:
            pass
        finally:
            cv2.destroyAllWindows()
    
    def _add_info_overlay(self, frame):
        """Adiciona overlay com informações do sistema"""
        try:
            # Obter métricas atuais
            metrics = self.processing_engine.get_metrics() if self.processing_engine else {}
            
            # Preparar textos
            fps = metrics.get('fps', {}).get('current', 0)
            frames_processed = self.session_metrics['frames_processed']
            anomalies = self.session_metrics['anomalies_detected']
            uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            # Status
            status = "PAUSADO" if self.paused else "ATIVO"
            status_color = (0, 255, 255) if self.paused else (0, 255, 0)
            
            # Adicionar textos ao frame
            y_offset = 30
            cv2.putText(frame, f"Status: {status}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            y_offset += 30
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Frames: {frames_processed}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Anomalias: {anomalies}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            y_offset += 25
            cv2.putText(frame, f"Tempo: {uptime/60:.1f}min", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        except Exception as e:
            logger.error(f"Erro ao adicionar overlay: {e}")
    
    def _show_statistics(self):
        """Mostra estatísticas detalhadas no console"""
        print("\n📊 ESTATÍSTICAS DO SISTEMA")
        print("=" * 50)
        
        # Métricas básicas
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        print(f"Tempo Ativo: {uptime/60:.1f} minutos")
        print(f"Frames Processados: {self.session_metrics['frames_processed']:,}")
        print(f"Anomalias Detectadas: {self.session_metrics['anomalies_detected']:,}")
        
        if self.session_metrics['frames_processed'] > 0:
            detection_rate = (self.session_metrics['anomalies_detected'] / self.session_metrics['frames_processed']) * 100
            print(f"Taxa de Detecção: {detection_rate:.2f}%")
        
        # Métricas do processing engine
        if self.processing_engine:
            metrics = self.processing_engine.get_metrics()
            fps_data = metrics.get('fps', {})
            processing_data = metrics.get('processing_time', {})
            system_data = metrics.get('system', {})
            
            print(f"\nPerformance:")
            print(f"  FPS Atual: {fps_data.get('current', 0):.1f}")
            print(f"  FPS Médio: {fps_data.get('average', 0):.1f}")
            print(f"  Tempo Proc. Atual: {processing_data.get('current', 0)*1000:.1f}ms")
            print(f"  CPU: {system_data.get('cpu_usage', 0):.1f}%")
            print(f"  RAM: {system_data.get('memory_usage', 0):.1f}%")
        
        print("=" * 50)
    
    def _show_help(self):
        """Mostra ajuda dos controles"""
        print("\n🎮 CONTROLES DISPONÍVEIS")
        print("=" * 30)
        print("ESPAÇO - Pausar/Resumir")
        print("S - Mostrar estatísticas")
        print("R - Exportar relatório")
        print("Q/ESC - Sair do sistema")
        print("H - Mostrar esta ajuda")
        print("=" * 30)
    
    def _export_report(self):
        """Exporta relatório de análise"""
        try:
            print("\n📋 Exportando relatório...")
            
            # Salvar métricas primeiro
            metrics_file = self._save_session_metrics()
            if not metrics_file:
                print("❌ Erro ao salvar métricas")
                return
            
            # Gerar análise quantitativa
            try:
                from src.utils.analytics import analyze_system_performance
                
                results = analyze_system_performance(metrics_file)
                
                if 'error' in results:
                    print(f"❌ Erro na análise: {results['error']}")
                    return
                
                print("✅ Relatório exportado com sucesso!")
                print(f"📁 Arquivos salvos em: data/analytics/")
                
                # Listar arquivos gerados
                for key, filepath in results.items():
                    if isinstance(filepath, str) and os.path.exists(filepath):
                        print(f"  📄 {key}: {os.path.basename(filepath)}")
                    elif isinstance(filepath, list):
                        print(f"  📄 {key}: {len(filepath)} arquivos")
            
            except ImportError:
                print("⚠️  Módulo de análise não disponível. Apenas métricas salvas.")
            except Exception as e:
                print(f"❌ Erro na análise: {e}")
                
        except Exception as e:
            logger.error(f"Erro ao exportar relatório: {e}")
            print(f"❌ Erro ao exportar relatório: {e}")
    
    def get_status(self) -> Dict:
        """Retorna status atual do sistema"""
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        status = {
            'running': self.running,
            'paused': self.paused,
            'uptime_seconds': uptime,
            'frames_processed': self.session_metrics['frames_processed'],
            'anomalies_detected': self.session_metrics['anomalies_detected'],
            'start_time': self.start_time.isoformat() if self.start_time else None
        }
        
        # Adicionar métricas do processing engine se disponível
        if self.processing_engine:
            status['performance_metrics'] = self.processing_engine.get_metrics()
        
        return status

def create_default_config():
    """Cria arquivo de configuração padrão"""
    config_path = "config.json"
    
    if os.path.exists(config_path):
        print(f"⚠️  Arquivo de configuração já existe: {config_path}")
        return
    
    default_config = {
        "system": {
            "log_level": "INFO",
            "max_workers": 2,
            "enable_gpu": False
        },
        "input": {
            "webcam_id": 0,
            "frame_width": 640,
            "frame_height": 480,
            "fps_limit": 30
        },
        "processing": {
            "queue_size": 10,
            "num_workers": 2,
            "frame_skip": 1
        },
        "thresholds": {
            "optical_flow": 0.3,
            "cae": 0.5,
            "convlstm": 0.6
        },
        "weights": {
            "optical_flow": 0.2,
            "cae": 0.4,
            "convlstm": 0.4
        },
        "alerts": {
            "cooldown_seconds": 5,
            "enable_email": False,
            "enable_webhook": False
        },
        "output": {
            "save_frames": True,
            "show_overlay": True,
            "enable_sound": False
        }
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Configuração padrão criada: {config_path}")
        
    except Exception as e:
        print(f"❌ Erro ao criar configuração: {e}")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Sistema de Detecção de Anomalias em Tempo Real",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  %(prog)s --mode webcam --camera 0          # Webcam padrão
  %(prog)s --mode video --video video.mp4    # Arquivo de vídeo
  %(prog)s --mode train --duration 15        # Treinamento por 15 min
  %(prog)s create-config                     # Criar configuração padrão
  %(prog)s demo                              # Modo demonstração
        """
    )
    
    parser.add_argument('command', nargs='?', choices=['create-config', 'demo'], 
                       help='Comando especial a executar')
    parser.add_argument('--mode', choices=['webcam', 'video', 'train'], 
                       default='webcam', help='Modo de operação')
    parser.add_argument('--camera', type=int, default=0, 
                       help='ID da câmera (padrão: 0)')
    parser.add_argument('--video', type=str, 
                       help='Caminho para arquivo de vídeo')
    parser.add_argument('--duration', type=int, default=10, 
                       help='Duração do treinamento em minutos')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Arquivo de configuração')
    parser.add_argument('--interactive', action='store_true',
                       help='Executar em modo interativo')
    parser.add_argument('--no-display', action='store_true',
                       help='Executar sem interface visual')
    
    args = parser.parse_args()
    
    # Comandos especiais
    if args.command == 'create-config':
        create_default_config()
        return 0
    
    if args.command == 'demo':
        # Modo demonstração
        print("🎭 Modo Demonstração")
        args.mode = 'webcam'
        args.camera = 0
        args.interactive = True
    
    # Verificar argumentos específicos
    if args.mode == 'video' and not args.video:
        print("❌ Modo vídeo requer --video")
        return 1
    
    if args.mode == 'video' and not os.path.exists(args.video):
        print(f"❌ Arquivo não encontrado: {args.video}")
        return 1
    
    # Criar sistema
    try:
        print("🤖 Sistema de Detecção de Anomalias v1.0")
        print("🔬 Algoritmos: Optical Flow + CAE + ConvLSTM")
        print("=" * 50)
        
        system = AnomalyDetectionSystem(args.config)
        
        # Iniciar sistema
        start_kwargs = {}
        if args.mode == 'webcam':
            start_kwargs['camera_id'] = args.camera
        elif args.mode == 'video':
            start_kwargs['video_path'] = args.video
        elif args.mode == 'train':
            start_kwargs['duration_minutes'] = args.duration
            start_kwargs['camera_id'] = args.camera
        
        success = system.start(args.mode, **start_kwargs)
        if not success:
            return 1
        
        # Modo de execução
        if args.interactive and not args.no_display:
            system.run_interactive()
        else:
            # Modo simples - aguardar interrupção
            try:
                print("🏃 Sistema em execução... (Ctrl+C para parar)")
                while system.running:
                    # Mostrar estatísticas periodicamente
                    time.sleep(10)
                    status = system.get_status()
                    print(f"📊 Frames: {status['frames_processed']:,} | "
                          f"Anomalias: {status['anomalies_detected']} | "
                          f"Uptime: {status['uptime_seconds']/60:.1f}min")
                    
            except KeyboardInterrupt:
                print("\n⚠️  Interrupção detectada...")
        
        # Parar sistema
        system.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"Erro crítico: {e}")
        print(f"💥 Erro crítico: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)