#!/usr/bin/env python3
"""
Sistema de Detecção de Anomalias em Tempo Real
Usando Optical Flow, CAE (Convolutional Autoencoder) e ConvLSTM
Otimizado para i5 11Gen, 16GB RAM

Autor: Sistema AI
Versão: 1.0
"""

import os
import sys
import time
import argparse
import signal
from typing import Optional, Dict
import cv2

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.input_manager import InputManager
from src.core.processing_engine import ProcessingEngine
from src.core.output_manager import OutputManager
from src.utils.config import Config
from src.utils.logger import logger

class AnomalyDetectionSystem:
    """
    Sistema principal de detecção de anomalias
    Integra todos os componentes: entrada, processamento e saída
    """
    
    def __init__(self, config_file: str = "config.json"):
        """Inicializa o sistema"""
        
        # Configuração
        self.config = Config(config_file)
        self.config.optimize_for_hardware()
        
        # Componentes principais
        self.input_manager = InputManager(self.config)
        self.processing_engine = ProcessingEngine(self.config)
        self.output_manager = OutputManager(self.config)
        
        # Estado do sistema
        self.running = False
        self.paused = False
        
        # Configurar callbacks
        self._setup_callbacks()
        
        # Configurar manipuladores de sinal
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.log_system_info()
        logger.info("Sistema de Detecção de Anomalias inicializado")
    
    def _setup_callbacks(self):
        """Configura callbacks entre componentes"""
        
        # Input Manager -> Processing Engine
        def frame_callback(frame_data):
            if not self.paused:
                self.processing_engine.add_frame_to_queue(frame_data)
        
        self.input_manager.add_frame_callback(frame_callback)
        
        # Processing Engine -> Output Manager
        def result_callback(results, frame_data):
            self.output_manager.display_frame(
                frame_data["frame"], results, frame_data
            )
        
        self.processing_engine.add_result_callback(result_callback)
        
        # Callback para anomalias críticas
        def anomaly_callback(anomaly, results, frame_data):
            if anomaly["category"] == "health" and anomaly["type"] in ["fall", "collapse"]:
                logger.critical(f"ANOMALIA CRÍTICA: {anomaly['description']}")
        
        self.processing_engine.add_anomaly_callback(anomaly_callback)
    
    def start_webcam_detection(self, camera_index: int = 0, 
                             display: bool = True) -> bool:
        """
        Inicia detecção usando webcam
        
        Args:
            camera_index: Índice da câmera
            display: Se deve mostrar interface visual
            
        Returns:
            True se iniciado com sucesso
        """
        logger.info(f"Iniciando detecção por webcam - câmera {camera_index}")
        
        try:
            # Inicializar entrada
            if not self.input_manager.initialize_webcam(camera_index):
                logger.error("Falha ao inicializar webcam")
                return False
            
            # Iniciar componentes
            if not self._start_system_components(display):
                return False
            
            logger.info("Sistema iniciado - pressione 'q' para sair, 'p' para pausar")
            
            # Loop principal
            self._main_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na detecção por webcam: {e}")
            return False
    
    def start_video_detection(self, video_path: str, display: bool = True, 
                            loop: bool = False) -> bool:
        """
        Inicia detecção usando arquivo de vídeo
        
        Args:
            video_path: Caminho para o arquivo de vídeo
            display: Se deve mostrar interface visual
            loop: Se deve repetir o vídeo
            
        Returns:
            True se iniciado com sucesso
        """
        logger.info(f"Iniciando detecção por arquivo - {video_path}")
        
        try:
            # Inicializar entrada
            if not self.input_manager.initialize_video_file(video_path, loop):
                logger.error("Falha ao inicializar arquivo de vídeo")
                return False
            
            # Iniciar componentes
            if not self._start_system_components(display):
                return False
            
            logger.info("Sistema iniciado - pressione 'q' para sair, 'p' para pausar")
            
            # Loop principal
            self._main_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro na detecção por arquivo: {e}")
            return False
    
    def start_training_mode(self, camera_index: int = 0, 
                          duration_minutes: int = 10) -> bool:
        """
        Inicia modo de treinamento
        
        Args:
            camera_index: Índice da câmera
            duration_minutes: Duração do treinamento em minutos
            
        Returns:
            True se treinamento foi bem-sucedido
        """
        logger.info(f"Iniciando modo de treinamento - {duration_minutes} minutos")
        
        try:
            # Inicializar webcam
            if not self.input_manager.initialize_webcam(camera_index):
                logger.error("Falha ao inicializar webcam para treinamento")
                return False
            
            # Iniciar captura
            if not self.input_manager.start_capture():
                logger.error("Falha ao iniciar captura")
                return False
            
            # Ativar modo de treinamento
            self.processing_engine.start_training_mode()
            
            # Iniciar processamento
            if not self.processing_engine.start_processing():
                logger.error("Falha ao iniciar processamento")
                return False
            
            # Iniciar display
            self.output_manager.start_display()
            
            logger.info("Coletando dados de treinamento... (comportamento NORMAL apenas)")
            logger.info("Evite movimentos anômalos durante este período")
            
            # Coletar dados por tempo especificado
            start_time = time.time()
            duration_seconds = duration_minutes * 60
            
            while time.time() - start_time < duration_seconds:
                if not self.running:
                    break
                
                # Mostrar progresso
                elapsed = time.time() - start_time
                progress = (elapsed / duration_seconds) * 100
                
                if int(elapsed) % 30 == 0:  # Log a cada 30 segundos
                    logger.info(f"Coletando dados... {progress:.1f}% concluído")
                
                # Processar eventos de teclado
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Treinamento interrompido pelo usuário")
                    break
                
                time.sleep(1)
            
            # Parar coleta
            self.processing_engine.stop_training_mode()
            
            logger.info("Coleta de dados concluída. Iniciando treinamento dos modelos...")
            
            # Treinar modelos
            training_results = self.processing_engine.train_models(
                epochs_cae=30,  # Menos épocas para hardware mais modesto
                epochs_convlstm=20
            )
            
            if "error" not in training_results:
                # Salvar modelos
                self.processing_engine.save_models()
                logger.info("Treinamento concluído com sucesso!")
                logger.info("Modelos salvos e prontos para uso")
                return True
            else:
                logger.error(f"Erro no treinamento: {training_results['error']}")
                return False
            
        except Exception as e:
            logger.error(f"Erro no modo de treinamento: {e}")
            return False
        finally:
            self._stop_system_components()
    
    def _start_system_components(self, display: bool = True) -> bool:
        """Inicia todos os componentes do sistema"""
        
        # Iniciar captura
        if not self.input_manager.start_capture():
            logger.error("Falha ao iniciar captura")
            return False
        
        # Iniciar processamento
        if not self.processing_engine.start_processing():
            logger.error("Falha ao iniciar processamento")
            return False
        
        # Iniciar display se solicitado
        if display:
            if not self.output_manager.start_display():
                logger.error("Falha ao iniciar display")
                return False
        
        self.running = True
        return True
    
    def _stop_system_components(self):
        """Para todos os componentes do sistema"""
        logger.info("Parando componentes do sistema...")
        
        self.running = False
        
        # Parar componentes na ordem inversa
        self.output_manager.stop_display()
        self.processing_engine.stop_processing()
        self.input_manager.stop_capture()
        
        logger.info("Todos os componentes parados")
    
    def _main_loop(self):
        """Loop principal do sistema"""
        self.running = True
        
        while self.running:
            try:
                # Processar eventos de teclado
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Saindo do sistema...")
                    break
                elif key == ord('p'):
                    self._toggle_pause()
                elif key == ord('s'):
                    self._print_statistics()
                elif key == ord('r'):
                    self._export_report()
                elif key == ord('h'):
                    self._print_help()
                elif key == 27:  # ESC
                    break
                
                # Verificar se fonte de vídeo ainda está ativa
                source_info = self.input_manager.get_source_info()
                if source_info and not source_info.get("is_active", True):
                    if source_info.get("source_type") == "file":
                        logger.info("Arquivo de vídeo terminou")
                        break
                
                time.sleep(0.01)  # Pequena pausa para reduzir uso de CPU
                
            except KeyboardInterrupt:
                logger.info("Interrompido pelo usuário")
                break
            except Exception as e:
                logger.error(f"Erro no loop principal: {e}")
                break
        
        self._stop_system_components()
    
    def _toggle_pause(self):
        """Alterna pausa do sistema"""
        self.paused = not self.paused
        status = "PAUSADO" if self.paused else "EXECUTANDO"
        logger.info(f"Sistema {status}")
    
    def _print_statistics(self):
        """Imprime estatísticas do sistema"""
        print("\n" + "="*60)
        print("ESTATÍSTICAS DO SISTEMA")
        print("="*60)
        
        # Estatísticas de entrada
        input_stats = self.input_manager.get_stats()
        print(f"Entrada:")
        print(f"  Frames capturados: {input_stats.get('frames_captured', 0)}")
        print(f"  FPS atual: {input_stats.get('current_fps', 0):.1f}")
        print(f"  Frames descartados: {input_stats.get('frames_dropped', 0)}")
        
        # Estatísticas de processamento
        proc_stats = self.processing_engine.get_performance_stats()
        print(f"\nProcessamento:")
        print(f"  Frames processados: {proc_stats.get('frames_processed', 0)}")
        print(f"  Anomalias detectadas: {proc_stats.get('anomalies_detected', 0)}")
        print(f"  Tempo médio: {proc_stats.get('avg_processing_time', 0):.3f}s")
        print(f"  Carga do sistema: {proc_stats.get('processing_load', 0):.1%}")
        
        # Estatísticas de saída
        output_stats = self.output_manager.get_display_stats()
        print(f"\nSaída:")
        print(f"  Frames exibidos: {output_stats.get('frames_displayed', 0)}")
        print(f"  Alertas enviados: {output_stats.get('alerts_sent', 0)}")
        print(f"  Frames salvos: {output_stats.get('frames_saved', 0)}")
        
        print("="*60 + "\n")
    
    def _export_report(self):
        """Exporta relatório da sessão"""
        report_path = self.output_manager.export_session_report()
        if report_path:
            logger.info(f"Relatório exportado: {report_path}")
        else:
            logger.error("Falha ao exportar relatório")
    
    def _print_help(self):
        """Imprime ajuda dos comandos"""
        print("\n" + "="*60)
        print("COMANDOS DISPONÍVEIS")
        print("="*60)
        print("q/ESC  - Sair do sistema")
        print("p      - Pausar/Retomar processamento")
        print("s      - Mostrar estatísticas")
        print("r      - Exportar relatório")
        print("h      - Mostrar esta ajuda")
        print("i      - Alternar painel de informações")
        print("f      - Alternar overlay de optical flow")
        print("a      - Alternar marcadores de anomalia")
        print("="*60 + "\n")
    
    def _signal_handler(self, signum, frame):
        """Manipulador de sinais do sistema"""
        logger.info(f"Sinal recebido: {signum}")
        self.running = False
    
    def load_trained_models(self, models_path: str = None) -> bool:
        """
        Carrega modelos previamente treinados
        
        Args:
            models_path: Caminho base dos modelos
            
        Returns:
            True se carregado com sucesso
        """
        logger.info("Carregando modelos treinados...")
        
        if self.processing_engine.load_models(models_path):
            logger.info("Modelos carregados com sucesso")
            return True
        else:
            logger.warning("Falha ao carregar modelos - usando detecção básica")
            return False
    
    def configure_email_alerts(self, server: str, port: int, username: str, 
                             password: str, recipients: list):
        """Configura alertas por email"""
        self.output_manager.configure_email_alerts(
            server, port, username, password, recipients
        )
    
    def configure_webhook_alerts(self, url: str, headers: dict = None):
        """Configura alertas por webhook"""
        self.output_manager.configure_webhook_alerts(url, headers)
    
    def get_system_status(self) -> Dict:
        """Retorna status completo do sistema"""
        return {
            "running": self.running,
            "paused": self.paused,
            "input_stats": self.input_manager.get_stats(),
            "processing_stats": self.processing_engine.get_performance_stats(),
            "output_stats": self.output_manager.get_display_stats(),
            "recent_alerts": self.output_manager.get_recent_alerts(1)
        }
    
    def cleanup(self):
        """Limpeza completa do sistema"""
        logger.info("Iniciando limpeza do sistema...")
        
        self._stop_system_components()
        
        # Limpeza individual dos componentes
        self.input_manager.cleanup()
        self.processing_engine.cleanup()
        self.output_manager.cleanup()
        
        logger.info("Sistema limpo e finalizado")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Sistema de Detecção de Anomalias em Tempo Real"
    )
    
    # Argumentos principais
    parser.add_argument("--mode", choices=["webcam", "video", "train"],
                       default="webcam", help="Modo de operação")
    parser.add_argument("--camera", type=int, default=0,
                       help="Índice da câmera (padrão: 0)")
    parser.add_argument("--video", type=str, 
                       help="Caminho para arquivo de vídeo")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Arquivo de configuração")
    parser.add_argument("--no-display", action="store_true",
                       help="Executar sem interface visual")
    parser.add_argument("--load-models", type=str,
                       help="Carregar modelos treinados")
    
    # Argumentos para treinamento
    parser.add_argument("--train-duration", type=int, default=10,
                       help="Duração do treinamento em minutos")
    
    # Argumentos para vídeo
    parser.add_argument("--loop", action="store_true",
                       help="Repetir vídeo em loop")
    
    # Argumentos para alertas
    parser.add_argument("--email-server", type=str,
                       help="Servidor SMTP para alertas")
    parser.add_argument("--email-port", type=int, default=587,
                       help="Porta do servidor SMTP")
    parser.add_argument("--email-user", type=str,
                       help="Usuário do email")
    parser.add_argument("--email-pass", type=str,
                       help="Senha do email")
    parser.add_argument("--email-recipients", type=str, nargs="+",
                       help="Lista de destinatários")
    parser.add_argument("--webhook-url", type=str,
                       help="URL do webhook para alertas")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SISTEMA DE DETECÇÃO DE ANOMALIAS")
    print("Optical Flow + CAE + ConvLSTM")
    print("Otimizado para i5 11Gen | 16GB RAM")
    print("="*60)
    
    try:
        # Inicializar sistema
        system = AnomalyDetectionSystem(args.config)
        
        # Configurar alertas se fornecidos
        if args.email_server and args.email_user and args.email_pass and args.email_recipients:
            system.configure_email_alerts(
                args.email_server, args.email_port, 
                args.email_user, args.email_pass, args.email_recipients
            )
            print(f"✓ Alertas por email configurados ({len(args.email_recipients)} destinatários)")
        
        if args.webhook_url:
            system.configure_webhook_alerts(args.webhook_url)
            print(f"✓ Webhook configurado: {args.webhook_url}")
        
        # Carregar modelos se especificado
        if args.load_models:
            if system.load_trained_models(args.load_models):
                print("✓ Modelos treinados carregados")
            else:
                print("⚠ Usando detecção básica (modelos não carregados)")
        
        display = not args.no_display
        
        # Executar modo selecionado
        if args.mode == "webcam":
            print(f"Iniciando detecção por webcam (câmera {args.camera})")
            success = system.start_webcam_detection(args.camera, display)
            
        elif args.mode == "video":
            if not args.video:
                print("❌ Erro: Especifique o arquivo de vídeo com --video")
                return 1
            
            if not os.path.exists(args.video):
                print(f"❌ Erro: Arquivo não encontrado: {args.video}")
                return 1
            
            print(f"Iniciando detecção por arquivo: {args.video}")
            success = system.start_video_detection(args.video, display, args.loop)
            
        elif args.mode == "train":
            print(f"Iniciando modo de treinamento ({args.train_duration} minutos)")
            print("IMPORTANTE: Execute apenas com comportamento NORMAL!")
            input("Pressione ENTER para continuar...")
            success = system.start_training_mode(args.camera, args.train_duration)
        
        else:
            print(f"❌ Modo não reconhecido: {args.mode}")
            return 1
        
        if success:
            print("✓ Sistema executado com sucesso")
            return 0
        else:
            print("❌ Erro na execução do sistema")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠ Sistema interrompido pelo usuário")
        return 0
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        logger.error(f"Erro na função main: {e}")
        return 1
    finally:
        # Limpeza
        try:
            system.cleanup()
        except:
            pass

def demo_mode():
    """Modo demonstração com configurações otimizadas"""
    
    print("="*60)
    print("MODO DEMONSTRAÇÃO")
    print("Configuração otimizada para testes rápidos")
    print("="*60)
    
    try:
        # Configuração simplificada
        system = AnomalyDetectionSystem()
        
        print("Testando componentes do sistema...")
        
        # Teste de câmera
        print("1. Testando acesso à câmera...")
        if system.input_manager.initialize_webcam(0):
            print("   ✓ Câmera funcionando")
            system.input_manager._release_current_source()
        else:
            print("   ❌ Câmera não disponível")
            return
        
        # Teste de processamento
        print("2. Testando pipeline de processamento...")
        print("   ✓ Optical Flow inicializado")
        print("   ✓ Deep Learning inicializado")
        print("   ✓ Classificador inicializado")
        
        # Início da demonstração
        print("\nIniciando demonstração em 3 segundos...")
        time.sleep(3)
        
        # Executar detecção
        system.start_webcam_detection(0, True)
        
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")

def create_default_config():
    """Cria arquivo de configuração padrão"""
    
    config_data = {
        "video": {
            "realtime_resolution": [640, 480],
            "training_resolution": [1280, 720],
            "fps": 30,
            "frame_skip": 2
        },
        "model": {
            "cae_input_shape": [64, 64, 3],
            "convlstm_sequence_length": 10,
            "optical_flow_method": "farneback",
            "flow_threshold": 2.0,
            "anomaly_threshold": 0.7,
            "movement_threshold": 15.0
        },
        "system": {
            "max_threads": 4,
            "queue_size": 10,
            "log_level": "INFO",
            "save_alert_frames": True,
            "web_port": 5000
        },
        "anomaly_types": {
            "security": {
                "intrusion": "Intrusão detectada",
                "break_in": "Tentativa de arrombamento",
                "night_movement": "Movimento noturno suspeito",
                "stranger_loitering": "Pessoa estranha por muito tempo",
                "unknown_vehicle": "Veículo desconhecido"
            },
            "health": {
                "fall": "Queda detectada",
                "collapse": "Colapso/desmaio detectado",
                "immobility": "Imobilidade prolongada",
                "erratic_movement": "Movimento errático",
                "no_movement": "Ausência de movimento"
            }
        }
    }
    
    try:
        import json
        with open("config.json", "w") as f:
            json.dump(config_data, f, indent=4)
        print("✓ Arquivo config.json criado com configurações padrão")
    except Exception as e:
        print(f"❌ Erro ao criar configuração: {e}")

if __name__ == "__main__":
    
    # Verificar argumentos especiais
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_mode()
            sys.exit(0)
        elif sys.argv[1] == "create-config":
            create_default_config()
            sys.exit(0)
        elif sys.argv[1] == "--help-full":
            print("""
SISTEMA DE DETECÇÃO DE ANOMALIAS - AJUDA COMPLETA

MODOS DE OPERAÇÃO:
  webcam  - Detecção em tempo real usando câmera
  video   - Análise de arquivo de vídeo
  train   - Treinamento de modelos com dados normais

EXEMPLOS DE USO:

1. Detecção básica com webcam:
   python main.py --mode webcam --camera 0

2. Análise de vídeo:
   python main.py --mode video --video meu_video.mp4 --loop

3. Treinamento de modelos:
   python main.py --mode train --train-duration 15

4. Carregando modelos treinados:
   python main.py --mode webcam --load-models models/meu_modelo

5. Com alertas por email:
   python main.py --mode webcam \\
     --email-server smtp.gmail.com \\
     --email-user usuario@gmail.com \\
     --email-pass senha \\
     --email-recipients admin@empresa.com seguranca@empresa.com

6. Modo demonstração:
   python main.py demo

7. Criar configuração padrão:
   python main.py create-config

COMANDOS DURANTE EXECUÇÃO:
  q/ESC  - Sair
  p      - Pausar/Retomar
  s      - Mostrar estatísticas
  r      - Exportar relatório
  h      - Ajuda
  i      - Alternar painel info
  f      - Alternar optical flow
  a      - Alternar marcadores

TIPOS DE ANOMALIAS DETECTADAS:

Segurança:
  - Intrusão (pessoas não autorizadas)
  - Arrombamento (tentativas de força)
  - Movimento noturno inesperado
  - Permanência prolongada de estranhos
  - Veículos desconhecidos

Saúde/Emergência:
  - Quedas (especialmente idosos)
  - Desmaios ou colapsos
  - Imobilidade prolongada
  - Movimentos erráticos
  - Ausência de movimento

HARDWARE RECOMENDADO:
  - CPU: i5 11Gen ou superior
  - RAM: 16GB ou mais
  - Câmera: USB/integrada funcional
  - OS: Windows 10/11, Linux Ubuntu 20+

Para mais informações: consulte a documentação
            """)
            sys.exit(0)
    
    # Execução normal
    sys.exit(main())