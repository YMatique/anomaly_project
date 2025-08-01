#!/usr/bin/env python3
"""
Script de Setup Automático para Sistema de Detecção de Anomalias
Instala dependências, cria estrutura de diretórios e configura o sistema
"""

import os
import sys
import subprocess
import json
import platform
from pathlib import Path

def print_header():
    """Exibe cabeçalho do instalador"""
    print("="*60)
    print("🚨 SISTEMA DE DETECÇÃO DE ANOMALIAS")
    print("   Setup Automático e Configuração")
    print("   Optical Flow + CAE + ConvLSTM")
    print("="*60)
    print()

def check_python_version():
    """Verifica versão do Python"""
    print("🐍 Verificando versão do Python...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detectado")
        print("   ⚠️  Requer Python 3.8 ou superior")
        print("   📥 Baixe em: https://python.org/downloads/")
        return False
    
    print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_system_requirements():
    """Verifica requisitos do sistema"""
    print("💻 Verificando sistema...")
    
    system = platform.system()
    machine = platform.machine()
    
    print(f"   Sistema: {system} {platform.release()}")
    print(f"   Arquitetura: {machine}")
    
    # Verificar memória (se disponível)
    try:
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   Memória: {memory_gb:.1f}GB")
        
        if memory_gb < 4:
            print("   ⚠️  Recomendado: 8GB+ RAM para melhor performance")
        elif memory_gb >= 16:
            print("   ✅ Memória adequada para performance máxima")
        else:
            print("   ✅ Memória adequada")
            
    except ImportError:
        print("   ℹ️  Instale psutil para verificação detalhada de recursos")
    
    return True

def create_directory_structure():
    """Cria estrutura de diretórios"""
    print("📁 Criando estrutura de diretórios...")
    
    directories = [
        "src/core",
        "src/detectors", 
        "src/utils",
        "src/web/templates",
        "models",
        "data/videos",
        "data/logs",
        "data/alerts",
        "data/captures", 
        "data/reports",
        "data/training",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")
    
    print("   📁 Estrutura de diretórios criada!")

def install_dependencies():
    """Instala dependências Python"""
    print("📦 Instalando dependências...")
    
    # Verificar se pip está disponível
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("   ❌ pip não encontrado")
        print("   📥 Instale pip: https://pip.pypa.io/en/stable/installation/")
        return False
    
    # Atualizar pip
    print("   🔄 Atualizando pip...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True)
        print("   ✅ pip atualizado")
    except subprocess.CalledProcessError:
        print("   ⚠️  Falha ao atualizar pip (continuando...)")
    
    # Instalar requirements
    if not os.path.exists("requirements.txt"):
        print("   ❌ requirements.txt não encontrado")
        return False
    
    print("   📥 Instalando pacotes...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("   ✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Erro na instalação: {e}")
        print("   🔧 Tente instalar manualmente: pip install -r requirements.txt")
        return False

def create_default_config():
    """Cria arquivo de configuração padrão"""
    print("⚙️  Criando configuração padrão...")
    
    config_data = {
        "video": {
            "realtime_resolution": [640, 480],
            "training_resolution": [1280, 720],
            "fps": 30,
            "frame_skip": 2,
            "buffer_size": 5
        },
        "model": {
            "cae_input_shape": [64, 64, 3],
            "cae_latent_dim": 128,
            "cae_batch_size": 16,
            "convlstm_sequence_length": 10,
            "convlstm_filters": 64,
            "optical_flow_method": "farneback",
            "flow_threshold": 2.0,
            "anomaly_threshold": 0.7,
            "movement_threshold": 15.0
        },
        "system": {
            "max_threads": 4,
            "queue_size": 10,
            "log_level": "INFO",
            "log_file": "data/logs/anomaly_detection.log",
            "save_alert_frames": True,
            "alert_cooldown": 5,
            "models_path": "models/",
            "data_path": "data/",
            "web_host": "127.0.0.1",
            "web_port": 5000,
            "web_debug": True
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
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        print("   ✅ config.json criado")
        return True
    except Exception as e:
        print(f"   ❌ Erro ao criar config.json: {e}")
        return False

def test_camera_access():
    """Testa acesso à câmera"""
    print("📷 Testando acesso à câmera...")
    
    try:
        import cv2
        
        # Testar câmeras de 0 a 3
        working_cameras = []
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    working_cameras.append(i)
                    print(f"   ✅ Câmera {i}: Funcionando")
                else:
                    print(f"   ⚠️  Câmera {i}: Detectada mas sem vídeo")
                cap.release()
            else:
                print(f"   ❌ Câmera {i}: Não disponível")
        
        if working_cameras:
            print(f"   📷 {len(working_cameras)} câmera(s) funcionando: {working_cameras}")
            return True
        else:
            print("   ⚠️  Nenhuma câmera funcional detectada")
            print("   ℹ️  Você ainda pode usar arquivos de vídeo")
            return False
            
    except ImportError:
        print("   ⚠️  OpenCV não instalado - teste de câmera pulado")
        return False
    except Exception as e:
        print(f"   ❌ Erro no teste de câmera: {e}")
        return False

def create_sample_scripts():
    """Cria scripts de exemplo"""
    print("📝 Criando scripts de exemplo...")
    
    # Script de teste rápido
    test_script = '''#!/usr/bin/env python3
"""Script de teste rápido do sistema"""

import sys
import os
sys.path.append('.')

def test_imports():
    """Testa importações principais"""
    try:
        import cv2
        print("✅ OpenCV:", cv2.__version__)
    except ImportError as e:
        print("❌ OpenCV:", e)
        return False
    
    try:
        import tensorflow as tf
        print("✅ TensorFlow:", tf.__version__)
    except ImportError as e:
        print("❌ TensorFlow:", e)
        return False
    
    try:
        import numpy as np
        print("✅ NumPy:", np.__version__)
    except ImportError as e:
        print("❌ NumPy:", e)
        return False
    
    return True

def test_system():
    """Testa componentes do sistema"""
    try:
        from src.utils.config import Config
        from src.utils.logger import logger
        
        config = Config()
        logger.info("Sistema testado com sucesso!")
        print("✅ Sistema: Componentes carregados")
        return True
    except Exception as e:
        print(f"❌ Sistema: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Teste Rápido do Sistema")
    print("=" * 30)
    
    if test_imports() and test_system():
        print("\\n🎉 Todos os testes passaram!")
        print("Sistema pronto para uso.")
    else:
        print("\\n❌ Alguns testes falharam.")
        print("Verifique a instalação.")
'''
    
    try:
        with open("test_system.py", "w", encoding="utf-8") as f:
            f.write(test_script)
        print("   ✅ test_system.py criado")
    except Exception as e:
        print(f"   ❌ Erro ao criar test_system.py: {e}")
    
    # Script de inicialização rápida  
    quick_start = '''#!/usr/bin/env python3
"""Script de inicialização rápida"""

import subprocess
import sys

def main():
    print("🚀 Inicialização Rápida - Sistema de Detecção de Anomalias")
    print()
    
    print("Escolha uma opção:")
    print("1. Detecção com Webcam")
    print("2. Interface Web")
    print("3. Modo Demonstração")
    print("4. Treinamento (15 min)")
    print("5. Teste do Sistema")
    print("0. Sair")
    print()
    
    choice = input("Digite sua escolha (0-5): ").strip()
    
    if choice == "1":
        subprocess.run([sys.executable, "main.py", "--mode", "webcam"])
    elif choice == "2":
        print("Iniciando interface web...")
        print("Acesse: http://localhost:5000")
        subprocess.run([sys.executable, "src/web/app.py"])
    elif choice == "3":
        subprocess.run([sys.executable, "main.py", "demo"])
    elif choice == "4":
        subprocess.run([sys.executable, "main.py", "--mode", "train", "--train-duration", "15"])
    elif choice == "5":
        subprocess.run([sys.executable, "test_system.py"])
    elif choice == "0":
        print("Saindo...")
    else:
        print("Opção inválida!")

if __name__ == "__main__":
    main()
'''
    
    try:
        with open("quick_start.py", "w", encoding="utf-8") as f:
            f.write(quick_start)
        print("   ✅ quick_start.py criado")
    except Exception as e:
        print(f"   ❌ Erro ao criar quick_start.py: {e}")

def optimize_for_hardware():
    """Otimiza configurações para o hardware detectado"""
    print("🔧 Otimizando para hardware...")
    
    try:
        import psutil
        
        # Detectar recursos
        cpu_count = psutil.cpu_count(logical=False)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        print(f"   CPU: {cpu_count} cores físicos")
        print(f"   RAM: {memory_gb:.1f}GB")
        
        # Carregar configuração existente
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                config = json.load(f)
        else:
            return False
        
        # Otimizar baseado no hardware
        if cpu_count >= 4:
            config["system"]["max_threads"] = min(cpu_count, 4)
            print(f"   ✅ Threads otimizadas: {config['system']['max_threads']}")
        
        if memory_gb >= 16:
            config["model"]["cae_batch_size"] = 32
            config["video"]["realtime_resolution"] = [800, 600]
            print("   ✅ Configurações para alta performance")
        elif memory_gb >= 8:
            config["model"]["cae_batch_size"] = 16
            config["video"]["realtime_resolution"] = [640, 480]
            print("   ✅ Configurações balanceadas")
        else:
            config["model"]["cae_batch_size"] = 8
            config["video"]["realtime_resolution"] = [480, 360]
            config["video"]["frame_skip"] = 3
            print("   ✅ Configurações para hardware limitado")
        
        # Salvar configurações otimizadas
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        print("   ✅ Configurações otimizadas salvas")
        return True
        
    except ImportError:
        print("   ⚠️  psutil não disponível - otimização manual necessária")
        return False
    except Exception as e:
        print(f"   ❌ Erro na otimização: {e}")
        return False

def show_usage_instructions():
    """Exibe instruções de uso"""
    print("📖 Instruções de Uso")
    print("=" * 30)
    print()
    print("🚀 INICIALIZAÇÃO RÁPIDA:")
    print("   python quick_start.py")
    print()
    print("🖥️  MODO CONSOLE:")
    print("   # Webcam básica")
    print("   python main.py --mode webcam")
    print()
    print("   # Análise de vídeo")
    print("   python main.py --mode video --video meu_video.mp4")
    print()
    print("   # Treinamento (comportamento normal apenas!)")
    print("   python main.py --mode train --train-duration 10")
    print()
    print("🌐 INTERFACE WEB:")
    print("   python src/web/app.py")
    print("   Acesse: http://localhost:5000")
    print()
    print("🧪 TESTE DO SISTEMA:")
    print("   python test_system.py")
    print()
    print("📁 ARQUIVOS IMPORTANTES:")
    print("   config.json       - Configurações")
    print("   data/logs/        - Logs do sistema")
    print("   data/alerts/      - Frames de alerta")
    print("   models/           - Modelos treinados")
    print()
    print("⌨️  CONTROLES (durante execução):")
    print("   Q/ESC - Sair       P - Pausar")
    print("   S - Estatísticas   R - Relatório")
    print("   H - Ajuda         I - Toggle Info")
    print()

def main():
    """Função principal do setup"""
    print_header()
    
    # Verificações preliminares
    if not check_python_version():
        input("Pressione ENTER para sair...")
        return 1
    
    if not check_system_requirements():
        print("⚠️  Problemas nos requisitos do sistema detectados")
    
    # Criação da estrutura
    create_directory_structure()
    
    # Instalação de dependências
    if not install_dependencies():
        print("❌ Falha na instalação das dependências")
        print("   Tente instalar manualmente: pip install -r requirements.txt")
        input("Pressione ENTER para continuar mesmo assim...")
    
    # Configuração
    create_default_config()
    optimize_for_hardware()
    
    # Testes
    test_camera_access()
    
    # Scripts auxiliares
    create_sample_scripts()
    
    # Finalização
    print()
    print("🎉 SETUP CONCLUÍDO COM SUCESSO!")
    print("=" * 40)
    print()
    
    show_usage_instructions()
    
    # Pergunta se quer testar
    print("🧪 Deseja executar o teste do sistema agora? (s/n): ", end="")
    if input().lower().startswith('s'):
        print()
        try:
            import subprocess
            subprocess.run([sys.executable, "test_system.py"])
        except Exception as e:
            print(f"Erro no teste: {e}")
    
    print()
    print("🚀 Sistema pronto para uso!")
    print("   Execute: python quick_start.py")
    print()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado no setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)