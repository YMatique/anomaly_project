"""
Detector de Deep Learning usando CAE (Convolutional Autoencoder) e ConvLSTM
Segunda camada de detecção para análise mais profunda de anomalias
Otimizado para i5 11Gen com 16GB RAM
VERSÃO CORRIGIDA COM LIMITES DE FRAMES
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import pickle
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import time
import threading
from queue import Queue
from tqdm import tqdm

from ..utils.helpers import time_function, VideoProcessor
from ..utils.logger import logger

def normalize_batch_fixed(data: np.ndarray, params: Dict = None) -> Tuple[np.ndarray, Dict]:
    """Função corrigida para normalizar dados"""
    if params is None:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std = np.where(std == 0, 1, std)  # Evitar divisão por zero
        params = {"mean": mean, "std": std}
    else:
        mean = params["mean"]
        std = params["std"]
    
    normalized = (data - mean) / std
    return normalized, params

def create_sequences_fixed(data: np.ndarray, sequence_length: int, overlap: int = 1) -> np.ndarray:
    """Cria sequências de frames para treinamento"""
    sequences = []
    step = max(1, sequence_length - overlap)
    
    for i in range(0, len(data) - sequence_length + 1, step):
        sequences.append(data[i:i + sequence_length])
    
    return np.array(sequences)

class ConvolutionalAutoencoder:
    """
    Convolutional Autoencoder otimizado para detecção de anomalias
    Treina em comportamentos normais e detecta anomalias por erro de reconstrução
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (64, 64, 3)):
        self.input_shape = input_shape
        self.model = None
        self.encoder = None
        self.decoder = None
        self.threshold = 0.0
        self.normalization_params = None
        
        # Configurações otimizadas para hardware
        self.batch_size = 16
        self.learning_rate = 0.001
        
        self._build_model()
        logger.info(f"CAE inicializado - input shape: {input_shape}")
    
    def _build_model(self):
        """Constrói arquitetura do autoencoder otimizada"""
        
        # Input
        input_layer = layers.Input(shape=self.input_shape)
        
        # Encoder (compressão)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder (reconstrução)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
        
        # Modelo completo
        self.model = keras.Model(input_layer, decoded)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Encoder separado para extração de features
        self.encoder = keras.Model(input_layer, encoded)
        
        logger.info("Arquitetura CAE construída")
    
    def train(self, training_data: np.ndarray, validation_data: Optional[np.ndarray] = None,
              epochs: int = 50, save_path: str = None) -> Dict:
        """
        Treina o autoencoder com dados normais
        
        Args:
            training_data: Array de frames normais [N, H, W, C]
            validation_data: Dados de validação opcionais
            epochs: Número de épocas
            save_path: Caminho para salvar modelo
        """
        logger.info(f"Iniciando treinamento CAE - {len(training_data)} samples, {epochs} epochs")
        
        # Normalizar dados - CORRIGIDO
        normalized_data, self.normalization_params = normalize_batch_fixed(training_data)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        if save_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    save_path, save_best_only=True, monitor='val_loss' if validation_data is not None else 'loss'
                )
            )
        
        # Treinamento
        validation_data_norm = None
        if validation_data is not None:
            validation_data_norm = normalize_batch_fixed(validation_data, self.normalization_params)[0]
        
        history = self.model.fit(
            normalized_data, normalized_data,  # Autoencoder: input = output
            epochs=epochs,
            batch_size=self.batch_size,
            validation_data=(validation_data_norm, validation_data_norm) if validation_data_norm is not None else None,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calcular threshold baseado nos dados de treinamento
        self._calculate_threshold(normalized_data)
        
        logger.info(f"Treinamento CAE concluído - threshold: {self.threshold:.4f}")
        
        return {
            "history": history.history,
            "threshold": self.threshold,
            "final_loss": history.history['loss'][-1]
        }
    
    def _calculate_threshold(self, training_data: np.ndarray):
        """Calcula threshold baseado nos erros de reconstrução dos dados normais"""
        predictions = self.model.predict(training_data, batch_size=self.batch_size)
        reconstruction_errors = np.mean((training_data - predictions) ** 2, axis=(1, 2, 3))
        
        # Threshold = média + 2 * desvio padrão
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        self.threshold = mean_error + 2 * std_error
        
        logger.info(f"Threshold calculado: {self.threshold:.4f} (média: {mean_error:.4f}, std: {std_error:.4f})")
    
    def predict(self, frame: np.ndarray) -> Dict:
        """
        Prediz se um frame é anômalo
        
        Args:
            frame: Frame a ser analisado [H, W, C]
            
        Returns:
            Dict com informações da predição
        """
        if self.model is None:
            logger.error("Modelo CAE não foi treinado")
            return {"error": "Modelo não treinado"}
        
        # Preprocessar frame
        processed_frame = VideoProcessor.preprocess_frame(frame, self.input_shape[:2])
        if processed_frame is None:
            return {"error": "Erro no preprocessamento"}
        
        # Normalizar
        if self.normalization_params:
            processed_frame = (processed_frame - self.normalization_params["mean"]) / self.normalization_params["std"]
        
        # Adicionar dimensão batch
        input_batch = np.expand_dims(processed_frame, axis=0)
        
        # Predição
        reconstruction = self.model.predict(input_batch, verbose=0)
        
        # Calcular erro de reconstrução
        reconstruction_error = np.mean((input_batch - reconstruction) ** 2)
        
        # Detectar anomalia
        is_anomaly = reconstruction_error > self.threshold
        confidence = min(reconstruction_error / self.threshold, 3.0) if self.threshold > 0 else 0.0
        
        return {
            "reconstruction_error": float(reconstruction_error),
            "threshold": float(self.threshold),
            "is_anomaly": bool(is_anomaly),
            "confidence": float(confidence),
            "reconstructed_frame": reconstruction[0]
        }
    
    def save_model(self, path: str):
        """Salva modelo e parâmetros"""
        if self.model is not None:
            self.model.save(f"{path}_cae.h5")
            
            # Salvar parâmetros adicionais
            params = {
                "threshold": self.threshold,
                "normalization_params": self.normalization_params,
                "input_shape": self.input_shape
            }
            
            with open(f"{path}_cae_params.pkl", "wb") as f:
                pickle.dump(params, f)
            
            logger.info(f"Modelo CAE salvo: {path}")
    
    def load_model(self, path: str):
        """Carrega modelo e parâmetros"""
        try:
            self.model = keras.models.load_model(f"{path}_cae.h5")
            
            with open(f"{path}_cae_params.pkl", "rb") as f:
                params = pickle.load(f)
            
            self.threshold = params["threshold"]
            self.normalization_params = params["normalization_params"]
            self.input_shape = params["input_shape"]
            
            # Recriar encoder
            self.encoder = keras.Model(self.model.input, self.model.layers[6].output)
            
            logger.info(f"Modelo CAE carregado: {path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar modelo CAE: {e}")
            return False

class ConvLSTMDetector:
    """
    ConvLSTM para análise de sequências temporais
    Detecta padrões anômalos em sequências de movimento
    """
    
    def __init__(self, input_shape: Tuple[int, int, int, int] = (10, 64, 64, 3)):
        self.input_shape = input_shape  # (timesteps, height, width, channels)
        self.model = None
        self.threshold = 0.0
        self.sequence_buffer = deque(maxlen=input_shape[0])
        self.normalization_params = None
        
        # Configurações
        self.batch_size = 8  # Menor para sequências
        self.learning_rate = 0.001
        
        self._build_model()
        logger.info(f"ConvLSTM inicializado - input shape: {input_shape}")
    
    def _build_model(self):
        """Constrói arquitetura ConvLSTM"""
        
        # Input
        input_layer = layers.Input(shape=self.input_shape)
        
        # ConvLSTM layers
        x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='relu'
        )(input_layer)
        x = layers.BatchNormalization()(x)
        
        x = layers.ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='relu'
        )(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.ConvLSTM2D(
            filters=16,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=False,
            activation='relu'
        )(x)
        
        # Decoder para reconstrução do último frame
        x = layers.Reshape((self.input_shape[1], self.input_shape[2], 16))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        output = layers.Conv2D(self.input_shape[3], (3, 3), activation='sigmoid', padding='same')(x)
        
        self.model = keras.Model(input_layer, output)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Arquitetura ConvLSTM construída")
    
    def add_frame(self, frame: np.ndarray):
        """Adiciona frame ao buffer de sequência"""
        processed_frame = VideoProcessor.preprocess_frame(frame, self.input_shape[1:3])
        if processed_frame is not None:
            self.sequence_buffer.append(processed_frame)
    
    def can_predict(self) -> bool:
        """Verifica se tem frames suficientes para predição"""
        return len(self.sequence_buffer) == self.input_shape[0]
    
    def predict(self) -> Dict:
        """
        Prediz anomalia baseada na sequência atual
        """
        if not self.can_predict():
            return {"error": "Sequência incompleta"}
        
        if self.model is None:
            return {"error": "Modelo não treinado"}
        
        # Preparar sequência
        sequence = np.array(list(self.sequence_buffer))
        
        # Normalizar
        if self.normalization_params:
            sequence = (sequence - self.normalization_params["mean"]) / self.normalization_params["std"]
        
        # Adicionar dimensão batch
        input_batch = np.expand_dims(sequence, axis=0)
        
        # Predição (reconstrói último frame)
        reconstruction = self.model.predict(input_batch, verbose=0)
        
        # Calcular erro baseado no último frame
        last_frame = sequence[-1]
        reconstruction_error = np.mean((last_frame - reconstruction[0]) ** 2)
        
        # Detectar anomalia
        is_anomaly = reconstruction_error > self.threshold
        confidence = min(reconstruction_error / self.threshold, 3.0) if self.threshold > 0 else 0.0
        
        # Análise temporal adicional
        temporal_analysis = self._analyze_temporal_sequence(sequence)
        
        return {
            "reconstruction_error": float(reconstruction_error),
            "threshold": float(self.threshold),
            "is_anomaly": bool(is_anomaly),
            "confidence": float(confidence),
            "temporal_analysis": temporal_analysis,
            "reconstructed_frame": reconstruction[0]
        }
    
    def _analyze_temporal_sequence(self, sequence: np.ndarray) -> Dict:
        """Analisa padrões temporais na sequência"""
        
        # Calcular diferenças entre frames consecutivos
        frame_diffs = []
        for i in range(1, len(sequence)):
            diff = np.mean((sequence[i] - sequence[i-1]) ** 2)
            frame_diffs.append(diff)
        
        temporal_info = {
            "avg_frame_diff": float(np.mean(frame_diffs)) if frame_diffs else 0.0,
            "max_frame_diff": float(np.max(frame_diffs)) if frame_diffs else 0.0,
            "temporal_consistency": float(1.0 / (1.0 + np.std(frame_diffs))) if frame_diffs else 1.0,
            "trend": "stable"
        }
        
        # Detectar tendência
        if len(frame_diffs) > 3:
            if np.mean(frame_diffs[-3:]) > np.mean(frame_diffs[:3]) * 1.5:
                temporal_info["trend"] = "increasing_change"
            elif np.mean(frame_diffs[-3:]) < np.mean(frame_diffs[:3]) * 0.5:
                temporal_info["trend"] = "decreasing_change"
        
        return temporal_info
    
    def train(self, sequences: np.ndarray, validation_sequences: Optional[np.ndarray] = None,
              epochs: int = 30, save_path: str = None) -> Dict:
        """Treina ConvLSTM com sequências normais"""
        
        logger.info(f"Iniciando treinamento ConvLSTM - {len(sequences)} sequências")
        
        # Normalizar - CORRIGIDO
        normalized_sequences, self.normalization_params = normalize_batch_fixed(sequences)
        
        # Target = último frame de cada sequência
        targets = normalized_sequences[:, -1, :, :, :]
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        if save_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(save_path, save_best_only=True)
            )
        
        # Preparar dados de validação
        validation_data = None
        if validation_sequences is not None:
            val_normalized = normalize_batch_fixed(validation_sequences, self.normalization_params)[0]
            val_targets = val_normalized[:, -1, :, :, :]
            validation_data = (val_normalized, val_targets)
        
        # Treinamento
        history = self.model.fit(
            normalized_sequences, targets,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calcular threshold
        self._calculate_threshold(normalized_sequences, targets)
        
        logger.info(f"Treinamento ConvLSTM concluído - threshold: {self.threshold:.4f}")
        
        return {
            "history": history.history,
            "threshold": self.threshold,
            "final_loss": history.history['loss'][-1]
        }
    
    def _calculate_threshold(self, sequences: np.ndarray, targets: np.ndarray):
        """Calcula threshold baseado nos erros de reconstrução"""
        predictions = self.model.predict(sequences, batch_size=self.batch_size)
        reconstruction_errors = np.mean((targets - predictions) ** 2, axis=(1, 2, 3))
        
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        self.threshold = mean_error + 2 * std_error
    
    def save_model(self, path: str):
        """Salva modelo ConvLSTM"""
        if self.model is not None:
            self.model.save(f"{path}_convlstm.h5")
            
            params = {
                "threshold": self.threshold,
                "normalization_params": self.normalization_params,
                "input_shape": self.input_shape
            }
            
            with open(f"{path}_convlstm_params.pkl", "wb") as f:
                pickle.dump(params, f)
            
            logger.info(f"Modelo ConvLSTM salvo: {path}")
    
    def load_model(self, path: str):
        """Carrega modelo ConvLSTM"""
        try:
            self.model = keras.models.load_model(f"{path}_convlstm.h5")
            
            with open(f"{path}_convlstm_params.pkl", "rb") as f:
                params = pickle.load(f)
            
            self.threshold = params["threshold"]
            self.normalization_params = params["normalization_params"]
            self.input_shape = params["input_shape"]
            
            logger.info(f"Modelo ConvLSTM carregado: {path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar ConvLSTM: {e}")
            return False

class DeepLearningDetector:
    """
    Detector principal que combina CAE e ConvLSTM
    Implementa arquitetura em cascata otimizada
    """
    
    def __init__(self, config):
        self.config = config
        
        # Inicializar detectores
        cae_shape = config.model.cae_input_shape
        convlstm_shape = (
            config.model.convlstm_sequence_length,
            cae_shape[0], cae_shape[1], cae_shape[2]
        )
        
        self.cae = ConvolutionalAutoencoder(cae_shape)
        self.convlstm = ConvLSTMDetector(convlstm_shape)
        
        # Configurações
        self.anomaly_threshold = config.model.anomaly_threshold
        self.models_path = config.system.models_path
        
        # Estado
        self.is_trained = False
        self.training_mode = False
        self.training_data = []
        
        # Threading para processamento assíncrono
        self.processing_queue = Queue(maxsize=10)
        self.result_queue = Queue()
        self.processing_thread = None
        self.processing_active = False
        
        logger.info("DeepLearningDetector inicializado")
    
    @time_function
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detecta anomalias usando CAE e ConvLSTM em cascata
        
        Args:
            frame: Frame a ser analisado
            
        Returns:
            Dict com resultados da detecção
        """
        if frame is None:
            return {"error": "Frame inválido"}
        
        results = {
            "timestamp": time.time(),
            "cae_result": {},
            "convlstm_result": {},
            "final_decision": {
                "is_anomaly": False,
                "confidence": 0.0,
                "anomaly_type": "none"
            }
        }
        
        # Primeira camada: CAE
        if self.cae.model is not None:
            cae_result = self.cae.predict(frame)
            results["cae_result"] = cae_result
            
            # Se CAE detecta anomalia, prosseguir para ConvLSTM
            if cae_result.get("is_anomaly", False):
                
                # Adicionar frame ao buffer ConvLSTM
                self.convlstm.add_frame(frame)
                
                # Se tem sequência completa, analisar com ConvLSTM
                if self.convlstm.can_predict() and self.convlstm.model is not None:
                    convlstm_result = self.convlstm.predict()
                    results["convlstm_result"] = convlstm_result
                    
                    # Decisão final baseada em ambos
                    results["final_decision"] = self._make_final_decision(
                        cae_result, convlstm_result
                    )
                else:
                    # Usar apenas CAE se ConvLSTM não está pronto
                    results["final_decision"] = {
                        "is_anomaly": cae_result["is_anomaly"],
                        "confidence": cae_result["confidence"] * 0.7,  # Reduzir confiança
                        "anomaly_type": "spatial_only"
                    }
            else:
                # CAE não detectou anomalia
                self.convlstm.add_frame(frame)  # Ainda adicionar ao buffer
                results["final_decision"] = {
                    "is_anomaly": False,
                    "confidence": 0.0,
                    "anomaly_type": "none"
                }
        
        # Modo de treinamento
        if self.training_mode:
            self.training_data.append(frame.copy())
            if len(self.training_data) > 1000:  # Limitar buffer de treinamento
                self.training_data = self.training_data[-1000:]
        
        return results
    
    def _make_final_decision(self, cae_result: Dict, convlstm_result: Dict) -> Dict:
        """Combina resultados do CAE e ConvLSTM para decisão final"""
        
        cae_anomaly = cae_result.get("is_anomaly", False)
        cae_confidence = cae_result.get("confidence", 0.0)
        
        convlstm_anomaly = convlstm_result.get("is_anomaly", False)
        convlstm_confidence = convlstm_result.get("confidence", 0.0)
        
        # Lógica de fusão
        if cae_anomaly and convlstm_anomaly:
            # Ambos detectaram - alta confiança
            final_confidence = min((cae_confidence + convlstm_confidence) / 2 * 1.2, 1.0)
            anomaly_type = "spatiotemporal"
        elif cae_anomaly:
            # Apenas CAE - confiança média
            final_confidence = cae_confidence * 0.8
            anomaly_type = "spatial"
        elif convlstm_anomaly:
            # Apenas ConvLSTM - confiança baixa (raro, pois CAE é filtro)
            final_confidence = convlstm_confidence * 0.6
            anomaly_type = "temporal"
        else:
            # Nenhum detectou
            final_confidence = 0.0
            anomaly_type = "none"
        
        is_final_anomaly = final_confidence > self.anomaly_threshold
        
        return {
            "is_anomaly": is_final_anomaly,
            "confidence": final_confidence,
            "anomaly_type": anomaly_type,
            "cae_contribution": cae_confidence,
            "convlstm_contribution": convlstm_confidence
        }
    
    def start_training_mode(self):
        """Inicia modo de coleta de dados para treinamento"""
        self.training_mode = True
        self.training_data = []
        logger.info("Modo de treinamento iniciado")
    
    def stop_training_mode(self):
        """Para modo de treinamento"""
        self.training_mode = False
        logger.info(f"Modo de treinamento parado - {len(self.training_data)} frames coletados")
    
    def train_models(self, external_data: Optional[np.ndarray] = None, 
                    epochs_cae: int = 10, epochs_convlstm: int = 8,  # REDUZIDO PARA TESTE
                    video_files: Optional[List[str]] = None,
                    max_frames_per_video: int = 50,     # NOVO PARÂMETRO
                    max_total_frames: int = 1000) -> Dict:  # NOVO PARÂMETRO
        """
        Treina ambos os modelos com suporte a múltiplos vídeos COM LIMITES
        
        Args:
            external_data: Dados externos de treinamento (opcional)
            epochs_cae: Épocas para CAE (padrão: 10 para teste)
            epochs_convlstm: Épocas para ConvLSTM (padrão: 8 para teste)
            video_files: Lista de arquivos de vídeo para treinamento
            max_frames_per_video: Máximo frames por vídeo (padrão: 50)
            max_total_frames: Máximo total de frames (padrão: 1000)
        """
        logger.info("Iniciando treinamento dos modelos")
        logger.info(f"CONFIGURAÇÃO: CAE {epochs_cae} épocas, ConvLSTM {epochs_convlstm} épocas")
        logger.info(f"LIMITES: {max_frames_per_video} frames/vídeo, {max_total_frames} total")
        
        # Coletar dados de múltiplas fontes
        all_training_data = []
        
        # Usar dados coletados online se disponível
        if self.training_data:
            frames_to_use = min(len(self.training_data), max_total_frames // 2)
            logger.info(f"Adicionando {frames_to_use} frames coletados online")
            all_training_data.extend(self.training_data[:frames_to_use])
        
        # Usar dados externos se fornecidos
        if external_data is not None:
            remaining_space = max_total_frames - len(all_training_data)
            frames_to_use = min(len(external_data), remaining_space)
            logger.info(f"Adicionando {frames_to_use} frames de dados externos")
            all_training_data.extend(external_data[:frames_to_use])
        
            # Processar vídeos COM LIMITE ABSOLUTO
        if video_files:
            logger.info(f"Processando {len(video_files)} vídeos")
            video_data = self._process_video_files(
                video_files, 
                max_frames_per_video=max_frames_per_video,
                max_total_frames=max_total_frames
            )
            if video_data:
                all_training_data = video_data[:max_total_frames]  # CORTAR PARA GARANTIR
        
        if not all_training_data:
            logger.error("Nenhum dado disponível para treinamento")
            return {"error": "Sem dados para treinamento"}
        
        # GARANTIR que não excede o limite
        # if len(all_training_data) > max_total_frames:
        #     logger.info(f"Limitando dataset de {len(all_training_data)} para {max_total_frames} frames")
        #     # Pegar frames distribuídos uniformemente
        #     indices = np.linspace(0, len(all_training_data)-1, max_total_frames, dtype=int)
        #     all_training_data = [all_training_data[i] for i in indices]
        
        # # Converter para numpy array
        # training_frames = np.array(all_training_data)
        # logger.info(f"Dataset final: {len(training_frames)} frames")
            # GARANTIR LIMITE FINAL
        if len(all_training_data) > max_total_frames:
            logger.info(f"🚨 CORTANDO dataset: {len(all_training_data)} -> {max_total_frames}")
            all_training_data = all_training_data[:max_total_frames]
        
        training_frames = np.array(all_training_data)
        logger.info(f"✅ Dataset final garantido: {len(training_frames)} frames")
        
        results = {}
        
        # Treinar CAE
        logger.info("Treinando CAE...")
        cae_data = np.array([
            VideoProcessor.preprocess_frame(frame, self.config.model.cae_input_shape[:2])
            for frame in training_frames
        ])
        # Remover frames inválidos
        cae_data = np.array([frame for frame in cae_data if frame is not None])
        
        if len(cae_data) > 0:
            # Dividir em treino/validação (80/20)
            split_idx = int(len(cae_data) * 0.8)
            train_data = cae_data[:split_idx]
            val_data = cae_data[split_idx:] if split_idx < len(cae_data) else None
            
            cae_result = self.cae.train(
                train_data,
                validation_data=val_data,
                epochs=epochs_cae,
                save_path=os.path.join(self.models_path, "cae_model")
            )
            results["cae_training"] = cae_result
        
        # Preparar dados para ConvLSTM (sequências)
        logger.info("Preparando dados para ConvLSTM...")
        sequences = create_sequences_fixed(
            cae_data, 
            self.config.model.convlstm_sequence_length,
            overlap=2
        )
        
        if len(sequences) > 0:
            # Dividir sequências em treino/validação
            split_idx = int(len(sequences) * 0.8)
            train_sequences = sequences[:split_idx]
            val_sequences = sequences[split_idx:] if split_idx < len(sequences) else None
            
            logger.info(f"Treinando ConvLSTM com {len(train_sequences)} sequências...")
            convlstm_result = self.convlstm.train(
                train_sequences,
                validation_sequences=val_sequences,
                epochs=epochs_convlstm,
                save_path=os.path.join(self.models_path, "convlstm_model")
            )
            results["convlstm_training"] = convlstm_result
        
        self.is_trained = True
        
        # Estatísticas finais
        results["training_summary"] = {
            "total_frames": len(training_frames),
            "cae_frames": len(cae_data),
            "convlstm_sequences": len(sequences) if len(sequences) > 0 else 0,
            "video_files_processed": len(video_files) if video_files else 0,
            "online_frames": len(self.training_data),
            "external_frames": len(external_data) if external_data is not None else 0,
            "max_frames_per_video": max_frames_per_video,
            "max_total_frames": max_total_frames
        }
        
        logger.info("Treinamento concluído")
        logger.info(f"Resumo: {results['training_summary']}")
        
        return results
    
    def _process_video_files(self, video_files: List[str], 
                           max_frames_per_video: int = 50, 
                           max_total_frames: int = 1000) -> List[np.ndarray]:
        """
        Processa múltiplos arquivos de vídeo para extração de frames COM LIMITE
        
        Args:
            video_files: Lista de caminhos para arquivos de vídeo
            max_frames_per_video: Máximo de frames por vídeo (padrão: 50)
            max_total_frames: Máximo total de frames (padrão: 1000)
            
        Returns:
            Lista de frames extraídos de todos os vídeos
        """
        all_frames = []
        
        logger.info(f"LIMITE: {max_frames_per_video} frames/vídeo, {max_total_frames} total")
        total_frames_collected = 0
        
        for i, video_path in enumerate(video_files):
            # PARAR se já coletou frames suficientes
            if total_frames_collected >= max_total_frames:
                logger.info(f"Limite total atingido ({max_total_frames}). Parando processamento.")
                break
                
            logger.info(f"Processando vídeo {i+1}/{len(video_files)}: {os.path.basename(video_path)}")
            
            if not os.path.exists(video_path):
                logger.warning(f"Arquivo não encontrado: {video_path}")
                continue
            
            try:
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    logger.warning(f"Não foi possível abrir: {video_path}")
                    continue
                
                # Obter informações do vídeo
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                logger.info(f"  📹 Frames: {total_frames}, FPS: {fps:.1f}")
                
                # Calcular quantos frames ainda pode coletar
                remaining_space = max_total_frames - total_frames_collected
                frames_to_extract = min(max_frames_per_video, remaining_space)
                
                if frames_to_extract <= 0:
                    logger.info("Limite total atingido, parando.")
                    break
                
                # Calcular skip para distribuir frames ao longo do vídeo
                if total_frames <= frames_to_extract:
                    frame_skip = 1
                else:
                    frame_skip = max(1, total_frames // frames_to_extract)
                
                frame_count = 0
                extracted_frames = 0
                
                logger.info(f"  🎯 Extraindo {frames_to_extract} frames (skip: {frame_skip})")
                
                with tqdm(total=frames_to_extract, desc=f"  Extraindo frames", leave=False) as pbar:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Aplicar frame skip
                        if frame_count % frame_skip == 0:
                            # Redimensionar frame para economizar memória
                            resized_frame = cv2.resize(frame, (224, 224))  # Usar tamanho fixo
                            all_frames.append(resized_frame)
                            extracted_frames += 1
                            pbar.update(1)
                        
                        frame_count += 1
                        
                        # Parar quando atingir limite por vídeo
                        if extracted_frames >= frames_to_extract:
                            break
                
                cap.release()
                total_frames_collected += extracted_frames
                logger.info(f"  ✅ Extraídos {extracted_frames} frames (Total: {total_frames_collected})")
                
                # Parar se atingiu limite total
                if total_frames_collected >= max_total_frames:
                    break
                
            except Exception as e:
                logger.error(f"Erro ao processar {video_path}: {e}")
                continue
        
        logger.info(f"Total de frames extraídos de todos os vídeos: {len(all_frames)}")
        return all_frames
    
    def train_from_video_directory(self, video_directory: str, 
                                 file_extensions: List[str] = None,
                                 epochs_cae: int = 10, epochs_convlstm: int = 8,  # REDUZIDO PARA TESTE
                                 max_frames_per_video: int = 50,    # NOVO PARÂMETRO
                                 max_total_frames: int = 1000) -> Dict:  # NOVO PARÂMETRO
        """
        Treina modelos usando todos os vídeos de um diretório COM LIMITES
        
        Args:
            video_directory: Diretório contendo vídeos de treinamento
            file_extensions: Extensões de arquivo aceitas
            epochs_cae: Épocas para CAE (padrão: 10 para teste)
            epochs_convlstm: Épocas para ConvLSTM (padrão: 8 para teste)
            max_frames_per_video: Máximo frames por vídeo (padrão: 50)
            max_total_frames: Máximo total de frames (padrão: 1000)
        """
        if file_extensions is None:
            file_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        
        logger.info(f"Buscando vídeos em: {video_directory}")
        logger.info(f"CONFIGURAÇÃO DE TESTE:")
        logger.info(f"  • Épocas CAE: {epochs_cae}")
        logger.info(f"  • Épocas ConvLSTM: {epochs_convlstm}")
        logger.info(f"  • Max frames/vídeo: {max_frames_per_video}")
        logger.info(f"  • Max total frames: {max_total_frames}")
        
        if not os.path.exists(video_directory):
            logger.error(f"Diretório não encontrado: {video_directory}")
            return {"error": "Diretório não encontrado"}
        
        # Encontrar todos os arquivos de vídeo
        video_files = []
        for root, dirs, files in os.walk(video_directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    video_files.append(os.path.join(root, file))
        
        if not video_files:
            logger.error(f"Nenhum arquivo de vídeo encontrado em: {video_directory}")
            return {"error": "Nenhum vídeo encontrado"}
        
        logger.info(f"Encontrados {len(video_files)} arquivos de vídeo")
        
        # Treinar com todos os vídeos COM LIMITE
        return self.train_models(
            video_files=video_files,
            epochs_cae=epochs_cae,
            epochs_convlstm=epochs_convlstm,
            max_frames_per_video=max_frames_per_video,    # NOVO
            max_total_frames=max_total_frames             # NOVO
        )
    
    def save_models(self, base_path: str = None):
        """Salva ambos os modelos"""
        if base_path is None:
            base_path = os.path.join(self.models_path, "anomaly_detector")
        
        self.cae.save_model(base_path)
        self.convlstm.save_model(base_path)
        
        logger.info(f"Modelos salvos: {base_path}")
    
    def load_models(self, base_path: str = None) -> bool:
        """Carrega ambos os modelos"""
        if base_path is None:
            base_path = os.path.join(self.models_path, "anomaly_detector")
        
        cae_loaded = self.cae.load_model(base_path)
        convlstm_loaded = self.convlstm.load_model(base_path)
        
        self.is_trained = cae_loaded and convlstm_loaded
        
        if self.is_trained:
            logger.info("Modelos carregados com sucesso")
        else:
            logger.warning("Falha ao carregar alguns modelos")
        
        return self.is_trained
    
    def get_model_info(self) -> Dict:
        """Retorna informações sobre os modelos"""
        return {
            "is_trained": self.is_trained,
            "training_mode": self.training_mode,
            "training_samples": len(self.training_data),
            "cae_threshold": self.cae.threshold,
            "convlstm_threshold": self.convlstm.threshold,
            "cae_input_shape": self.cae.input_shape,
            "convlstm_input_shape": self.convlstm.input_shape
        }