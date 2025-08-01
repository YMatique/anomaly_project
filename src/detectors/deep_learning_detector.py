"""
Detector de Deep Learning usando CAE (Convolutional Autoencoder) e ConvLSTM
Segunda camada de detecção para análise mais profunda de anomalias
Otimizado para i5 11Gen com 16GB RAM
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

from ..utils.helpers import time_function, ModelUtils, VideoProcessor
from ..utils.logger import logger

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
        
        # Normalizar dados
        normalized_data, self.normalization_params = ModelUtils.normalize_batch(training_data)
        
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
            validation_data_norm = ModelUtils.normalize_batch(validation_data, self.normalization_params)[0]
        
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
        
        # Normalizar
        normalized_sequences, self.normalization_params = ModelUtils.normalize_batch(sequences)
        
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
            val_normalized = ModelUtils.normalize_batch(validation_sequences, self.normalization_params)[0]
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
                    epochs_cae: int = 50, epochs_convlstm: int = 30) -> Dict:
        """
        Treina ambos os modelos
        
        Args:
            external_data: Dados externos de treinamento (opcional)
            epochs_cae: Épocas para CAE
            epochs_convlstm: Épocas para ConvLSTM
        """
        logger.info("Iniciando treinamento dos modelos")
        
        # Usar dados coletados ou externos
        if external_data is not None:
            training_frames = external_data
        elif self.training_data:
            training_frames = np.array(self.training_data)
        else:
            logger.error("Nenhum dado disponível para treinamento")
            return {"error": "Sem dados para treinamento"}
        
        results = {}
        
        # Treinar CAE
        logger.info("Treinando CAE...")
        cae_data = np.array([
            VideoProcessor.preprocess_frame(frame, self.config.model.cae_input_shape[:2])
            for frame in training_frames
        ])
        cae_data = cae_data[cae_data != None]  # Remover frames inválidos
        
        if len(cae_data) > 0:
            cae_result = self.cae.train(
                cae_data, 
                epochs=epochs_cae,
                save_path=os.path.join(self.models_path, "cae_model")
            )
            results["cae_training"] = cae_result
        
        # Preparar dados para ConvLSTM (sequências)
        logger.info("Preparando dados para ConvLSTM...")
        sequences = ModelUtils.create_sequences(
            cae_data, 
            self.config.model.convlstm_sequence_length,
            overlap=2
        )
        
        if len(sequences) > 0:
            logger.info("Treinando ConvLSTM...")
            convlstm_result = self.convlstm.train(
                sequences,
                epochs=epochs_convlstm,
                save_path=os.path.join(self.models_path, "convlstm_model")
            )
            results["convlstm_training"] = convlstm_result
        
        self.is_trained = True
        logger.info("Treinamento concluído")
        
        return results
    
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