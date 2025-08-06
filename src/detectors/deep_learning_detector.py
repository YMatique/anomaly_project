#!/usr/bin/env python3
"""
SUBSTITUA SEU ARQUIVO src/detectors/deep_learning_detector.py POR ESTE
Detector de Deep Learning com métodos de treinamento implementados
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

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.helpers import time_function, VideoProcessor, normalize_batch_fixed
from utils.logger import logger

class ConvolutionalAutoencoder:
    """
    Convolutional Autoencoder para detecção de anomalias espaciais
    Treina apenas com dados normais e detecta anomalias via erro de reconstrução
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (64, 64, 3)):
        self.input_shape = input_shape
        self.model = None
        self.encoder = None
        self.threshold = 0.0
        self.normalization_params = None
        
        # Configurações
        self.batch_size = 16
        self.learning_rate = 0.001
        
        self._build_model()
        logger.info(f"CAE inicializado - input shape: {input_shape}")
    
    def _build_model(self):
        """Constrói arquitetura encoder-decoder"""
        
        # Encoder
        input_layer = layers.Input(shape=self.input_shape)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Decoder
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        decoded = layers.Conv2D(self.input_shape[2], (3, 3), activation='sigmoid', padding='same')(x)
        
        # Compilar modelo
        self.model = keras.Model(input_layer, decoded)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Criar encoder separado
        self.encoder = keras.Model(input_layer, encoded)
        
        logger.info("Arquitetura CAE construída")
    
    def train(self, training_data: np.ndarray, validation_split: float = 0.2,
              epochs: int = 50, batch_size: int = None, save_path: str = None) -> Dict:
        """
        Treina o autoencoder com dados normais
        
        Args:
            training_data: Array de frames normais [N, H, W, C]
            validation_split: Proporção para validação
            epochs: Número de épocas
            batch_size: Tamanho do batch (usa padrão se None)
            save_path: Caminho para salvar modelo
            
        Returns:
            Dict com histórico de treinamento
        """
        logger.info(f"Iniciando treinamento CAE - {len(training_data)} samples, {epochs} epochs")
        
        if batch_size is not None:
            self.batch_size = batch_size
        
        # Normalizar dados
        normalized_data, self.normalization_params = normalize_batch_fixed(training_data)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        if save_path:
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    f"{save_path}_cae.h5", save_best_only=True, monitor='val_loss'
                )
            )
        
        # Treinamento
        history = self.model.fit(
            normalized_data, normalized_data,  # Autoencoder: input = output
            epochs=epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calcular threshold baseado nos erros de reconstrução
        self._calculate_threshold(normalized_data)
        
        # Salvar modelo e parâmetros
        if save_path:
            self.save_model(save_path)
        
        logger.info(f"Treinamento CAE concluído - threshold: {self.threshold:.6f}")
        
        return {
            "history": history.history,
            "threshold": self.threshold,
            "final_loss": history.history['loss'][-1],
            "best_val_loss": min(history.history['val_loss']) if 'val_loss' in history.history else None
        }
    
    def _calculate_threshold(self, normalized_data: np.ndarray):
        """Calcula threshold baseado nos erros de reconstrução do training set"""
        predictions = self.model.predict(normalized_data, batch_size=self.batch_size, verbose=0)
        reconstruction_errors = np.mean((normalized_data - predictions) ** 2, axis=(1, 2, 3))
        
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        self.threshold = mean_error + 2 * std_error
        
        logger.info(f"Threshold calculado: {self.threshold:.6f} (mean: {mean_error:.6f}, std: {std_error:.6f})")
    
    def predict(self, frame: np.ndarray) -> Dict:
        """
        Prediz se frame é anômalo baseado no erro de reconstrução
        """
        if self.model is None:
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
        
        # Calcular erro de reconstrução (MSE)
        reconstruction_error = np.mean((processed_frame - reconstruction[0]) ** 2)
        
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
            encoder_layer = None
            for i, layer in enumerate(self.model.layers):
                if 'max_pooling2d' in layer.name and i > 4:  # Último pooling do encoder
                    encoder_layer = layer.output
                    break
            
            if encoder_layer is not None:
                self.encoder = keras.Model(self.model.input, encoder_layer)
            
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
    
    def train(self, sequences: np.ndarray, validation_split: float = 0.2,
              epochs: int = 40, batch_size: int = None, save_path: str = None) -> Dict:
        """
        Treina ConvLSTM com sequências normais
        
        Args:
            sequences: Array de sequências [N, T, H, W, C]
            validation_split: Proporção para validação
            epochs: Número de épocas
            batch_size: Tamanho do batch
            save_path: Caminho para salvar
            
        Returns:
            Dict com histórico de treinamento
        """
        logger.info(f"Iniciando treinamento ConvLSTM - {len(sequences)} sequências, {epochs} epochs")
        
        if batch_size is not None:
            self.batch_size = batch_size
        
        # Normalizar sequências
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
                keras.callbacks.ModelCheckpoint(
                    f"{save_path}_convlstm.h5", save_best_only=True, monitor='val_loss'
                )
            )
        
        # Treinamento
        history = self.model.fit(
            normalized_sequences, targets,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calcular threshold
        self._calculate_threshold(normalized_sequences, targets)
        
        # Salvar modelo
        if save_path:
            self.save_model(save_path)
        
        logger.info(f"Treinamento ConvLSTM concluído - threshold: {self.threshold:.6f}")
        
        return {
            "history": history.history,
            "threshold": self.threshold,
            "final_loss": history.history['loss'][-1],
            "best_val_loss": min(history.history['val_loss']) if 'val_loss' in history.history else None
        }
    
    def _calculate_threshold(self, sequences: np.ndarray, targets: np.ndarray):
        """Calcula threshold baseado nos erros de reconstrução"""
        predictions = self.model.predict(sequences, batch_size=self.batch_size, verbose=0)
        reconstruction_errors = np.mean((targets - predictions) ** 2, axis=(1, 2, 3))
        
        mean_error = np.mean(reconstruction_errors)
        std_error = np.std(reconstruction_errors)
        self.threshold = mean_error + 2 * std_error
        
        logger.info(f"ConvLSTM threshold calculado: {self.threshold:.6f}")
    
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
        
        return {
            "reconstruction_error": float(reconstruction_error),
            "threshold": float(self.threshold),
            "is_anomaly": bool(is_anomaly),
            "confidence": float(confidence),
            "reconstructed_frame": reconstruction[0]
        }
    
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
            logger.error(f"Erro ao carregar modelo ConvLSTM: {e}")
            return False


class DeepLearningDetector:
    """
    Detector principal que combina CAE e ConvLSTM
    Implementa arquitetura em cascata otimizada
    """
    
    def __init__(self, config):
        self.config = config
        
        # Inicializar detectores
        cae_shape = tuple(config.model.cae_input_shape)
        convlstm_shape = tuple([config.model.convlstm_sequence_length] + list(cae_shape))
        
        self.cae = ConvolutionalAutoencoder(cae_shape)
        self.convlstm = ConvLSTMDetector(convlstm_shape)
        
        # Configurações
        self.anomaly_threshold = config.model.anomaly_threshold
        self.models_path = config.system.models_path
        
        # Estado
        self.is_trained = False
        self.training_mode = False
        self.training_data = []
        
        logger.info("DeepLearningDetector inicializado")
    
    @time_function
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detecta anomalias usando CAE e ConvLSTM em cascata
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
                    
                    # Decisão final baseada em ambos os modelos
                    final_confidence = (cae_result.get("confidence", 0) + 
                                      convlstm_result.get("confidence", 0)) / 2
                    
                    results["final_decision"] = {
                        "is_anomaly": True,
                        "confidence": final_confidence,
                        "anomaly_type": self._classify_anomaly(cae_result, convlstm_result)
                    }
                else:
                    # Apenas CAE detectou
                    results["final_decision"] = {
                        "is_anomaly": True,
                        "confidence": cae_result.get("confidence", 0),
                        "anomaly_type": "spatial_anomaly"
                    }
        
        return results
    
    def detect_sequence(self, frames: List[np.ndarray]) -> Dict:
        """
        Detecta anomalias em uma sequência de frames
        """
        if not frames or len(frames) == 0:
            return {"error": "Sequência vazia"}
        
        logger.info(f"🔍 Analisando sequência de {len(frames)} frames")
        
        results = {
            "timestamp": time.time(),
            "sequence_length": len(frames),
            "frame_results": [],
            "final_decision": {
                "is_anomaly": False,
                "confidence": 0.0,
                "anomaly_type": "none",
                "anomaly_frames": []
            }
        }
        
        # Analisar cada frame individualmente
        anomaly_count = 0
        total_confidence = 0.0
        anomaly_frames = []
        
        for i, frame in enumerate(frames):
            frame_result = self.detect(frame)
            results["frame_results"].append({
                "frame_index": i,
                "result": frame_result
            })
            
            # Contar anomalias
            if frame_result.get("final_decision", {}).get("is_anomaly", False):
                anomaly_count += 1
                total_confidence += frame_result.get("final_decision", {}).get("confidence", 0)
                anomaly_frames.append(i)
        
        # Decisão final da sequência
        anomaly_ratio = anomaly_count / len(frames) if len(frames) > 0 else 0
        avg_confidence = total_confidence / anomaly_count if anomaly_count > 0 else 0
        
        # Considera sequência anômala se mais de 30% dos frames são anômalos
        sequence_is_anomaly = anomaly_ratio > 0.3
        
        results["final_decision"] = {
            "is_anomaly": sequence_is_anomaly,
            "confidence": avg_confidence,
            "anomaly_type": "sequence_anomaly" if sequence_is_anomaly else "normal",
            "anomaly_frames": anomaly_frames
        }
        
        logger.info(f"✅ Sequência analisada: {anomaly_count}/{len(frames)} frames anômalos")
        
        return results
    
    def _classify_anomaly(self, cae_result: Dict, convlstm_result: Dict) -> str:
        """Classifica tipo de anomalia baseado nos resultados"""
        
        cae_confidence = cae_result.get("confidence", 0)
        convlstm_confidence = convlstm_result.get("confidence", 0)
        
        if cae_confidence > convlstm_confidence * 1.5:
            return "spatial_anomaly"
        elif convlstm_confidence > cae_confidence * 1.5:
            return "temporal_anomaly"
        else:
            return "spatiotemporal_anomaly"
    
    def save_models(self):
        """Salva ambos os modelos"""
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        
        # Salvar CAE
        cae_path = os.path.join(self.models_path, "cae_model")
        self.cae.save_model(cae_path)
        
        # Salvar ConvLSTM
        convlstm_path = os.path.join(self.models_path, "convlstm_model")
        self.convlstm.save_model(convlstm_path)
        
        logger.info("Modelos salvos com sucesso")
    
    def load_models(self) -> bool:
        """Carrega ambos os modelos"""
        success = True
        
        # Carregar CAE
        cae_path = os.path.join(self.models_path, "cae_model")
        if not self.cae.load_model(cae_path):
            success = False
        
        # Carregar ConvLSTM
        convlstm_path = os.path.join(self.models_path, "convlstm_model")
        if not self.convlstm.load_model(convlstm_path):
            success = False
        
        if success:
            self.is_trained = True
            logger.info("Ambos os modelos carregados com sucesso")
        else:
            logger.warning("Falha ao carregar alguns modelos")
        
        return success