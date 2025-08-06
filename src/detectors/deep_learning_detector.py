#!/usr/bin/env python3
"""
CORREÇÃO DO DEEP LEARNING DETECTOR
Adicionando método detect_sequence que estava faltando
"""

import numpy as np
import time
import pickle
from typing import Dict, List, Tuple, Optional
from collections import deque
from queue import Queue
import threading
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.logger import logger
from utils.helpers import VideoProcessor, time_function, normalize_batch_fixed


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
        self.batch_size = 32
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
    
    # *** MÉTODO QUE ESTAVA FALTANDO! ***
    def detect_sequence(self, frames: List[np.ndarray]) -> Dict:
        """
        Detecta anomalias em uma sequência de frames
        ESTE É O MÉTODO QUE ESTAVA FALTANDO!
        
        Args:
            frames: Lista de frames para analisar
            
        Returns:
            Dict com resultado da análise da sequência
        """
        if not frames or len(frames) == 0:
            return {"error": "Sequência vazia"}
        
        logger.info(f"🔍 Analisando sequência de {len(frames)} frames")
        
        results = {
            "timestamp": time.time(),
            "sequence_length": len(frames),
            "frame_results": [],
            "sequence_analysis": {},
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
        
        # Análise da sequência como um todo
        anomaly_ratio = anomaly_count / len(frames) if len(frames) > 0 else 0
        avg_confidence = total_confidence / anomaly_count if anomaly_count > 0 else 0
        
        results["sequence_analysis"] = {
            "total_frames": len(frames),
            "anomaly_frames": anomaly_count,
            "anomaly_ratio": anomaly_ratio,
            "avg_confidence": avg_confidence
        }
        
        # Decisão final da sequência
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
        temporal_analysis = convlstm_result.get("temporal_analysis", {})
        
        # Classificação baseada em padrões
        if temporal_analysis.get("trend") == "increasing_change":
            return "movement_escalation"
        elif temporal_analysis.get("trend") == "decreasing_change":
            return "sudden_stop"
        elif cae_confidence > convlstm_confidence * 1.5:
            return "spatial_anomaly"
        elif convlstm_confidence > cae_confidence * 1.5:
            return "temporal_anomaly"
        else:
            return "spatiotemporal_anomaly"
    
    def start_training_mode(self):
        """Inicia modo de treinamento"""
        self.training_mode = True
        self.training_data = []
        logger.info("Modo de treinamento ativado")
    
    def stop_training_mode(self):
        """Para modo de treinamento e treina modelos"""
        self.training_mode = False
        logger.info(f"Modo de treinamento desativado - {len(self.training_data)} amostras coletadas")
        
        if len(self.training_data) > 100:  # Mínimo de amostras
            self._train_models()
        else:
            logger.warning("Poucos dados para treinamento - colete mais amostras")
    
    def _train_models(self):
        """Treina ambos os modelos com os dados coletados"""
        logger.info("Iniciando treinamento dos modelos...")
        
        # Converter dados para arrays numpy
        frames = np.array(self.training_data)
        
        # Treinar CAE
        logger.info("Treinando CAE...")
        cae_result = self.cae.train(frames, epochs=20)
        
        # Criar sequências para ConvLSTM
        sequences = []
        seq_length = self.convlstm.input_shape[0]
        
        for i in range(len(frames) - seq_length + 1):
            sequences.append(frames[i:i + seq_length])
        
        if len(sequences) > 10:
            sequences = np.array(sequences)
            logger.info("Treinando ConvLSTM...")
            convlstm_result = self.convlstm.train(sequences, epochs=15)
        
        # Salvar modelos
        self.save_models()
        self.is_trained = True
        
        logger.info("Treinamento concluído e modelos salvos")
    
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