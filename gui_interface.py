#!/usr/bin/env python3
"""
Interface GUI Unificada - Idêntica à Interface Web
Sistema de Detecção de Anomalias
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import time
from datetime import datetime
import os

class AnomalyGUI:
    """Interface gráfica principal - Design idêntico à web"""
    
    def __init__(self, detection_system):
        self.system = detection_system
        self.root = tk.Tk()
        self.setup_window()
        self.create_interface()
        self.setup_callbacks()
        
        # Estados
        self.current_image = None
        self.is_recording = False
        
        # Dados para gráficos
        self.anomaly_history = []
        self.time_history = []
        self.fps_history = []
        self.max_history = 100
        
    def setup_window(self):
        """Configura janela principal"""
        self.root.title("Sistema de Detecção de Anomalias - v2.0")
        self.root.geometry("1600x1000")
        self.root.minsize(1400, 900)
        self.root.configure(bg='#0f0f0f')
        
        # Centralizar
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1600 // 2)
        y = (self.root.winfo_screenheight() // 2) - (1000 // 2)
        self.root.geometry(f"1600x1000+{x}+{y}")
        
    def create_interface(self):
        """Cria interface completa"""
        # Container principal
        main_container = tk.Frame(self.root, bg='#0f0f0f')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        self.create_header(main_container)
        
        # Status Bar
        self.create_status_bar(main_container)
        
        # Layout principal em grid
        content_frame = tk.Frame(main_container, bg='#0f0f0f')
        content_frame.pack(fill='both', expand=True, pady=(20, 0))
        
        # Configurar grid
        content_frame.grid_columnconfigure(0, weight=2)  # Vídeo
        content_frame.grid_columnconfigure(1, weight=1)  # Controles
        content_frame.grid_columnconfigure(2, weight=1)  # Stats
        content_frame.grid_rowconfigure(0, weight=2)     # Área principal
        content_frame.grid_rowconfigure(1, weight=1)     # Gráficos
        
        # Painéis principais
        self.create_video_panel(content_frame)
        self.create_control_panel(content_frame)
        self.create_stats_panel(content_frame)
        self.create_charts_panel(content_frame)
        
    def create_header(self, parent):
        """Header estilo web"""
        header = tk.Frame(parent, bg='#1a1a1a', height=80, relief='flat')
        header.pack(fill='x', pady=(0, 10))
        header.pack_propagate(False)
        
        # Container do header
        header_content = tk.Frame(header, bg='#1a1a1a')
        header_content.pack(fill='both', expand=True, padx=30, pady=15)
        
        # Logo/Título
        title_frame = tk.Frame(header_content, bg='#1a1a1a')
        title_frame.pack(side='left')
        
        title = tk.Label(title_frame, 
                        text="🎯 ANOMALY DETECTION SYSTEM",
                        bg='#1a1a1a', fg='#00ff88',
                        font=('JetBrains Mono', 18, 'bold'))
        title.pack(anchor='w')
        
        subtitle = tk.Label(title_frame,
                           text="Optical Flow + CAE + ConvLSTM | Real-time Monitoring",
                           bg='#1a1a1a', fg='#888888',
                           font=('JetBrains Mono', 10))
        subtitle.pack(anchor='w')
        
        # Status do sistema (direita)
        status_frame = tk.Frame(header_content, bg='#1a1a1a')
        status_frame.pack(side='right')
        
        self.system_status = tk.Label(status_frame,
                                     text="SYSTEM READY",
                                     bg='#1a1a1a', fg='#00ff88',
                                     font=('JetBrains Mono', 12, 'bold'))
        self.system_status.pack(anchor='e')
        
        self.time_display = tk.Label(status_frame,
                                    text=datetime.now().strftime("%H:%M:%S"),
                                    bg='#1a1a1a', fg='#cccccc',
                                    font=('JetBrains Mono', 10))
        self.time_display.pack(anchor='e')
        
    def create_status_bar(self, parent):
        """Barra de status estilo web"""
        status_bar = tk.Frame(parent, bg='#2a2a2a', height=40, relief='flat')
        status_bar.pack(fill='x', pady=(0, 10))
        status_bar.pack_propagate(False)
        
        status_content = tk.Frame(status_bar, bg='#2a2a2a')
        status_content.pack(fill='both', expand=True, padx=20, pady=8)
        
        # Indicadores de status
        self.create_status_indicator(status_content, "🟢 CAE Model", "LOADED", 'left')
        self.create_status_indicator(status_content, "🟢 ConvLSTM", "LOADED", 'left')
        self.create_status_indicator(status_content, "🟢 Optical Flow", "READY", 'left')
        
        # Status da detecção (direita)
        self.detection_status = tk.Label(status_content,
                                        text="⏸ STANDBY",
                                        bg='#2a2a2a', fg='#ffaa00',
                                        font=('JetBrains Mono', 10, 'bold'))
        self.detection_status.pack(side='right')
        
    def create_status_indicator(self, parent, label, status, side):
        """Cria indicador de status"""
        indicator = tk.Frame(parent, bg='#2a2a2a')
        indicator.pack(side=side, padx=(0, 20))
        
        tk.Label(indicator, text=f"{label}: {status}",
                bg='#2a2a2a', fg='#cccccc',
                font=('JetBrains Mono', 9)).pack()
        
    def create_video_panel(self, parent):
        """Painel de vídeo estilo web"""
        video_frame = tk.Frame(parent, bg='#1a1a1a', relief='flat', bd=2)
        video_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10), pady=(0, 10))
        
        # Header do painel
        panel_header = tk.Frame(video_frame, bg='#333333', height=40)
        panel_header.pack(fill='x')
        panel_header.pack_propagate(False)
        
        tk.Label(panel_header, text="📹 LIVE FEED",
                bg='#333333', fg='#ffffff',
                font=('JetBrains Mono', 12, 'bold')).pack(side='left', padx=15, pady=10)
        
        # Info do vídeo
        video_info = tk.Frame(panel_header, bg='#333333')
        video_info.pack(side='right', padx=15, pady=5)
        
        self.resolution_label = tk.Label(video_info, text="Resolution: --",
                                        bg='#333333', fg='#888888',
                                        font=('JetBrains Mono', 9))
        self.resolution_label.pack()
        
        self.fps_label = tk.Label(video_info, text="FPS: 0",
                                 bg='#333333', fg='#888888',
                                 font=('JetBrains Mono', 9))
        self.fps_label.pack()
        
        # Canvas do vídeo
        video_container = tk.Frame(video_frame, bg='#000000')
        video_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.video_canvas = tk.Canvas(video_container,
                                     bg='#000000',
                                     highlightthickness=0)
        self.video_canvas.pack(fill='both', expand=True)
        
        # Overlay de informações
        self.create_video_overlay()
        
    def create_video_overlay(self):
        """Overlay no vídeo"""
        overlay_text = "WAITING FOR INPUT..."
        self.video_canvas.create_text(
            400, 300, text=overlay_text,
            fill='#666666', font=('JetBrains Mono', 16, 'bold'),
            tags='overlay'
        )
        
    def create_control_panel(self, parent):
        """Painel de controles estilo dashboard"""
        control_frame = tk.Frame(parent, bg='#1a1a1a', relief='flat', bd=2)
        control_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 5), pady=(0, 10))
        
        # Header
        panel_header = tk.Frame(control_frame, bg='#333333', height=40)
        panel_header.pack(fill='x')
        panel_header.pack_propagate(False)
        
        tk.Label(panel_header, text="🎮 CONTROLS",
                bg='#333333', fg='#ffffff',
                font=('JetBrains Mono', 12, 'bold')).pack(padx=15, pady=10)
        
        # Container dos controles
        controls = tk.Frame(control_frame, bg='#1a1a1a')
        controls.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Botões principais
        self.create_control_button(controls, "📷 START CAMERA", self.start_camera, '#00aa44')
        self.create_control_button(controls, "📁 LOAD VIDEO", self.load_video, '#0088cc')
        self.create_control_button(controls, "⏹ STOP", self.stop_detection, '#cc4444', state='disabled')
        
        # Separador
        tk.Frame(controls, bg='#333333', height=2).pack(fill='x', pady=20)
        
        # Treinamento
        training_label = tk.Label(controls, text="TRAINING",
                                 bg='#1a1a1a', fg='#888888',
                                 font=('JetBrains Mono', 10, 'bold'))
        training_label.pack(pady=(0, 10))
        
        self.create_control_button(controls, "🎓 TRAIN (CAMERA)", self.train_camera, '#8844cc')
        self.create_control_button(controls, "📊 TRAIN (DATASET)", self.train_dataset, '#cc8844')
        
        # Separador
        tk.Frame(controls, bg='#333333', height=2).pack(fill='x', pady=20)
        
        # Configurações
        config_label = tk.Label(controls, text="CONFIGURATION",
                               bg='#1a1a1a', fg='#888888',
                               font=('JetBrains Mono', 10, 'bold'))
        config_label.pack(pady=(0, 10))
        
        # Threshold
        threshold_frame = tk.Frame(controls, bg='#1a1a1a')
        threshold_frame.pack(fill='x', pady=5)
        
        tk.Label(threshold_frame, text="Anomaly Threshold:",
                bg='#1a1a1a', fg='#cccccc',
                font=('JetBrains Mono', 9)).pack(anchor='w')
        
        self.threshold_var = tk.DoubleVar(value=0.1)
        self.threshold_scale = tk.Scale(threshold_frame,
                                       from_=0.01, to=0.5,
                                       resolution=0.01,
                                       orient='horizontal',
                                       variable=self.threshold_var,
                                       bg='#2a2a2a', fg='#cccccc',
                                       highlightthickness=0,
                                       length=200,
                                       troughcolor='#404040')
        self.threshold_scale.pack(fill='x', pady=5)
        
    def create_control_button(self, parent, text, command, color, state='normal'):
        """Cria botão de controle estilizado"""
        btn = tk.Button(parent, text=text, command=command,
                       bg=color, fg='white',
                       font=('JetBrains Mono', 10, 'bold'),
                       relief='flat', bd=0,
                       height=2, state=state,
                       activebackground=self.lighten_color(color),
                       cursor='hand2')
        btn.pack(fill='x', pady=5)
        
        # Armazenar referência para controle de estado
        button_name = text.split()[1].lower() if len(text.split()) > 1 else text.lower()
        setattr(self, f"{button_name}_btn", btn)
        
        return btn
    
    def lighten_color(self, color):
        """Clareia cor para hover effect"""
        color_map = {
            '#00aa44': '#00cc55',
            '#0088cc': '#00aaff',
            '#cc4444': '#ff5555',
            '#8844cc': '#aa55ff',
            '#cc8844': '#ffaa55'
        }
        return color_map.get(color, color)
        
    def create_stats_panel(self, parent):
        """Painel de estatísticas em tempo real"""
        stats_frame = tk.Frame(parent, bg='#1a1a1a', relief='flat', bd=2)
        stats_frame.grid(row=0, column=2, sticky='nsew', padx=(5, 0), pady=(0, 10))
        
        # Header
        panel_header = tk.Frame(stats_frame, bg='#333333', height=40)
        panel_header.pack(fill='x')
        panel_header.pack_propagate(False)
        
        tk.Label(panel_header, text="📊 REAL-TIME STATS",
                bg='#333333', fg='#ffffff',
                font=('JetBrains Mono', 12, 'bold')).pack(padx=15, pady=10)
        
        # Container das estatísticas
        stats_container = tk.Frame(stats_frame, bg='#1a1a1a')
        stats_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Grid de cards de estatísticas
        self.create_stat_card(stats_container, "FRAMES", "0", 0, 0, '#00aa44')
        self.create_stat_card(stats_container, "ANOMALIES", "0", 0, 1, '#ff4444')
        self.create_stat_card(stats_container, "RATE", "0%", 1, 0, '#ffaa00')
        self.create_stat_card(stats_container, "SCORE", "0.000", 1, 1, '#00aaff')
        self.create_stat_card(stats_container, "FPS", "0", 2, 0, '#aa44ff')
        self.create_stat_card(stats_container, "RUNTIME", "0s", 2, 1, '#44ffaa')
        
        # Separador
        tk.Frame(stats_container, bg='#333333', height=2).pack(fill='x', pady=20)
        
        # Alert log
        alert_label = tk.Label(stats_container, text="🚨 ALERT LOG",
                              bg='#1a1a1a', fg='#ffffff',
                              font=('JetBrains Mono', 11, 'bold'))
        alert_label.pack(pady=(0, 10))
        
        # Container do log com scrollbar
        log_container = tk.Frame(stats_container, bg='#1a1a1a')
        log_container.pack(fill='both', expand=True)
        
        self.alert_text = tk.Text(log_container,
                                 height=8, bg='#000000', fg='#ff4444',
                                 font=('JetBrains Mono', 8),
                                 relief='flat', bd=0,
                                 wrap=tk.WORD,
                                 insertbackground='#ff4444')
        
        scrollbar = tk.Scrollbar(log_container, command=self.alert_text.yview,
                                bg='#333333', troughcolor='#1a1a1a')
        self.alert_text.config(yscrollcommand=scrollbar.set)
        
        self.alert_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
    def create_stat_card(self, parent, title, value, row, col, color):
        """Cria card de estatística estilo web"""
        card = tk.Frame(parent, bg='#2a2a2a', relief='flat', bd=1)
        card.grid(row=row, column=col, sticky='ew', padx=5, pady=5)
        parent.grid_columnconfigure(col, weight=1)
        
        # Título
        title_label = tk.Label(card, text=title,
                              bg='#2a2a2a', fg='#888888',
                              font=('JetBrains Mono', 8, 'bold'))
        title_label.pack(pady=(8, 2))
        
        # Valor
        value_label = tk.Label(card, text=value,
                              bg='#2a2a2a', fg=color,
                              font=('JetBrains Mono', 14, 'bold'))
        value_label.pack(pady=(0, 8))
        
        # Armazenar referência
        setattr(self, f"stat_{title.lower()}", value_label)
        
    def create_charts_panel(self, parent):
        """Painel de gráficos"""
        charts_frame = tk.Frame(parent, bg='#1a1a1a', relief='flat', bd=2)
        charts_frame.grid(row=1, column=0, columnspan=3, sticky='nsew', pady=(10, 0))
        
        # Header
        panel_header = tk.Frame(charts_frame, bg='#333333', height=40)
        panel_header.pack(fill='x')
        panel_header.pack_propagate(False)
        
        tk.Label(panel_header, text="📈 MONITORING CHARTS",
                bg='#333333', fg='#ffffff',
                font=('JetBrains Mono', 12, 'bold')).pack(padx=15, pady=10)
        
        # Container dos gráficos
        charts_container = tk.Frame(charts_frame, bg='#1a1a1a')
        charts_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configurar matplotlib para tema escuro
        plt.style.use('dark_background')
        
        # Figura principal
        self.fig = Figure(figsize=(16, 4), dpi=80, facecolor='#1a1a1a')
        self.fig.patch.set_facecolor('#1a1a1a')
        
        # Três subplots
        self.ax1 = self.fig.add_subplot(131, facecolor='#000000')
        self.ax2 = self.fig.add_subplot(132, facecolor='#000000')
        self.ax3 = self.fig.add_subplot(133, facecolor='#000000')
        
        # Configurar eixos
        self.setup_chart_axes()
        
        # Canvas
        self.canvas = FigureCanvasTkAgg(self.fig, charts_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Inicializar linhas dos gráficos
        self.line1, = self.ax1.plot([], [], '#00ff88', linewidth=2, label='Anomaly Score')
        self.line2, = self.ax2.plot([], [], '#ff4444', linewidth=2, label='Detections')
        self.line3, = self.ax3.plot([], [], '#00aaff', linewidth=2, label='FPS')
        
    def setup_chart_axes(self):
        """Configura eixos dos gráficos"""
        charts_config = [
            (self.ax1, 'Anomaly Score Over Time', 'Score', '#00ff88'),
            (self.ax2, 'Anomaly Detections', 'Detections', '#ff4444'),
            (self.ax3, 'Performance (FPS)', 'FPS', '#00aaff')
        ]
        
        for ax, title, ylabel, color in charts_config:
            ax.set_title(title, color='white', fontsize=10, fontweight='bold')
            ax.set_ylabel(ylabel, color='white', fontsize=9)
            ax.set_xlabel('Time (s)', color='white', fontsize=9)
            ax.tick_params(colors='white', labelsize=8)
            ax.grid(True, alpha=0.3, color='#333333')
            
            # Estilo dos eixos
            for spine in ax.spines.values():
                spine.set_color('#333333')
        
    def setup_callbacks(self):
        """Configura callbacks do sistema"""
        self.system.set_callbacks(
            frame_cb=self.update_video_frame,
            anomaly_cb=self.handle_anomaly,
            stats_cb=self.update_stats
        )
        
        # Timer para atualizar interface
        self.update_interface()
        
    def update_interface(self):
        """Atualiza interface periodicamente"""
        # Atualizar hora
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_display.config(text=current_time)
        
        # Atualizar threshold do sistema
        if hasattr(self, 'threshold_var'):
            self.system.cae.threshold = self.threshold_var.get()
        
        # Agendar próxima atualização
        self.root.after(1000, self.update_interface)
        
    def update_video_frame(self, frame, flow_score=None):
        """Atualiza frame do vídeo"""
        try:
            # Limpar canvas
            self.video_canvas.delete('all')
            
            # Redimensionar frame para canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Manter aspect ratio
                h, w = frame.shape[:2]
                scale = min(canvas_width/w, canvas_height/h)
                new_w, new_h = int(w*scale), int(h*scale)
                
                frame_resized = cv2.resize(frame, (new_w, new_h))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Converter para PhotoImage
                image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image)
                
                # Centralizar no canvas
                x = canvas_width // 2
                y = canvas_height // 2
                self.video_canvas.create_image(x, y, image=photo)
                
                # Overlay de informações
                if flow_score is not None:
                    info_text = f"Optical Flow: {flow_score:.2f}"
                    self.video_canvas.create_text(10, 10, text=info_text,
                                                 fill='#00ff88', anchor='nw',
                                                 font=('JetBrains Mono', 10, 'bold'))
                
                # Manter referência
                self.current_image = photo
                
                # Atualizar info do vídeo
                self.resolution_label.config(text=f"Resolution: {w}x{h}")
                
        except Exception as e:
            print(f"Erro ao atualizar frame: {e}")
    
    def handle_anomaly(self, combined_score, mse_cae, mse_lstm):
        """Manipula detecção de anomalia"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Determinar severidade
        if combined_score > 0.3:
            severity = "HIGH"
            color = "red"
        elif combined_score > 0.15:
            severity = "MEDIUM"
            color = "orange"
        else:
            severity = "LOW"
            color = "yellow"
        
        # Adicionar ao log
        alert_msg = f"[{timestamp}] {severity} ANOMALY | Score: {combined_score:.4f} | CAE: {mse_cae:.4f} | LSTM: {mse_lstm:.4f}\n"
        
        self.alert_text.insert(tk.END, alert_msg)
        self.alert_text.see(tk.END)
        
        # Manter apenas últimas 100 linhas
        lines = self.alert_text.get("1.0", tk.END).split('\n')
        if len(lines) > 100:
            self.alert_text.delete("1.0", "2.0")
        
        # Atualizar status
        self.system_status.config(text=f"⚠️ {severity} ANOMALY", fg='#ff4444')
        
        # Reset status após 3 segundos
        self.root.after(3000, lambda: self.system_status.config(
            text="🔍 MONITORING", fg='#00ff88'))
    
    def update_stats(self, stats):
        """Atualiza estatísticas"""
        try:
            # Atualizar cards
            self.stat_frames.config(text=str(stats['frame_count']))
            self.stat_anomalies.config(text=str(stats['anomaly_count']))
            self.stat_rate.config(text=f"{stats['anomaly_rate']:.1f}%")
            self.stat_score.config(text=f"{stats['current_score']:.3f}")
            self.stat_fps.config(text=f"{stats['fps']:.1f}")
            
            # Formatar runtime
            runtime = int(stats['runtime'])
            if runtime >= 60:
                runtime_text = f"{runtime//60}m{runtime%60}s"
            else:
                runtime_text = f"{runtime}s"
            self.stat_runtime.config(text=runtime_text)
            
            # Atualizar gráficos
            current_time = time.time()
            self.time_history.append(stats['runtime'])
            self.anomaly_history.append(stats['current_score'])
            self.fps_history.append(stats['fps'])
            
            # Manter histórico limitado
            if len(self.time_history) > self.max_history:
                self.time_history.pop(0)
                self.anomaly_history.pop(0)
                self.fps_history.pop(0)
            
            # Atualizar linhas dos gráficos
            if len(self.time_history) > 1:
                self.update_chart_lines()
                
        except Exception as e:
            print(f"Erro ao atualizar estatísticas: {e}")
    
    def update_chart_lines(self):
        """Atualiza linhas dos gráficos"""
        try:
            # Gráfico 1: Anomaly Score
            self.line1.set_data(self.time_history, self.anomaly_history)
            self.ax1.relim()
            self.ax1.autoscale_view()
            
            # Gráfico 2: Detecções (binário)
            threshold = self.threshold_var.get()
            detections = [1 if score > threshold else 0 for score in self.anomaly_history]
            self.line2.set_data(self.time_history, detections)
            self.ax2.relim()
            self.ax2.autoscale_view()
            self.ax2.set_ylim(-0.1, 1.1)
            
            # Gráfico 3: FPS
            self.line3.set_data(self.time_history, self.fps_history)
            self.ax3.relim()
            self.ax3.autoscale_view()
            
            self.canvas.draw_idle()
            
        except Exception as e:
            print(f"Erro ao atualizar gráficos: {e}")
    
    def start_camera(self):
        """Inicia detecção por câmera"""
        try:
            if self.system.start_detection('camera'):
                self.camera_btn.config(state='disabled', bg='#666666')
                self.video_btn.config(state='disabled', bg='#666666')
                self.stop_btn.config(state='normal', bg='#cc4444')
                
                self.system_status.config(text="🔍 MONITORING CAMERA", fg='#00ff88')
                self.detection_status.config(text="🔴 RECORDING", fg='#ff4444')
                
                # Limpar log
                self.alert_text.delete("1.0", tk.END)
                
            else:
                messagebox.showerror("Error", "Cannot start camera")
                
        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {e}")
    
    def load_video(self):
        """Carrega vídeo"""
        try:
            video_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                    ("All files", "*.*")
                ]
            )
            
            if video_path:
                if self.system.start_detection('video', video_path):
                    self.camera_btn.config(state='disabled', bg='#666666')
                    self.video_btn.config(state='disabled', bg='#666666')
                    self.stop_btn.config(state='normal', bg='#cc4444')
                    
                    filename = os.path.basename(video_path)
                    self.system_status.config(text=f"🎬 PLAYING: {filename}", fg='#00ff88')
                    self.detection_status.config(text="▶️ PLAYING", fg='#00aaff')
                    
                    # Limpar log
                    self.alert_text.delete("1.0", tk.END)
                    
                else:
                    messagebox.showerror("Error", "Cannot load video")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Video error: {e}")
    
    def stop_detection(self):
        """Para detecção"""
        try:
            self.system.stop_detection()
            
            self.camera_btn.config(state='normal', bg='#00aa44')
            self.video_btn.config(state='normal', bg='#0088cc')
            self.stop_btn.config(state='disabled', bg='#666666')
            
            self.system_status.config(text="⏹ STOPPED", fg='#ffaa00')
            self.detection_status.config(text="⏸ STANDBY", fg='#ffaa00')
            
            # Limpar canvas
            self.video_canvas.delete('all')
            self.create_video_overlay()
            
        except Exception as e:
            messagebox.showerror("Error", f"Stop error: {e}")
    
    def train_camera(self):
        """Treina usando câmera"""
        try:
            result = messagebox.askyesno(
                "Confirm Training",
                "Start training using camera?\n\nThis process may take a few minutes."
            )
            
            if result:
                self.start_training("camera")
                
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {e}")
    
    def train_dataset(self):
        """Treina usando dataset"""
        try:
            dataset_path = filedialog.askdirectory(title="Select Dataset Folder")
            
            if dataset_path:
                result = messagebox.askyesno(
                    "Confirm Training",
                    f"Start training using dataset:\n{dataset_path}\n\nThis process may take a few minutes."
                )
                
                if result:
                    self.start_training("dataset", dataset_path)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {e}")
    
    def start_training(self, mode, path=None):
        """Inicia treinamento"""
        # Desabilitar botões
        self.camera_btn.config(state='disabled', bg='#666666')
        self.train_btn.config(state='disabled', bg='#666666')
        self.dataset_btn.config(state='disabled', bg='#666666')
        
        self.system_status.config(text="🎓 TRAINING...", fg='#8844cc')
        
        def train_thread():
            use_camera = mode == "camera"
            self.system.train_models(use_camera=use_camera, dataset_path=path)
            
            # Reabilitar botões na thread principal
            self.root.after(0, self.training_completed)
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def training_completed(self):
        """Callback de treinamento concluído"""
        self.camera_btn.config(state='normal', bg='#00aa44')
        self.train_btn.config(state='normal', bg='#8844cc')
        self.dataset_btn.config(state='normal', bg='#cc8844')
        
        self.system_status.config(text="✅ TRAINING COMPLETED", fg='#00ff88')
        messagebox.showinfo("Success", "Training completed successfully!")
    
    def on_closing(self):
        """Manipula fechamento"""
        try:
            if self.system.is_running:
                self.system.stop_detection()
            self.root.destroy()
            
        except Exception as e:
            print(f"Error closing: {e}")
            self.root.destroy()
    
    def run(self):
        """Executa aplicação"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    # Teste standalone
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    
    from main import AnomalyDetectionSystem
    system = AnomalyDetectionSystem()
    app = AnomalyGUI(system)
    app.run()