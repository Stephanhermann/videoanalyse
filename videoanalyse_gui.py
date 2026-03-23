#!/Users/stephanhermann/videoanalyse_venv/bin/python3
"""Streamlit web GUI for videoanalyse_emb.py"""

import streamlit as st
import subprocess
import os
import json
import psutil
import GPUtil
import time
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.absolute()
SCRIPT_PATH = SCRIPT_DIR / "videoanalyse_emb.py"

# Page config
st.set_page_config(
    page_title="Videoanalyse EMB",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🎬 Videoanalyse EMB")
st.markdown("Analyse von Videos mit OpenCV, Whisper & Ollama Vision")

# Metrics overlay
if "metrics_data" not in st.session_state:
    st.session_state.metrics_data = {
        "time": [],
        "cpu": [],
        "ram": [],
        "gpu": []
    }

cpu_percent = psutil.cpu_percent(interval=None)
ram_percent = psutil.virtual_memory().percent

try:
    gpus = GPUtil.getGPUs()
    gpu_percent = gpus[0].load * 100 if gpus else 0.0
except Exception:
    gpu_percent = 0.0

st.session_state.metrics_data["time"].append(datetime.now().strftime("%H:%M:%S"))
st.session_state.metrics_data["cpu"].append(cpu_percent)
st.session_state.metrics_data["ram"].append(ram_percent)
st.session_state.metrics_data["gpu"].append(gpu_percent)

if len(st.session_state.metrics_data["time"]) > 60:
    for key in ["time", "cpu", "ram", "gpu"]:
        st.session_state.metrics_data[key].pop(0)

metrics_df = {
    "CPU": st.session_state.metrics_data["cpu"],
    "RAM": st.session_state.metrics_data["ram"],
    "GPU": st.session_state.metrics_data["gpu"]
}

st.markdown("### 📈 Systemauslastung")
st.write(
    f"CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}% | GPU: {gpu_percent:.1f}%"
)
st.line_chart(metrics_df)

# Initialize session state
if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False
if "log_output" not in st.session_state:
    st.session_state.log_output = ""

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Konfiguration")
    
    mode = st.radio("Modus", ["📹 Einzelvideo", "📁 Ordner / Batch"])
    
    if mode == "📹 Einzelvideo":
        input_path = st.text_input("Videodatei-Pfad", placeholder="/path/to/video.mp4")
    else:
        input_path = st.text_input("Ordner-Pfad", placeholder="/path/to/videos")
    
    output_path = st.text_input("Ausgabeordner (optional)", placeholder="/path/to/output")
    
    st.subheader("Modelle")
    vision_model = st.text_input("Vision Modell", value="minicpm-v")
    whisper_model = st.selectbox("Whisper Modell", ["tiny", "base", "small", "medium", "large"])
    
    st.subheader("Parameter")
    frame_interval = st.slider("Frame-Intervall (Sekunden)", 1, 60, 10)
    max_vision_frames = st.slider("Max Vision Frames", 1, 100, 20)
    ollama_url = st.text_input("Ollama URL", value="http://127.0.0.1:11434")
    
    st.subheader("Optionen")
    use_vision = st.checkbox("Vision-Analyse aktivieren", value=True)
    write_nfo = st.checkbox("NFO schreiben", value=True)
    update_emby = st.checkbox("Emby Bibliothek aktualisieren", value=False)
    
    if update_emby:
        emby_url = st.text_input("Emby URL", placeholder="http://emby-server:8096")
        emby_api_key = st.text_input("Emby API Key", type="password")
    else:
        emby_url = ""
        emby_api_key = ""

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Analyse")
    
    if st.button("▶ Start Analyse", key="start_btn", use_container_width=True):
        if not input_path:
            st.error("❌ Bitte geben Sie einen Videodatei-Pfad oder Ordner an.")
        elif not os.path.exists(input_path):
            st.error(f"❌ Pfad nicht gefunden: {input_path}")
        else:
            st.session_state.analysis_running = True
            st.session_state.log_output = ""
            
            # Build command
            cmd = [
                str(SCRIPT_PATH),
                "--vision-model", vision_model,
                "--whisper-model", whisper_model,
                "--frame-interval", str(frame_interval),
                "--max-vision-frames", str(max_vision_frames),
                "--ollama-url", ollama_url,
            ]
            
            if mode == "📹 Einzelvideo":
                cmd.extend(["--input", input_path])
            else:
                cmd.extend(["--input-folder", input_path])
            
            if output_path:
                cmd.extend(["--output", output_path])
            
            if use_vision:
                cmd.append("--use-vision")
            
            if write_nfo:
                cmd.append("--write-nfo")
            
            if update_emby and emby_url and emby_api_key:
                cmd.append("--update-emby")
                cmd.extend(["--emby-url", emby_url])
                cmd.extend(["--emby-api-key", emby_api_key])
            
            # Run analysis
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_area = st.empty()
            
            status_text.info("⏳ Analyse läuft...")
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                last_percent = 0
                
                for line in process.stdout:
                    st.session_state.log_output += line
                    
                    # Parse progress JSON if available
                    if line.strip().startswith('{"type": "progress"'):
                        try:
                            data = json.loads(line.strip())
                            percent = data.get("percent", 0)
                            message = data.get("message", "")
                            
                            if percent > last_percent:
                                progress_bar.progress(min(100, percent / 100.0))
                                last_percent = percent
                            
                            status_text.info(f"⏳ {message}")
                        except json.JSONDecodeError:
                            pass
                    elif "abgeschlossen" in line.lower():
                        progress_bar.progress(1.0)
                        status_text.success("✅ Analyse abgeschlossen!")
                    elif "fehler" in line.lower():
                        status_text.warning(f"⚠️ {line.strip()}")
                
                process.wait()
                
                if process.returncode == 0:
                    st.session_state.analysis_running = False
                    status_text.success("✅ Erfolgreich abgeschlossen!")
                    progress_bar.progress(1.0)
                else:
                    st.session_state.analysis_running = False
                    status_text.error(f"❌ Fehler: Prozess endete mit Code {process.returncode}")
                
            except Exception as e:
                st.session_state.analysis_running = False
                status_text.error(f"❌ Fehler: {e}")
    
    # Log output
    st.subheader("📋 Log")
    log_display = st.empty()
    
    if st.session_state.log_output:
        log_display.text_area(
            "Analyse-Log",
            value=st.session_state.log_output,
            height=300,
            disabled=True,
            key="log_area"
        )
    else:
        log_display.info("Kein Log verfügbar. Starten Sie eine Analyse.")

with col2:
    st.subheader("ℹ️ Info")
    
    st.info(f"""
    **Version:** 1.0.0
    
    **Script:** videoanalyse_emb.py
    
    **Pfad:** {SCRIPT_PATH}
    """)
    
    st.subheader("🎨 Features")
    st.markdown("""
    - 🎥 Video-Analyse mit OpenCV
    - 🎤 Audio-Transkription (Whisper)
    - 👁️ Frame-Vision mit Ollama
    - 📝 NFO-Datei-Export
    - 🔄 Batch-Verarbeitung
    - 📊 Live-Progress
    """)
    
    st.subheader("🔗 Schnelllinks")
    if st.button("🔄 Seite aktualisieren"):
        st.rerun()

# Footer
st.divider()
st.caption(f"Videoanalyse EMB WebUI | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
