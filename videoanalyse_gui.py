#!/Users/stephanhermann/videoanalyse_venv/bin/python3
"""Streamlit web GUI for videoanalyse_emb.py"""

import streamlit as st
import subprocess
import os
import json
import re
import psutil
import GPUtil
import time
import requests
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET

SCRIPT_DIR = Path(__file__).parent.absolute()
SCRIPT_PATH = SCRIPT_DIR / "videoanalyse_emb.py"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".wmv", ".flv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".heic", ".heif"}


def _scan_video_files(folder: str, recursive: bool = True):
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []

    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]

    return sorted(files, key=lambda p: str(p).lower())


def _scan_photo_files(folder: str, recursive: bool = True):
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: str(p).lower())


def _guess_title_year(file_path: Path):
    name = file_path.stem
    year = 0
    title = name

    match = re.search(r"\b(19|20)\d{2}\b", name)
    if match:
        year = int(match.group())
        title = name[:match.start()]

    title = re.sub(r"[._-]+", " ", title).strip()
    if not title:
        title = name

    return title, year


def _find_emb_nfo(video_path: Path):
    candidates = [
        video_path.with_suffix(".nfo"),
        video_path.parent / "output" / video_path.stem / f"{video_path.stem}.nfo",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _read_emb_nfo(video_path: Path):
    nfo_path = _find_emb_nfo(video_path)
    if not nfo_path:
        return {}

    try:
        root = ET.parse(nfo_path).getroot()
    except Exception:
        return {}

    title = (root.findtext("title") or "").strip()
    plot = (root.findtext("plot") or "").strip()
    premiered = (root.findtext("premiered") or "").strip()
    runtime_text = (root.findtext("runtime") or "").strip()

    year = 0
    if len(premiered) >= 4 and premiered[:4].isdigit():
        year = int(premiered[:4])

    runtime_minutes = 0
    if runtime_text.isdigit():
        runtime_minutes = int(runtime_text)

    return {
        "title": title,
        "plot": plot,
        "year": year,
        "premiered": premiered,
        "runtime_minutes": runtime_minutes,
        "nfo_path": str(nfo_path),
    }


def _post_movie(moviedb_url: str, api_key: str, payload: dict):
    url = moviedb_url.rstrip("/") + "/movie"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return requests.post(url, headers=headers, json=payload, timeout=20)


def _post_photo(moviedb_url: str, api_key: str, payload: dict):
    url = moviedb_url.rstrip("/") + "/photos"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return requests.post(url, headers=headers, json=payload, timeout=20)


def _get_current_page() -> str:
    page = st.query_params.get("page", "analyse")
    if page not in {"analyse", "import"}:
        page = "analyse"
    return page


def _expected_summary_paths(mode: str, input_path: str, output_path: str):
    input_p = Path(input_path)
    if mode == "📹 Einzelvideo":
        if output_path:
            out_dir = Path(output_path)
        else:
            out_dir = input_p.parent / "output" / input_p.stem
        return [out_dir / "minicpm_v_analysis.json"]

    if output_path:
        batch_root = Path(output_path)
    else:
        batch_root = input_p / "output"

    summary_paths = []
    for video in _scan_video_files(str(input_p), recursive=True):
        summary_paths.append(batch_root / video.stem / "minicpm_v_analysis.json")

    # remove duplicates (same stem in different folders can still collide)
    unique = []
    seen = set()
    for p in summary_paths:
        s = str(p)
        if s not in seen:
            seen.add(s)
            unique.append(p)
    return unique


def _build_analysis_summary(mode: str, input_path: str, output_path: str):
    rows = []
    for summary_path in _expected_summary_paths(mode, input_path, output_path):
        if not summary_path.exists():
            continue
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        video_info = data.get("video_info", {}) or {}
        moviepy_info = data.get("moviepy_info", {}) or {}
        whisper = data.get("whisper", {}) or {}
        frame_info = data.get("frame_info", {}) or {}

        input_video = data.get("input_video", "")
        duration_seconds = float(moviepy_info.get("duration") or video_info.get("duration_seconds") or 0.0)

        rows.append(
            {
                "video": Path(input_video).name if input_video else summary_path.parent.name,
                "status": data.get("status", ""),
                "dauer_min": round(duration_seconds / 60.0, 1) if duration_seconds else 0.0,
                "sprache": whisper.get("language", ""),
                "frames": int(frame_info.get("saved_frames") or 0),
                "nfo": "ja" if data.get("nfo_path") else "nein",
                "fertig_am": (data.get("finished_at") or "")[:19],
                "fehler": data.get("error", ""),
            }
        )

    return rows

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
if "analysis_summary_rows" not in st.session_state:
    st.session_state.analysis_summary_rows = []
if "import_candidates" not in st.session_state:
    st.session_state.import_candidates = []
if "import_preview_rows" not in st.session_state:
    st.session_state.import_preview_rows = []
if "import_last_results" not in st.session_state:
    st.session_state.import_last_results = []
if "import_photo_candidates" not in st.session_state:
    st.session_state.import_photo_candidates = []
if "import_photo_preview_rows" not in st.session_state:
    st.session_state.import_photo_preview_rows = []
if "import_photo_last_results" not in st.session_state:
    st.session_state.import_photo_last_results = []

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

current_page = _get_current_page()
page_labels = {
    "analyse": "📊 Analyse",
    "import": "🗂️ Import (Seite 2)",
}
selected_page_label = st.radio(
    "Navigation",
    list(page_labels.values()),
    index=0 if current_page == "analyse" else 1,
    horizontal=True,
)
selected_page = next(key for key, value in page_labels.items() if value == selected_page_label)
if selected_page != current_page:
    st.query_params["page"] = selected_page
    current_page = selected_page

st.caption("Direktlink zur zweiten Seite: http://localhost:8501/?page=import")

if current_page == "analyse":
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
                st.session_state.analysis_summary_rows = []

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

                status_text.info("⏳ Analyse läuft...")

                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
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
                        st.session_state.analysis_summary_rows = _build_analysis_summary(mode, input_path, output_path)
                        status_text.success("✅ Erfolgreich abgeschlossen!")
                        progress_bar.progress(1.0)
                    else:
                        st.session_state.analysis_running = False
                        status_text.error(f"❌ Fehler: Prozess endete mit Code {process.returncode}")

                except Exception as e:
                    st.session_state.analysis_running = False
                    status_text.error(f"❌ Fehler: {e}")

        st.subheader("📋 Log")
        log_display = st.empty()

        if st.session_state.log_output:
            log_display.text_area(
                "Analyse-Log",
                value=st.session_state.log_output,
                height=300,
                disabled=True,
                key="log_area",
            )
        else:
            log_display.info("Kein Log verfügbar. Starten Sie eine Analyse.")

        if st.session_state.analysis_summary_rows:
            rows = st.session_state.analysis_summary_rows
            ok_count = sum(1 for r in rows if r.get("status") == "completed")
            err_count = sum(1 for r in rows if r.get("status") == "error")
            st.subheader("✅ Zusammenfassung")
            st.caption(f"Analysierte Videos: {len(rows)} | Erfolgreich: {ok_count} | Fehler: {err_count}")
            st.dataframe(rows, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("ℹ️ Info")

        st.info(f"""
        **Version:** 1.1.0

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
        - 🗂️ MoviemetaDb Import (Seite 2)
        """)

        st.subheader("🔗 Schnelllinks")
        st.markdown("[Zur Import-Seite wechseln](?page=import)")
        if st.button("🔄 Seite aktualisieren"):
            st.rerun()

if current_page == "import":
    st.subheader("🗂️ MoviemetaDb Import")
    st.caption("Manuelle Eingabe oder Verzeichnis durchsuchen und importieren")

    api_col1, api_col2 = st.columns([2, 1])
    with api_col1:
        moviedb_url = st.text_input(
            "MoviemetaDb URL",
            value=os.getenv("MOVIEMETADB_URL", "http://127.0.0.1:8001"),
            key="moviedb_url_input",
        )
    with api_col2:
        moviedb_key = st.text_input(
            "API Key (optional)",
            value=os.getenv("MOVIEMETADB_API_KEY", ""),
            type="password",
            key="moviedb_key_input",
        )

    st.markdown("### ✍️ Manuelle Eingabe")
    man_col1, man_col2, man_col3 = st.columns(3)
    with man_col1:
        manual_title = st.text_input("Titel", key="manual_title")
    with man_col2:
        manual_year = st.number_input("Jahr", min_value=0, max_value=2100, value=0, step=1, key="manual_year")
    with man_col3:
        manual_rating = st.number_input("Rating", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="manual_rating")

    manual_file_path = st.text_input("Dateipfad (optional)", key="manual_file_path")
    manual_language = st.text_input("Sprache (optional)", value="", key="manual_language")
    manual_plot = st.text_area("Plot / Beschreibung (optional)", key="manual_plot")

    if st.button("➕ Manuell importieren", key="manual_import_btn"):
        if not manual_title.strip():
            st.error("Bitte einen Titel eingeben.")
        else:
            payload = {
                "title": manual_title.strip(),
                "year": int(manual_year),
                "rating": float(manual_rating),
                "file_path": manual_file_path.strip(),
                "language": manual_language.strip(),
                "plot": manual_plot.strip(),
                "analysed_at": datetime.now().isoformat(),
            }
            try:
                resp = _post_movie(moviedb_url, moviedb_key, payload)
                if resp.status_code in (200, 201):
                    st.success(f"✅ Importiert: {payload['title']} ({payload['year']})")
                else:
                    st.error(f"❌ API Fehler {resp.status_code}: {resp.text[:400]}")
            except Exception as exc:
                st.error(f"❌ Verbindungsfehler: {exc}")

    st.markdown("### 📁 Verzeichnis durchsuchen")
    scan_col1, scan_col2 = st.columns([3, 1])
    with scan_col1:
        scan_dir = st.text_input("Verzeichnis", placeholder="/pfad/zum/video-ordner", key="scan_dir_input")
    with scan_col2:
        recursive_scan = st.checkbox("Rekursiv", value=True, key="scan_recursive")

    if st.button("🔎 Verzeichnis scannen", key="scan_btn"):
        if not scan_dir.strip():
            st.error("Bitte ein Verzeichnis angeben.")
        else:
            files = _scan_video_files(scan_dir.strip(), recursive=recursive_scan)
            candidates = []
            nfo_found_count = 0
            for p in files:
                guessed_title, guessed_year = _guess_title_year(p)
                nfo_data = _read_emb_nfo(p)
                if nfo_data:
                    nfo_found_count += 1

                title = nfo_data.get("title") or guessed_title
                year = int(nfo_data.get("year") or guessed_year or 0)
                candidates.append({
                    "path": str(p),
                    "title": title,
                    "year": year,
                    "plot": nfo_data.get("plot", ""),
                    "runtime_minutes": int(nfo_data.get("runtime_minutes") or 0),
                    "nfo_path": nfo_data.get("nfo_path", ""),
                })
            st.session_state.import_candidates = candidates
            st.session_state.import_preview_rows = [
                {
                    "import": True,
                    "title": c["title"],
                    "year": int(c["year"]),
                    "rating": 0.0,
                    "language": "",
                    "plot": c.get("plot", ""),
                    "runtime_minutes": int(c.get("runtime_minutes") or 0),
                    "nfo_path": c.get("nfo_path", ""),
                    "path": c["path"],
                }
                for c in candidates
            ]
            st.success(f"{len(candidates)} Videodatei(en) gefunden. EMB-NFO erkannt: {nfo_found_count}")

    candidates = st.session_state.import_candidates
    if candidates:
        st.markdown("### 📥 Gefundene Dateien (anpassbar)")
        st.caption("Du kannst Titel/Jahr/Rating/Sprache/Plot pro Eintrag vor dem Import ändern. Wenn vorhanden, werden EMB-NFO-Dateien automatisch ausgelesen.")

        imp_col1, imp_col2, imp_col3 = st.columns(3)
        with imp_col1:
            import_rating = st.number_input("Standard-Rating", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="import_rating")
        with imp_col2:
            import_language = st.text_input("Standard-Sprache", value="", key="import_language")
        with imp_col3:
            apply_defaults = st.button("🧩 Standardwerte auf alle anwenden", key="apply_defaults_btn")

        if apply_defaults:
            updated_rows = []
            for row in st.session_state.import_preview_rows:
                current = dict(row)
                current["rating"] = float(import_rating)
                current["language"] = import_language.strip()
                updated_rows.append(current)
            st.session_state.import_preview_rows = updated_rows
            st.rerun()

        edited_rows = st.data_editor(
            st.session_state.import_preview_rows,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            key="import_preview_editor",
            column_config={
                "import": st.column_config.CheckboxColumn("Import"),
                "title": st.column_config.TextColumn("Titel"),
                "year": st.column_config.NumberColumn("Jahr", min_value=0, max_value=2100, step=1),
                "rating": st.column_config.NumberColumn("Rating", min_value=0.0, max_value=10.0, step=0.1),
                "language": st.column_config.TextColumn("Sprache"),
                "plot": st.column_config.TextColumn("Plot"),
                "runtime_minutes": st.column_config.NumberColumn("Laufzeit (Min)", min_value=0, step=1),
                "nfo_path": st.column_config.TextColumn("NFO", disabled=True),
                "path": st.column_config.TextColumn("Dateipfad", disabled=True),
            },
        )

        selected_rows = [r for r in edited_rows if r.get("import")]
        st.caption(f"Ausgewählt für Import: {len(selected_rows)} von {len(edited_rows)}")

        if st.button("⬆️ Auswahl importieren", key="bulk_import_btn"):
            if not selected_rows:
                st.warning("Keine Dateien ausgewählt.")
            else:
                ok_count = 0
                err_count = 0
                progress = st.progress(0.0)
                error_messages = []
                result_rows = []

                for idx, row in enumerate(selected_rows, start=1):
                    payload = {
                        "title": str(row.get("title", "")).strip(),
                        "year": int(row.get("year") or 0),
                        "rating": float(row.get("rating") or 0.0),
                        "file_path": str(row.get("path", "")).strip(),
                        "duration_seconds": float(int(row.get("runtime_minutes") or 0) * 60),
                        "language": str(row.get("language", "")).strip(),
                        "plot": str(row.get("plot", "")).strip(),
                        "analysed_at": datetime.now().isoformat(),
                    }

                    if not payload["title"]:
                        err_count += 1
                        error_messages.append(f"Zeile {idx}: Titel fehlt")
                        result_rows.append({"title": "", "year": payload["year"], "status": "error", "detail": "Titel fehlt"})
                        progress.progress(idx / len(selected_rows))
                        continue

                    try:
                        resp = _post_movie(moviedb_url, moviedb_key, payload)
                        if resp.status_code in (200, 201):
                            ok_count += 1
                            result_rows.append(
                                {
                                    "title": payload["title"],
                                    "year": payload["year"],
                                    "status": "ok",
                                    "detail": "importiert",
                                }
                            )
                        else:
                            err_count += 1
                            detail = f"API {resp.status_code}: {resp.text[:180]}"
                            error_messages.append(f"{payload['title']}: {detail}")
                            result_rows.append(
                                {
                                    "title": payload["title"],
                                    "year": payload["year"],
                                    "status": "error",
                                    "detail": detail,
                                }
                            )
                    except Exception as exc:
                        err_count += 1
                        error_messages.append(f"{payload['title']}: {exc}")
                        result_rows.append(
                            {
                                "title": payload["title"],
                                "year": payload["year"],
                                "status": "error",
                                "detail": str(exc),
                            }
                        )

                    progress.progress(idx / len(selected_rows))

                st.session_state.import_last_results = result_rows

                if ok_count:
                    st.success(f"✅ {ok_count} Einträge importiert.")
                if err_count:
                    st.error(f"❌ {err_count} Einträge fehlgeschlagen.")
                    st.text_area("Fehlerdetails", "\n".join(error_messages), height=120)

    if st.session_state.import_last_results:
        st.markdown("### 📄 Letzter Import-Report (Videos)")
        st.dataframe(st.session_state.import_last_results, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 📷 Fotos importieren")
    st.caption("Bilddateien aus einem Verzeichnis scannen und in MoviemetaDb importieren")

    photo_scan_col1, photo_scan_col2 = st.columns([3, 1])
    with photo_scan_col1:
        photo_scan_dir = st.text_input(
            "Verzeichnis (Fotos)",
            placeholder="/pfad/zum/foto-ordner",
            key="photo_scan_dir_input",
        )
    with photo_scan_col2:
        photo_recursive = st.checkbox("Rekursiv", value=True, key="photo_scan_recursive")

    if st.button("🔎 Fotos scannen", key="photo_scan_btn"):
        if not photo_scan_dir.strip():
            st.error("Bitte ein Verzeichnis angeben.")
        else:
            photos = _scan_photo_files(photo_scan_dir.strip(), recursive=photo_recursive)
            photo_candidates = []
            for p in photos:
                album = p.parent.name
                photo_candidates.append({"path": str(p), "album": album})
            st.session_state.import_photo_candidates = photo_candidates
            st.session_state.import_photo_preview_rows = [
                {
                    "import": True,
                    "file_path": c["path"],
                    "album": c["album"],
                    "description": "",
                    "tags": "",
                    "camera": "",
                    "taken_at": "",
                }
                for c in photo_candidates
            ]
            st.success(f"{len(photo_candidates)} Bilddatei(en) gefunden.")

    photo_candidates = st.session_state.import_photo_candidates
    if photo_candidates:
        st.markdown("#### 📥 Gefundene Fotos (anpassbar)")
        st.caption("Album, Beschreibung, Tags und Kamera können vor dem Import angepasst werden.")

        edited_photos = st.data_editor(
            st.session_state.import_photo_preview_rows,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            key="photo_preview_editor",
            column_config={
                "import": st.column_config.CheckboxColumn("Import"),
                "file_path": st.column_config.TextColumn("Dateipfad", disabled=True),
                "album": st.column_config.TextColumn("Album"),
                "description": st.column_config.TextColumn("Beschreibung"),
                "tags": st.column_config.TextColumn("Tags (kommagetrennt)"),
                "camera": st.column_config.TextColumn("Kamera"),
                "taken_at": st.column_config.TextColumn("Aufnahmedatum"),
            },
        )

        selected_photos = [r for r in edited_photos if r.get("import")]
        st.caption(f"Ausgewählt für Import: {len(selected_photos)} von {len(edited_photos)}")

        if st.button("⬆️ Fotos importieren", key="photo_bulk_import_btn"):
            if not selected_photos:
                st.warning("Keine Fotos ausgewählt.")
            else:
                ok_count = 0
                err_count = 0
                progress = st.progress(0.0)
                photo_results = []

                for idx, row in enumerate(selected_photos, start=1):
                    payload = {
                        "file_path": str(row.get("file_path", "")).strip(),
                        "album": str(row.get("album", "")).strip(),
                        "description": str(row.get("description", "")).strip(),
                        "tags": str(row.get("tags", "")).strip(),
                        "camera": str(row.get("camera", "")).strip(),
                        "taken_at": str(row.get("taken_at", "")).strip(),
                    }

                    if not payload["file_path"]:
                        err_count += 1
                        photo_results.append({"file_path": "", "status": "error", "detail": "Pfad fehlt"})
                        progress.progress(idx / len(selected_photos))
                        continue

                    try:
                        resp = _post_photo(moviedb_url, moviedb_key, payload)
                        if resp.status_code in (200, 201):
                            ok_count += 1
                            photo_results.append({"file_path": payload["file_path"], "status": "ok", "detail": "importiert"})
                        else:
                            err_count += 1
                            photo_results.append({"file_path": payload["file_path"], "status": "error", "detail": f"API {resp.status_code}: {resp.text[:180]}"})
                    except Exception as exc:
                        err_count += 1
                        photo_results.append({"file_path": payload["file_path"], "status": "error", "detail": str(exc)})

                    progress.progress(idx / len(selected_photos))

                st.session_state.import_photo_last_results = photo_results

                if ok_count:
                    st.success(f"✅ {ok_count} Foto(s) importiert.")
                if err_count:
                    st.error(f"❌ {err_count} Foto(s) fehlgeschlagen.")

    if st.session_state.import_photo_last_results:
        st.markdown("#### 📄 Letzter Foto-Import-Report")
        st.dataframe(st.session_state.import_photo_last_results, use_container_width=True, hide_index=True)

# Footer
st.divider()
st.caption(f"Videoanalyse EMB WebUI | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
