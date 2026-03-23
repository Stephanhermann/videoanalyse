#!/Users/stephanhermann/videoanalyse_venv/bin/python3
import os
import re
import sys
import cv2
import json
import base64
import argparse
import traceback
import warnings

# Suppress LibreSSL warning on macOS system Python/OpenSSL combinations.
warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)

import urllib3
import requests
import whisper
from functools import lru_cache
from typing import Optional
from moviepy.editor import VideoFileClip
from datetime import datetime
import subprocess


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)


def _load_dotenv():
    """Load .env file from the script directory if it exists."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:  # don't overwrite existing env vars
                os.environ[key] = value


_load_dotenv()

http = requests.Session()
http.verify = False

SCRIPT_VERSION = "1.0.0"
DEFAULT_TIMEOUT = 120


def log(msg: str):
    print(msg, flush=True)


def emit_progress(stage: str, percent: int, message: str = "", current_video: int = 0, total_videos: int = 0):
    payload = {
        "type": "progress",
        "stage": stage,
        "percent": int(percent),
        "message": message,
        "current_video": current_video,
        "total_videos": total_videos,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def safe_mkdir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_video_basename(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def get_output_dir(video_path: str, output_dir: Optional[str]) -> str:
    if output_dir:
        safe_mkdir(output_dir)
        return output_dir

    base_dir = os.path.dirname(os.path.abspath(video_path))
    out_dir = os.path.join(base_dir, "output", get_video_basename(video_path))
    safe_mkdir(out_dir)
    return out_dir


def find_video_files(folder_path: str):
    video_extensions = (".mp4", ".mkv", ".mov", ".avi", ".m4v")
    results = []

    for root, dirs, files in os.walk(folder_path):
        for name in files:
            if name.lower().endswith(video_extensions):
                results.append(os.path.join(root, name))

    return sorted(results)


def write_json(output_dir: str, name: str, data: dict):
    path = os.path.join(output_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log(f"JSON geschrieben: {path}")


def write_txt(output_dir: str, name: str, text: str):
    path = os.path.join(output_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text or "")
    log(f"Text geschrieben: {path}")


def xml_escape(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def write_nfo(output_dir: str, title: str, plot: str, video_info: dict):
    nfo_path = os.path.join(output_dir, f"{title}.nfo")

    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<movie>
  <title>{xml_escape(title)}</title>
  <plot>{xml_escape(plot)}</plot>
  <outline>{xml_escape(plot[:300])}</outline>
  <runtime>{int(video_info.get("duration_seconds", 0) // 60)}</runtime>
  <premiered>{datetime.now().strftime("%Y-%m-%d")}</premiered>
</movie>
"""

    with open(nfo_path, "w", encoding="utf-8") as f:
        f.write(xml)

    log(f"NFO geschrieben: {nfo_path}")
    return nfo_path


def analyse_video_basic(video_path: str, output_dir: str) -> dict:
    log("Öffne Video mit OpenCV ...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 else 0

        log(f"Frames gesamt: {frame_count}")
        log(f"FPS: {fps:.2f}")
        log(f"Auflösung: {width}x{height}")
        log(f"Dauer: {duration:.2f} Sekunden")

        preview_path = None
        ok, frame = cap.read()
        if ok:
            preview_path = os.path.join(output_dir, "preview.jpg")
            cv2.imwrite(preview_path, frame)
            log(f"Vorschaubild gespeichert: {preview_path}")

        return {
            "frame_count": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "preview_path": preview_path,
        }
    finally:
        cap.release()


def extract_frames(video_path: str, output_dir: str, interval_seconds: int = 10):
    log("Extrahiere Frames ...")

    frames_dir = os.path.join(output_dir, "frames")
    safe_mkdir(frames_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video konnte nicht geöffnet werden: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            log("Warnung: FPS konnte nicht ermittelt werden, verwende 25.0")
            fps = 25.0

        frame_interval = max(1, int(fps * max(1, interval_seconds)))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_index = 0
        saved = 0
        saved_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                timestamp_seconds = frame_index / fps
                filename = f"frame_{saved:05d}_{int(timestamp_seconds):06d}s.jpg"
                full_path = os.path.join(frames_dir, filename)
                cv2.imwrite(full_path, frame)

                saved_frames.append({
                    "index": saved,
                    "frame_number": frame_index,
                    "timestamp_seconds": round(timestamp_seconds, 2),
                    "file": filename,
                    "path": full_path,
                })
                saved += 1

            frame_index += 1

        log(f"Frames gespeichert: {saved} (gesamt {frame_count})")

        return {
            "frames_dir": frames_dir,
            "saved_frames": saved,
            "interval_seconds": interval_seconds,
            "frames": saved_frames,
            "frame_count": frame_count,
            "fps": fps,
        }
    finally:
        cap.release()


def analyse_with_moviepy(video_path: str) -> dict:
    log("Lese Video mit MoviePy ...")

    clip = VideoFileClip(video_path)
    duration = float(clip.duration or 0.0)
    audio_present = clip.audio is not None
    size = tuple(clip.size) if clip.size else (0, 0)

    try:
        clip.close()
    except Exception:
        pass

    return {
        "moviepy_duration": duration,
        "audio_present": audio_present,
        "moviepy_size": size,
    }


def _get_best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@lru_cache(maxsize=4)
def load_whisper_model(whisper_model_name: str):
    device = _get_best_device()
    log(f"Lade Whisper-Modell (cached): {whisper_model_name} [device={device}]")
    return whisper.load_model(whisper_model_name, device=device)


def transcribe_with_whisper(video_path: str, whisper_model_name: str) -> dict:
    log(f"Verwende Whisper-Modell: {whisper_model_name}")
    model = load_whisper_model(whisper_model_name)

    log("Starte Audio-Transkription ...")
    result = model.transcribe(video_path)

    text = (result or {}).get("text", "").strip()
    language = (result or {}).get("language", "")

    log("Whisper abgeschlossen.")
    if language:
        log(f"Erkannte Sprache: {language}")

    return {
        "language": language,
        "text": text,
        "segments": (result or {}).get("segments", []),
        "raw": result,
    }


def check_ollama_server(ollama_url: str):
    url = ollama_url.rstrip("/") + "/api/tags"
    log(f"Prüfe Ollama-Server: {url}")

    r = requests.get(url, timeout=20)
    r.raise_for_status()

    data = r.json()
    models = [m.get("name", "") for m in data.get("models", [])]

    log(f"Ollama erreichbar. Modelle gefunden: {len(models)}")
    return models


def normalize_model_name(model_name: str):
    """Return normalized full and base model names for robust comparisons."""
    norm = (model_name or "").strip().lower()
    if not norm:
        return "", ""
    base = norm.split(":", 1)[0]
    return norm, base


def is_ollama_model_available(requested_model: str, available_models):
    """Match models case-insensitively and independent of tag (e.g. :latest)."""
    req_full, req_base = normalize_model_name(requested_model)
    if not req_full:
        return False

    for model in available_models or []:
        model_full, model_base = normalize_model_name(model)
        if not model_full:
            continue

        if model_full == req_full:
            return True
        if model_base == req_base:
            return True

        # Common Ollama case: requested without tag, available with :latest.
        if ":" not in req_full and model_full == f"{req_full}:latest":
            return True

    return False


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def build_vision_prompt():
    return (
        "du analysierst ein Einzelbild aus einem Video"
        "Beschreibe das Bild so präzise wie moeglich auf Deutsch"
        "Antworte ausschließlich strukturiert in folgenden Kategorien"
        "Szene"
       "Kurzbeschreibung der Situation im Bild"
        "Umgebung"
      "Ort oder Umgebung, Innenraum, Studio, Strasse, Natur, Bildschirmoberflaeche usw."
       "Personen"
       "Anzahl der Personen, Aussehen, Kleidung, Pose, Blickrichtung" 
       "Objekte"
       "Wichtige sichtbare Gegenstaende oder Elemente im Bild"
       "Handlung"
       "Was passiert in diesem Moment im Bild"
       "Stimmung / Atmosphäre"
       "Welche Stimmung vermittelt das Bild"
        "Auffaellige Details"
       "Besondere oder ungewoehnliche visuelle Details"
        "Unsicherheiten"
       "Welche Dinge sind im Bild nicht eindeutig erkennbar"
        "Antworte kurz, praezise und nur mit den genannten Kategorien"
       "Erfinde keine Details"
    )


def analyse_single_frame_with_ollama(image_path: str, ollama_url: str, vision_model: str) -> dict:
    url = ollama_url.rstrip("/") + "/api/generate"
    image_b64 = image_to_base64(image_path)

    payload = {
        "model": vision_model,
        "prompt": build_vision_prompt(),
        "images": [image_b64],
        "stream": False
    }

    r = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()

    data = r.json()
    response_text = data.get("response", "").strip()

    return {
        "model": vision_model,
        "response": response_text,
        "done": data.get("done", True),
        "total_duration": data.get("total_duration"),
        "eval_count": data.get("eval_count"),
    }


def analyse_frames_with_ollama(
    frame_info: dict,
    ollama_url: str,
    vision_model: str,
    max_frames: int = 20,
    current_video: int = 1,
    total_videos: int = 1
) -> dict:
    frames = frame_info.get("frames", [])
    selected = frames[:max_frames]

    if not selected:
        log("Keine Frames für Vision-Analyse vorhanden.")
        return {
            "analysis_model": vision_model,
            "ollama_url": ollama_url,
            "analysed_frames": 0,
            "frame_descriptions": [],
        }

    log(f"Starte Ollama-Vision-Analyse für {len(selected)} Frames ...")

    results = []

    for idx, frame in enumerate(selected, start=1):
        image_path = frame["path"]
        log(f"[Vision {idx}/{len(selected)}] Analysiere {os.path.basename(image_path)}")

        sub_percent = 65 + int((idx / max(len(selected), 1)) * 25)
        emit_progress(
            "vision_frame",
            sub_percent,
            f"Vision Frame {idx}/{len(selected)}",
            current_video,
            total_videos,
        )

        try:
            ollama_result = analyse_single_frame_with_ollama(
                image_path=image_path,
                ollama_url=ollama_url,
                vision_model=vision_model,
            )

            results.append({
                "index": frame["index"],
                "frame_number": frame["frame_number"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "file": frame["file"],
                "description": ollama_result.get("response", ""),
                "meta": {
                    "model": ollama_result.get("model"),
                    "done": ollama_result.get("done"),
                    "total_duration": ollama_result.get("total_duration"),
                    "eval_count": ollama_result.get("eval_count"),
                }
            })

        except Exception as e:
            results.append({
                "index": frame["index"],
                "frame_number": frame["frame_number"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "file": frame["file"],
                "description": "",
                "error": str(e),
            })

    log("Ollama-Vision-Analyse abgeschlossen.")

    return {
        "analysis_model": vision_model,
        "ollama_url": ollama_url,
        "analysed_frames": len(results),
        "frame_descriptions": results,
    }


def update_emby_library(emby_url: str, emby_api_key: str):
    if not emby_url or not emby_api_key:
        log("Emby-Update übersprungen: URL oder API-Key fehlt.")
        return

    refresh_url = emby_url.rstrip("/") + "/Library/Refresh"
    headers = {
        "X-Emby-Token": emby_api_key,
        "Content-Type": "application/json",
    }

    log(f"Sende Emby Refresh an: {refresh_url}")

    response = http.post(
        refresh_url,
        headers=headers,
        timeout=DEFAULT_TIMEOUT,
    )

    log(f"Emby HTTP Status: {response.status_code}")

    if response.status_code not in (200, 204):
        raise RuntimeError(
            f"Emby Refresh fehlgeschlagen: HTTP {response.status_code} - {response.text}"
        )

    log("Emby-Bibliothek erfolgreich aktualisiert.")


def save_summary(output_dir: str, summary: dict):
    summary_path = os.path.join(output_dir, "minicpm_v_analysis.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"Summary geschrieben: {summary_path}")


def push_to_moviemetadb(summary: dict, args) -> None:
    """Analyseergebnisse an MoviemetaDb API senden, falls konfiguriert."""
    moviemetadb_url = os.getenv("MOVIEMETADB_URL", "").rstrip("/")
    if not moviemetadb_url:
        return

    api_key = os.getenv("MOVIEMETADB_API_KEY", "")
    video_info = summary.get("video_info", {})
    whisper = summary.get("whisper", {})
    moviepy_info = summary.get("moviepy_info", {})

    # Titel und Jahr aus Dateiname extrahieren
    input_video = summary.get("input_video", "")
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    title = base_name
    year = 0
    year_match = re.search(r'\b(19|20)\d{2}\b', base_name)
    if year_match:
        year = int(year_match.group())
        title = re.sub(r'[\._\-\s]+$', '', base_name[:year_match.start()]).strip()

    # Vision-Zusammenfassung (erste 5 Frames)
    vision_summary = ""
    if "vision_analysis" in summary:
        frames_data = summary["vision_analysis"].get("frames", [])
        snippets = [f.get("response", "") for f in frames_data if f.get("response")]
        vision_summary = "\n\n".join(snippets[:5])

    transcript_text = whisper.get("text_preview", "")
    payload = {
        "title": title if title else base_name,
        "year": year,
        "rating": 0.0,
        "file_path": input_video,
        "duration_seconds": float(moviepy_info.get("duration") or video_info.get("duration_seconds") or 0),
        "width": int(video_info.get("width") or 0),
        "height": int(video_info.get("height") or 0),
        "fps": float(video_info.get("fps") or 0),
        "language": whisper.get("language", ""),
        "transcript": transcript_text,
        "plot": transcript_text[:500] if transcript_text else vision_summary[:500],
        "preview_path": video_info.get("preview_frame_path", ""),
        "vision_model": summary.get("vision_model", ""),
        "whisper_model": summary.get("whisper_model", ""),
        "analysed_at": summary.get("finished_at", ""),
    }

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = http.post(f"{moviemetadb_url}/movies", json=payload, headers=headers, timeout=30)
        if resp.ok:
            log(f"MoviemetaDb: '{payload['title']}' gespeichert ({moviemetadb_url})")
        else:
            log(f"MoviemetaDb: Fehler {resp.status_code} – {resp.text[:200]}")
    except Exception as exc:
        log(f"MoviemetaDb: Verbindungsfehler – {exc}")


def run_single_video(video_path: str, output_dir: str, args, current_video: int = 1, total_videos: int = 1):
    safe_mkdir(output_dir)

    base_name = get_video_basename(video_path)

    log("======================================")
    log("Videoanalyse EMB gestartet")
    log("======================================")
    log(f"Input: {video_path}")
    log(f"Output: {output_dir}")
    log(f"Vision-Modell: {args.vision_model}")
    log(f"Whisper-Modell: {args.whisper_model}")

    summary = {
        "input_video": video_path,
        "output_dir": output_dir,
        "vision_model": args.vision_model,
        "whisper_model": args.whisper_model,
        "ollama_url": args.ollama_url,
        "started_at": datetime.now().isoformat(),
        "status": "running",
    }

    emit_progress("start", 0, f"Starte Analyse: {base_name}", current_video, total_videos)
    save_summary(output_dir, summary)

    try:
        video_info = analyse_video_basic(video_path, output_dir)
        summary["video_info"] = video_info
        save_summary(output_dir, summary)
        emit_progress("video_info", 10, "Videoinfos gelesen", current_video, total_videos)

        frame_info = extract_frames(video_path, output_dir, interval_seconds=args.frame_interval)
        summary["frame_info"] = frame_info
        save_summary(output_dir, summary)
        emit_progress("frames", 30, f"Frames extrahiert: {frame_info.get('saved_frames', 0)}", current_video, total_videos)

        moviepy_info = analyse_with_moviepy(video_path)
        summary["moviepy_info"] = moviepy_info
        save_summary(output_dir, summary)
        emit_progress("moviepy", 40, "MoviePy Analyse fertig", current_video, total_videos)

        whisper_info = transcribe_with_whisper(video_path, args.whisper_model)
        summary["whisper"] = {
            "language": whisper_info.get("language", ""),
            "text_preview": whisper_info.get("text", "")[:2000],
            "segments_count": len(whisper_info.get("segments", [])),
        }
        write_txt(output_dir, "transcript.txt", whisper_info.get("text", ""))
        save_summary(output_dir, summary)
        emit_progress("whisper", 65, "Whisper Transkript erstellt", current_video, total_videos)

        if args.use_vision:
            vision_info = analyse_frames_with_ollama(
                frame_info=frame_info,
                ollama_url=args.ollama_url,
                vision_model=args.vision_model,
                max_frames=args.max_vision_frames,
                current_video=current_video,
                total_videos=total_videos,
            )
            summary["vision_analysis"] = vision_info
            save_summary(output_dir, summary)
            emit_progress("vision", 90, "Vision Analyse abgeschlossen", current_video, total_videos)
        else:
            log("Vision-Analyse übersprungen (--use-vision nicht gesetzt).")
            emit_progress("vision_skip", 90, "Vision Analyse übersprungen", current_video, total_videos)

        if args.write_nfo:
            plot = whisper_info.get("text", "").strip()
            if not plot:
                plot = "Automatisch erzeugte Videoanalyse ohne Transkriptinhalt."
            nfo_path = write_nfo(output_dir, base_name, plot, video_info)
            summary["nfo_path"] = nfo_path
            save_summary(output_dir, summary)
            emit_progress("nfo", 95, "NFO geschrieben", current_video, total_videos)

        summary["finished_at"] = datetime.now().isoformat()
        summary["status"] = "completed"
        save_summary(output_dir, summary)

        if getattr(args, "use_moviemetadb", False):
            push_to_moviemetadb(summary, args)

        emit_progress("done", 100, f"Analyse abgeschlossen: {base_name}", current_video, total_videos)
        log("Analyse abgeschlossen")

    except Exception as e:
        summary["status"] = "error"
        summary["error"] = str(e)
        summary["finished_at"] = datetime.now().isoformat()
        save_summary(output_dir, summary)
        log(f"Fehler bei Videoanalyse: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Videoanalyse EMB mit Ollama Vision")

    parser.add_argument("--input", help="Pfad zur Videodatei")
    parser.add_argument("--input-folder", help="Ordner mit Videodateien")
    parser.add_argument("--output", help="Ausgabeordner")

    parser.add_argument("--vision-model", default="minicpm-v", help="Ollama Vision Modell")
    parser.add_argument("--whisper-model", default="base", help="Whisper Modell")
    parser.add_argument("--frame-interval", type=int, default=10, help="Frame-Abstand in Sekunden")
    parser.add_argument("--max-vision-frames", type=int, default=20, help="Maximal zu analysierende Frames pro Video")

    parser.add_argument("--use-vision", action="store_true", default=os.getenv("USE_VISION", "").lower() in ("1", "true", "yes"), help="Ollama Vision aktivieren")
    parser.add_argument("--write-nfo", action="store_true", default=os.getenv("WRITE_NFO", "").lower() in ("1", "true", "yes"), help="NFO schreiben")
    parser.add_argument("--update-emby", action="store_true", default=os.getenv("UPDATE_EMBY", "").lower() in ("1", "true", "yes"), help="Emby Bibliothek aktualisieren")
    parser.add_argument("--use-moviemetadb", action="store_true", default=os.getenv("USE_MOVIEMETADB", "").lower() in ("1", "true", "yes"), help="Ergebnisse an MoviemetaDb API senden")

    parser.add_argument("--ollama-url", default=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"), help="Ollama Basis-URL")
    parser.add_argument("--emby-url", default=os.getenv("EMBY_URL", ""), help="Emby Basis-URL")
    parser.add_argument("--emby-api-key", default=os.getenv("EMBY_API_KEY", ""), help="Emby API Key")
    parser.add_argument("--version", action="store_true", help="Zeige Skriptversion und beende")
    parser.add_argument("--gui", action="store_true", help="Starte die GUI (videoanalyse_mac_gui.py)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.version:
        print(f"videoanalyse_emb.py version: {SCRIPT_VERSION}")
        sys.exit(0)

    if args.gui:
        gui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videoanalyse_mac_gui.py")
        if not os.path.exists(gui_path):
            raise FileNotFoundError(f"GUI-Script nicht gefunden: {gui_path}")

        try:
            subprocess.run([sys.executable, gui_path], check=True)
        except subprocess.CalledProcessError as e:
            log(f"GUI konnte nicht gestartet werden: {e}")
            log("Bitte starte die GUI direkt mit `python3 videoanalyse_mac_gui.py` oder nutze den CLI-Modus.")
            return
        except Exception as e:
            log(f"Unerwarteter Fehler beim Starten der GUI: {e}")
            return

        return

    if not args.input and not args.input_folder:
        raise ValueError("Bitte --input oder --input-folder angeben.")

    if args.use_vision:
        models = check_ollama_server(args.ollama_url)
        if models and not is_ollama_model_available(args.vision_model, models):
            log(f"Hinweis: Modell '{args.vision_model}' ist nicht in ollama list gefunden worden.")
            preview = ", ".join(models[:5])
            if preview:
                log(f"Verfügbare Modelle (Auszug): {preview}")
            log("Falls nötig: ollama pull " + args.vision_model)

    if args.input_folder:
        folder_path = os.path.abspath(args.input_folder)
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Ordner nicht gefunden: {folder_path}")

        video_files = find_video_files(folder_path)
        if not video_files:
            raise RuntimeError(f"Keine Videodateien gefunden in: {folder_path}")

        if args.output:
            batch_output_root = os.path.abspath(args.output)
        else:
            batch_output_root = os.path.join(folder_path, "output")

        safe_mkdir(batch_output_root)

        log("======================================")
        log("Batch-Analyse gestartet")
        log("======================================")
        log(f"Ordner: {folder_path}")
        log(f"Output Root: {batch_output_root}")
        log(f"Gefundene Videos: {len(video_files)}")

        for idx, video_path in enumerate(video_files, start=1):
            log("")
            log("======================================")
            log(f"[{idx}/{len(video_files)}] Verarbeite: {video_path}")
            log("======================================")

            base_name = get_video_basename(video_path)
            video_output_dir = os.path.join(batch_output_root, base_name)
            safe_mkdir(video_output_dir)

            run_single_video(
                video_path,
                video_output_dir,
                args,
                current_video=idx,
                total_videos=len(video_files)
            )

        if args.update_emby:
            update_emby_library(args.emby_url, args.emby_api_key)

        log("")
        log("======================================")
        log("Batch-Analyse abgeschlossen")
        log("======================================")
        return

    video_path = os.path.abspath(args.input)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {video_path}")

    # Handle user input where a folder is passed to --input by mistake.
    if os.path.isdir(video_path):
        detected_videos = find_video_files(video_path)
        if not detected_videos:
            raise RuntimeError(
                f"Eingabepfad ist ein Ordner ohne Videodateien: {video_path}. "
                "Nutze --input-folder für Ordner oder gib eine konkrete Videodatei mit --input an."
            )

        if len(detected_videos) == 1:
            log(
                f"Hinweis: Ordner bei --input erkannt. "
                f"Verwende automatisch die einzige Videodatei: {detected_videos[0]}"
            )
            video_path = detected_videos[0]
            output_dir = get_output_dir(video_path, args.output)
            run_single_video(video_path, output_dir, args, current_video=1, total_videos=1)
        else:
            log(
                f"Hinweis: Ordner bei --input erkannt mit {len(detected_videos)} Videodateien. "
                "Starte automatisch Batch-Modus."
            )
            if args.output:
                batch_output_root = os.path.abspath(args.output)
            else:
                batch_output_root = os.path.join(video_path, "output")

            safe_mkdir(batch_output_root)

            for idx, folder_video_path in enumerate(detected_videos, start=1):
                base_name = get_video_basename(folder_video_path)
                video_output_dir = os.path.join(batch_output_root, base_name)
                safe_mkdir(video_output_dir)
                run_single_video(
                    folder_video_path,
                    video_output_dir,
                    args,
                    current_video=idx,
                    total_videos=len(detected_videos),
                )
    else:
        output_dir = get_output_dir(video_path, args.output)
        run_single_video(video_path, output_dir, args, current_video=1, total_videos=1)

    if args.update_emby:
        update_emby_library(args.emby_url, args.emby_api_key)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("======================================")
        log("FEHLER im Analyse-Script")
        log("======================================")
        log(str(e))
        log(traceback.format_exc())
        sys.exit(1)