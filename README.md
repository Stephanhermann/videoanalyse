# Videoanalyse EMB

Dieses Projekt implementiert eine Videoanalyse-Pipeline auf Python-Basis mit:

- OpenCV Video-Metadaten + Frame-Extraktion
- Whisper Audio-Transkription
- Ollama Vision-Analyse pro Frame
- NFO-Erstellung
- Emby Bibliotheks-Refresh
- MoviemetaDb-Import (manuell und per Verzeichnis-Scan)

## Usability-Modi

- `videoanalyse_emb.py`: CLI-Engine
- `videoanalyse_mac_gui.py`: Tkinter-GUI (macOS Tk-abhängig)
- `videoanalyse_gui.py`: Streamlit Web-GUI (empfohlen)
- `videoanalyse-interactive`: Terminal-Menü (kein GUI)
- `videoanalyse`: Wrapper (Wählt CLI/GUI)
- `videoanalyse-web`: Startet Streamlit Web-GUI
- `start-moviemetadb.sh`: Startet MoviemetaDb API lokal auf Port `8001`

## Installation (virtualenv)

```bash
cd ~/Downloads/output
python3 -m venv ~/videoanalyse_venv
source ~/videoanalyse_venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Quickstart

### CLI

```bash
cd ~/Downloads/output
./videoanalyse --input '/Pfad/zur/Datei.mp4' --use-vision --write-nfo
```

### Web GUI

```bash
cd ~/Downloads/output
./videoanalyse-web
```

### MoviemetaDb API starten

```bash
cd ~/Downloads/output
./start-moviemetadb.sh
```

Danach ist die API unter `http://127.0.0.1:8001` erreichbar.

### Import über GUI (Seite 2)

- In der Streamlit GUI den Tab `🗂️ Import (Seite 2)` öffnen.
- Entweder manuell Titel/Jahr usw. eintragen oder ein Verzeichnis scannen.
- Vor dem Import können alle gefundenen Einträge in der Tabelle angepasst werden.

### Interaktives Terminal

```bash
cd ~/Downloads/output
./videoanalyse-interactive
```

## Systemmonitor

Streamlit GUI zeigt jetzt:
- CPU-Auslastung
- RAM-Auslastung
- GPU-Auslastung (sofern vorhanden)
- Liniengrafik historischen Verlaufs (60 Werte)

## Git

```bash
git init
git add .
git commit -m "Initial commit"
```

## Hinweise

- GUI via `videoanalyse_mac_gui.py` kann auf `macOS 16` scheitern wegen fehlendem `_tkinter`.
- Streamlit ist plattformunabhängig und empfohlen.
- Für den Import muss MoviemetaDb laufen und `MOVIEMETADB_URL` (Standard: `http://127.0.0.1:8001`) erreichbar sein.
