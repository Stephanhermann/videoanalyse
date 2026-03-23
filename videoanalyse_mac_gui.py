#!/Users/stephanhermann/videoanalyse_venv/bin/python3
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess
import threading
import queue
import os
import sys
import json
import requests
from datetime import datetime


class VideoAnalyseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Videoanalyse EMB / AI Studio")
        self.root.geometry("1120x860")

        self.process = None
        self.log_queue = queue.Queue()

        self.input_mode = tk.StringVar(value="file")
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()

        self.vision_model = tk.StringVar(value="minicpm-v")
        self.whisper_model = tk.StringVar(value="base")
        self.frame_interval = tk.StringVar(value="10")
        self.max_vision_frames = tk.StringVar(value="20")
        self.ollama_url = tk.StringVar(value="http://127.0.0.1:11434")

        self.script_version = "1.0.0"

        self.vision_models = []

        self.use_vision = tk.BooleanVar(value=True)
        self.write_nfo = tk.BooleanVar(value=True)
        self.update_emby = tk.BooleanVar(value=False)

        self.current_logfile = None

        self.progress_value = tk.DoubleVar(value=0)
        self.progress_text = tk.StringVar(value="0 %")
        self.detail_text = tk.StringVar(value="Bereit")

        self.build_ui()
        self.load_ollama_models()
        self.root.after(100, self.process_log_queue)

    def build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        input_frame = ttk.LabelFrame(main, text="Eingabe", padding=10)
        input_frame.pack(fill="x", pady=(0, 8))

        mode_frame = ttk.Frame(input_frame)
        mode_frame.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        ttk.Radiobutton(mode_frame, text="Einzelvideo", variable=self.input_mode, value="file").pack(side="left", padx=5)
        ttk.Radiobutton(mode_frame, text="Ordner / Bulk", variable=self.input_mode, value="folder").pack(side="left", padx=5)

        ttk.Label(input_frame, text="Video / Ordner").grid(row=1, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.input_path, width=90).grid(row=1, column=1, padx=5, sticky="ew")
        ttk.Button(input_frame, text="Auswählen", command=self.select_input).grid(row=1, column=2)

        ttk.Label(input_frame, text="Ausgabeordner").grid(row=2, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.output_path, width=90).grid(row=2, column=1, padx=5, sticky="ew")
        ttk.Button(input_frame, text="Auswählen", command=self.select_output).grid(row=2, column=2)

        input_frame.columnconfigure(1, weight=1)

        settings_frame = ttk.LabelFrame(main, text="Analyse-Einstellungen", padding=10)
        settings_frame.pack(fill="x", pady=(0, 8))

        ttk.Label(settings_frame, text="Vision Modell").grid(row=0, column=0, sticky="w")
        self.vision_combobox = ttk.Combobox(settings_frame, textvariable=self.vision_model, values=self.vision_models, width=25)
        self.vision_combobox.grid(row=0, column=1, sticky="w", padx=(5, 15))
        ttk.Button(settings_frame, text="Ollama laden", command=self.load_ollama_models).grid(row=0, column=2, sticky="w")

        ttk.Label(settings_frame, text="Whisper Modell").grid(row=0, column=3, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.whisper_model, width=20).grid(row=0, column=4, sticky="w", padx=(5, 15))

        ttk.Label(settings_frame, text="Frame Intervall (Sek.)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings_frame, textvariable=self.frame_interval, width=10).grid(row=1, column=1, sticky="w", padx=(5, 15), pady=(8, 0))

        ttk.Label(settings_frame, text="Max Vision Frames").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings_frame, textvariable=self.max_vision_frames, width=10).grid(row=1, column=3, sticky="w", padx=(5, 15), pady=(8, 0))

        ttk.Label(settings_frame, text="Ollama URL").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(settings_frame, textvariable=self.ollama_url, width=40).grid(row=2, column=1, columnspan=3, sticky="w", padx=(5, 15), pady=(8, 0))

        options_frame = ttk.LabelFrame(main, text="Optionen", padding=10)
        options_frame.pack(fill="x", pady=(0, 8))

        ttk.Checkbutton(options_frame, text="Vision Analyse", variable=self.use_vision).pack(side="left", padx=5)
        ttk.Checkbutton(options_frame, text="NFO schreiben", variable=self.write_nfo).pack(side="left", padx=5)
        ttk.Checkbutton(options_frame, text="Emby aktualisieren", variable=self.update_emby).pack(side="left", padx=5)

        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=(0, 8))

        ttk.Button(controls, text="Start Analyse", command=self.start_analysis).pack(side="left", padx=4)
        ttk.Button(controls, text="Stop", command=self.stop_analysis).pack(side="left", padx=4)
        ttk.Button(controls, text="Log kopieren", command=self.copy_log).pack(side="left", padx=4)
        ttk.Button(controls, text="Log speichern", command=self.save_log_as).pack(side="left", padx=4)
        ttk.Button(controls, text="Log leeren", command=self.clear_log).pack(side="left", padx=4)

        progress_frame = ttk.LabelFrame(main, text="Fortschritt", padding=10)
        progress_frame.pack(fill="x", pady=(0, 8))

        self.progress = ttk.Progressbar(
            progress_frame,
            mode="determinate",
            maximum=100,
            variable=self.progress_value
        )
        self.progress.pack(fill="x", expand=True)

        info_frame = ttk.Frame(progress_frame)
        info_frame.pack(fill="x", pady=(6, 0))

        ttk.Label(info_frame, textvariable=self.progress_text).pack(side="left")
        ttk.Label(info_frame, textvariable=self.detail_text).pack(side="right")

        status_frame = ttk.Frame(main)
        status_frame.pack(fill="x")

        self.status = ttk.Label(status_frame, text="Bereit")
        self.status.pack(side="left", anchor="w", pady=(0, 6))

        self.version_label = ttk.Label(status_frame, text=f"Version: {self.script_version}")
        self.version_label.pack(side="right", anchor="e", pady=(0, 6))

        log_frame = ttk.LabelFrame(main, text="Log", padding=8)
        log_frame.pack(fill="both", expand=True)

        self.log = tk.Text(log_frame, wrap="word", undo=False)
        self.log.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        scrollbar.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scrollbar.set)

        self.log.bind("<Command-a>", self.select_all)
        self.log.bind("<Control-a>", self.select_all)
        self.log.bind("<Command-c>", self.copy_selected)
        self.log.bind("<Control-c>", self.copy_selected)

    def select_input(self):
        if self.input_mode.get() == "folder":
            folder = filedialog.askdirectory(title="Ordner mit Videos wählen")
            if folder:
                self.input_path.set(folder)
        else:
            file = filedialog.askopenfilename(
                title="Video wählen",
                filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi *.m4v"), ("Alle Dateien", "*.*")]
            )
            if file:
                self.input_path.set(file)

    def select_output(self):
        folder = filedialog.askdirectory(title="Ausgabeordner wählen")
        if folder:
            self.output_path.set(folder)

    def load_ollama_models(self):
        ollama_url = self.ollama_url.get().strip() or "http://127.0.0.1:11434"
        try:
            response = requests.get(ollama_url.rstrip("/") + "/api/tags", timeout=10)
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]

            if models:
                self.vision_models = models
                self.vision_combobox["values"] = models
                if self.vision_model.get() not in models:
                    self.vision_model.set(models[0])
                self.write_log_line(f"Ollama-Modelle geladen: {', '.join(models)}\n")
                self.status.config(text="Ollama-Modelle geladen")
            else:
                self.write_log_line("Keine Ollama-Modelle gefunden.\n")
                self.status.config(text="Ollama: keine Modelle")

        except Exception as e:
            self.write_log_line(f"Ollama-Modellabfrage fehlgeschlagen: {e}\n")
            self.status.config(text="Ollama Fehler")

    def validate_inputs(self):
        input_value = self.input_path.get().strip()
        if not input_value:
            messagebox.showerror("Fehler", "Bitte Video oder Ordner auswählen.")
            return False

        if self.input_mode.get() == "folder":
            if not os.path.isdir(input_value):
                messagebox.showerror("Fehler", f"Ordner nicht gefunden:\n{input_value}")
                return False
        else:
            if not os.path.isfile(input_value):
                messagebox.showerror("Fehler", f"Datei nicht gefunden:\n{input_value}")
                return False

        try:
            int(self.frame_interval.get().strip() or "10")
        except ValueError:
            messagebox.showerror("Fehler", "Frame Intervall muss eine Zahl sein.")
            return False

        try:
            int(self.max_vision_frames.get().strip() or "20")
        except ValueError:
            messagebox.showerror("Fehler", "Max Vision Frames muss eine Zahl sein.")
            return False

        return True

    def reset_progress(self):
        self.progress_value.set(0)
        self.progress_text.set("0 %")
        self.detail_text.set("Warte auf Start ...")

    def update_progress(self, percent, detail):
        try:
            percent = max(0, min(100, int(percent)))
        except Exception:
            percent = 0

        self.progress_value.set(percent)
        self.progress_text.set(f"{percent} %")
        self.detail_text.set(detail)

    def start_analysis(self):
        if not self.validate_inputs():
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "videoanalyse_emb.py")

        if not os.path.exists(script_path):
            messagebox.showerror("Fehler", f"Analyse-Script nicht gefunden:\n\n{script_path}")
            return

        cmd = [
            sys.executable,
            "-u",
            script_path,
            "--vision-model", self.vision_model.get().strip() or "minicpm-v",
            "--whisper-model", self.whisper_model.get().strip() or "base",
            "--frame-interval", self.frame_interval.get().strip() or "10",
            "--max-vision-frames", self.max_vision_frames.get().strip() or "20",
            "--ollama-url", self.ollama_url.get().strip() or "http://127.0.0.1:11434",
        ]

        if self.input_mode.get() == "folder":
            cmd.extend(["--input-folder", self.input_path.get().strip()])
        else:
            cmd.extend(["--input", self.input_path.get().strip()])

        if self.output_path.get().strip():
            cmd.extend(["--output", self.output_path.get().strip()])

        if self.use_vision.get():
            cmd.append("--use-vision")

        if self.write_nfo.get():
            cmd.append("--write-nfo")

        if self.update_emby.get():
            cmd.append("--update-emby")

        self.log.delete("1.0", tk.END)
        self.reset_progress()

        log_dir = self.output_path.get().strip()
        if not log_dir:
            if self.input_mode.get() == "folder":
                log_dir = self.input_path.get().strip()
            else:
                log_dir = os.path.dirname(os.path.abspath(self.input_path.get().strip()))

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_logfile = os.path.join(log_dir, f"videoanalyse_log_{timestamp}.txt")

        self.write_log_line(f"Starte Analyse:\n{script_path}\n\n")
        self.write_log_line(f"Modus: {'Bulk / Ordner' if self.input_mode.get() == 'folder' else 'Einzelvideo'}\n")
        self.write_log_line(f"Befehl:\n{' '.join(cmd)}\n\n")
        self.write_log_line(f"Logdatei: {self.current_logfile}\n\n")

        self.status.config(text="Analyse läuft ...")
        self.update_progress(0, "Analyse gestartet")

        thread = threading.Thread(target=self.run_process, args=(cmd,), daemon=True)
        thread.start()

    def run_process(self, cmd):
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            for line in self.process.stdout:
                self.log_queue.put(line)

            self.process.wait()

            if self.process.returncode == 0:
                self.log_queue.put("\nAnalyse abgeschlossen\n")
            else:
                self.log_queue.put(f"\nFehler im Analyse-Script (Code {self.process.returncode})\n")

        except Exception as e:
            self.log_queue.put(f"\nFehler beim Starten:\n{e}\n")

        finally:
            self.log_queue.put("__done__")

    def stop_analysis(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.write_log_line("\nAnalyse gestoppt\n")
            self.status.config(text="Gestoppt")
            self.update_progress(0, "Gestoppt")

    def process_log_queue(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()

                if msg == "__done__":
                    if self.status.cget("text") != "Gestoppt":
                        self.status.config(text="Fertig")
                        self.update_progress(100, "Fertig")
                else:
                    stripped = msg.strip()

                    if stripped.startswith("{") and '"type": "progress"' in stripped:
                        try:
                            data = json.loads(stripped)
                            percent = data.get("percent", 0)
                            message = data.get("message", "")
                            current_video = data.get("current_video", 0)
                            total_videos = data.get("total_videos", 0)

                            detail = message
                            if total_videos and current_video:
                                detail = f"Video {current_video}/{total_videos} - {message}"

                            self.update_progress(percent, detail)
                        except Exception:
                            self.write_log_line(msg)
                    else:
                        self.write_log_line(msg)

        except queue.Empty:
            pass

        self.root.after(100, self.process_log_queue)

    def write_log_line(self, text):
        self.log.insert(tk.END, text)
        self.log.see(tk.END)

        if self.current_logfile:
            try:
                with open(self.current_logfile, "a", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass

    def copy_log(self):
        text = self.log.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Hinweis", "Kein Log vorhanden.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        messagebox.showinfo("OK", "Kompletter Log wurde kopiert.")

    def copy_selected(self, event=None):
        try:
            text = self.log.get(tk.SEL_FIRST, tk.SEL_LAST)
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()
        except tk.TclError:
            pass
        return "break"

    def select_all(self, event=None):
        self.log.tag_add(tk.SEL, "1.0", tk.END)
        self.log.mark_set(tk.INSERT, "1.0")
        self.log.see(tk.INSERT)
        return "break"

    def save_log_as(self):
        text = self.log.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Hinweis", "Kein Log vorhanden.")
            return

        filename = filedialog.asksaveasfilename(
            title="Log speichern",
            defaultextension=".txt",
            filetypes=[("Textdatei", "*.txt"), ("Alle Dateien", "*.*")]
        )
        if filename:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(text)
            messagebox.showinfo("OK", f"Log gespeichert:\n{filename}")

    def clear_log(self):
        self.log.delete("1.0", tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnalyseGUI(root)
    root.mainloop()