"""Tkinter desktop client for the ITRI625 phishing e-mail detector.

1) Start the API first:   uvicorn api:app
2) Then run this app:     python gui.py
"""
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import requests

API_URL = "http://127.0.0.1:8000"

RED, GREEN, GREY = "#b00020", "#1b5e20", "#555555"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phishing E-mail Detector - ITRI625")
        self.geometry("860x680")
        self.minsize(720, 560)
        self._build_ui()
        self.after(200, self.check_api)

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        header = ttk.Frame(self)
        header.pack(fill="x", **pad)
        ttk.Label(header, text="Phishing E-mail Detector", font=("Segoe UI", 16, "bold")).pack(side="left")
        self.status = ttk.Label(header, text="API: checking...", foreground=GREY)
        self.status.pack(side="right")

        ttk.Label(self, text="Paste an e-mail (subject + body) below, or load a .txt / .eml file:").pack(
            anchor="w", padx=12
        )
        self.input = scrolledtext.ScrolledText(self, wrap="word", height=14, font=("Consolas", 10))
        self.input.pack(fill="both", expand=True, **pad)
        self.input.tag_configure("phish", background="#ffd6d6", foreground=RED)
        self.input.tag_configure("safe", background="#d9f2dd", foreground=GREEN)

        buttons = ttk.Frame(self)
        buttons.pack(fill="x", **pad)
        self.btn_load = ttk.Button(buttons, text="Load file...", command=self.load_file)
        self.btn_analyse = ttk.Button(buttons, text="Analyse", command=self.analyse)
        self.btn_explain = ttk.Button(buttons, text="Explain (highlight words)", command=self.explain)
        self.btn_clear = ttk.Button(buttons, text="Clear", command=self.clear)
        for b in (self.btn_load, self.btn_analyse, self.btn_explain, self.btn_clear):
            b.pack(side="left", padx=(0, 8))

        result = ttk.LabelFrame(self, text="Result")
        result.pack(fill="x", **pad)
        self.verdict = ttk.Label(result, text="-", font=("Segoe UI", 20, "bold"))
        self.verdict.pack(anchor="w", padx=12, pady=(8, 2))
        self.bar = ttk.Progressbar(result, maximum=100, length=400)
        self.bar.pack(fill="x", padx=12, pady=4)
        self.prob_label = ttk.Label(result, text="Phishing probability: -")
        self.prob_label.pack(anchor="w", padx=12)
        self.legend = ttk.Label(
            result,
            text="After 'Explain':  red = pushes towards PHISHING,  green = pushes towards LEGITIMATE",
            foreground=GREY,
        )
        self.legend.pack(anchor="w", padx=12, pady=(2, 8))

        self.busy = ttk.Label(self, text="", foreground=GREY)
        self.busy.pack(anchor="w", padx=12, pady=(0, 6))

    # ------------------------------------------------------------- helpers --
    def _set_busy(self, message=""):
        state = "disabled" if message else "normal"
        for b in (self.btn_load, self.btn_analyse, self.btn_explain, self.btn_clear):
            b.config(state=state)
        self.busy.config(text=message)

    def _get_text(self):
        text = self.input.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showinfo("Nothing to analyse", "Paste or load an e-mail first.")
            return None
        return text

    def _call_api(self, endpoint, text, timeout, on_success, busy_message):
        self._set_busy(busy_message)

        def worker():
            try:
                r = requests.post(f"{API_URL}/{endpoint}", json={"text": text}, timeout=timeout)
                r.raise_for_status()
                self.after(0, lambda: (self._set_busy(), on_success(r.json())))
            except Exception as exc:  # network / server errors
                self.after(0, lambda: (self._set_busy(), self._show_error(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _show_error(self, exc):
        self.status.config(text="API: offline", foreground=RED)
        messagebox.showerror(
            "Could not reach the API",
            f"{exc}\n\nIs the server running?\n    uvicorn api:app",
        )

    # ------------------------------------------------------------- actions --
    def check_api(self):
        def worker():
            try:
                ok = requests.get(f"{API_URL}/health", timeout=3).json().get("model_loaded")
            except Exception:
                ok = False
            text, colour = ("API: online", GREEN) if ok else ("API: offline", RED)
            self.after(0, lambda: self.status.config(text=text, foreground=colour))

        threading.Thread(target=worker, daemon=True).start()

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("E-mail / text", "*.txt *.eml"), ("All files", "*.*")])
        if path:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                self.input.delete("1.0", "end")
                self.input.insert("1.0", fh.read())

    def clear(self):
        self.input.delete("1.0", "end")
        self._clear_tags()
        self.verdict.config(text="-", foreground="black")
        self.bar["value"] = 0
        self.prob_label.config(text="Phishing probability: -")

    def analyse(self):
        text = self._get_text()
        if text:
            self._clear_tags()
            self._call_api("predict", text, 15, self._show_prediction, "Analysing...")

    def explain(self):
        text = self._get_text()
        if text:
            self._call_api("explain", text, 180, self._show_explanation, "Explaining (LIME)... this can take a few seconds")

    def _show_prediction(self, data):
        p = data["probability"]
        if data["is_phishing"]:
            self.verdict.config(text="PHISHING", foreground=RED)
        else:
            self.verdict.config(text="LEGITIMATE", foreground=GREEN)
        self.bar["value"] = p * 100
        self.prob_label.config(text=f"Phishing probability: {p * 100:.1f}%   (decision threshold {data['threshold']:.2f})")

    def _clear_tags(self):
        self.input.tag_remove("phish", "1.0", "end")
        self.input.tag_remove("safe", "1.0", "end")

    def _show_explanation(self, data):
        self._clear_tags()
        content = self.input.get("1.0", "end-1c")
        for feat in data["features"]:
            tag = "phish" if feat["weight"] > 0 else "safe"
            for m in re.finditer(rf"\b{re.escape(feat['word'])}\b", content, flags=re.IGNORECASE):
                self.input.tag_add(tag, f"1.0+{m.start()}c", f"1.0+{m.end()}c")


if __name__ == "__main__":
    App().mainloop()
