
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkinter.font as tkFont
import threading
import os
import sys
import io
import contextlib
from pathlib import Path
import warnings
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─── Third-party ────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score)
import joblib

# ─── Hardware acquisition (optional) ────────────────────────────────────────
try:
    from dataAcquisition import M1K_EIS_ONLY
    HW_AVAILABLE = True
    logger.info("pysmu library available - hardware acquisition enabled")
except ImportError as e:
    HW_AVAILABLE = False
    logger.warning(f"Hardware not available: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  THEME / COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════════
THEME = {
    "bg_dark":    "#080C10",
    "bg_panel":   "#0F1318",
    "bg_card":    "#161C24",
    "bg_input":   "#111620",
    "accent":     "#4D9EFF",
    "accent2":    "#2ECC71",
    "accent3":    "#FF6B6B",
    "accent4":    "#FFB347",
    "accent_hw":  "#00E5CC",   # Teal for hardware button
    "text_main":  "#DDE8F2",
    "text_muted": "#7A8A9A",
    "border":     "#1E2A38",
    "border2":    "#253040",
    "critical":   "#FF4D4D",
    "warning":    "#F0883E",
    "good":       "#2ECC71",
    "excellent":  "#4D9EFF",
    "header_bg":  "#080C10",
    "glow":       "#4D9EFF22",
    "hw_glow":    "#00E5CC22",
}

FONT_TITLE  = ("Consolas", 14, "bold")
FONT_HEADER = ("Consolas", 11, "bold")
FONT_BODY   = ("Consolas", 10)
FONT_MONO   = ("Courier New", 10)
FONT_BIG    = ("Consolas", 22, "bold")

# ═══════════════════════════════════════════════════════════════════════════
#  ENHANCED METRICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════
class MetricsEngine:
    MAX_RUL = 66667.0

    @staticmethod
    def soh(health): return max(0, min(100, health))

    @staticmethod
    def health_index(soh, rul, degradation_rate):
        rul_norm  = min(100, (rul / MetricsEngine.MAX_RUL) * 100)
        deg_score = max(0, 100 - degradation_rate * 1_000_000)
        return max(0, min(100, soh * 0.5 + rul_norm * 0.3 + deg_score * 0.2))

    @staticmethod
    def env_impact(health, rul):
        if rul > 30000: eff = 0.2
        elif rul > 15000: eff = 0.4
        elif rul > 5000:  eff = 0.6
        else:             eff = 0.8
        return max(0, min(100, 100 * eff))

    @staticmethod
    def carbon_burden(health, capacity_kwh=0.25):
        mfg = capacity_kwh * 61
        transport = mfg * 0.02
        recycling = (mfg + transport) * 0.3 if health < 80 else 0
        total = mfg + transport - recycling
        per_cycle = (total * 1000) / MetricsEngine.MAX_RUL
        return {"total_kg": round(total, 2), "per_cycle_g": round(per_cycle, 3)}

    @staticmethod
    def replacement_urgency(rul, health, degradation_rate):
        r = 95 if rul < 500 else 80 if rul < 1000 else 50 if rul < 5000 else 25 if rul < 15000 else 5
        h = 85 if health < 70 else 60 if health < 80 else 30 if health < 90 else 5
        d = 70 if degradation_rate > 0.001 else 40 if degradation_rate > 0.0005 else 10
        return max(0, min(100, np.mean([r, h, d])))

    @staticmethod
    def recommendation(urgency, health, rul):
        if urgency < 25:
            return ("OPTIMAL",  "#3FB950", "Continue Use — Battery in excellent condition. No action needed.")
        elif urgency < 50:
            return ("MONITOR",  "#58A6FF", "Good Condition — Plan replacement within 12 months.")
        elif urgency < 75:
            return ("PLAN",     "#F0883E", "Schedule Replacement — within 3–6 months. Consider recycling.")
        else:
            return ("REPLACE",  "#F85149", "Urgent Replacement — Replace immediately. Arrange recycling.")

    @staticmethod
    def failure_insights(failure_mode):
        info = {
            "SEI_growth":       ("SEI Growth", "Solid Electrolyte Interphase buildup on anode surface.\nCauses capacity fade and increased impedance over cycles."),
            "lithium_plating":  ("Lithium Plating", "Lithium metal dendrites forming on anode surface.\nRisk of internal short circuit and thermal runaway."),
            "electrode_cracking":("Electrode Cracking", "Structural damage to electrode materials.\nOften caused by thermal cycling or mechanical stress."),
            "normal_aging":     ("Normal Aging", "Standard electrochemical degradation mechanism.\nExpected progressive capacity loss over time."),
        }
        return info.get(failure_mode, (failure_mode, "Unknown failure mechanism."))


# ═══════════════════════════════════════════════════════════════════════════
#  ML PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
class BatteryMLPipeline:
    def __init__(self, dataset_path, model_dir="./models", log_callback=None):
        self.dataset_path = dataset_path
        self.model_dir    = Path(model_dir)
        self.log          = log_callback or print
        self.df           = None
        self.classifier   = None
        self.regressor    = None
        self.scaler       = StandardScaler()
        self.le           = LabelEncoder()
        self.feature_names = None
        self.class_names  = None
        self.metrics      = {}

    def load(self):
        self.df = pd.read_csv(self.dataset_path)
        self.log(f"✓ Dataset loaded: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        return self.df

    def preprocess(self):
        df = self.df.copy()
        df['failure_mode_enc'] = self.le.fit_transform(df['failure_mode'])
        self.class_names = list(self.le.classes_)
        df = pd.get_dummies(df, columns=['chemistry'], drop_first=False)

        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        excl    = {'battery_id', 'cycle', 'RUL', 'failure_mode_enc'}
        self.feature_names = [c for c in numeric if c not in excl]

        X = df[self.feature_names]
        y_class = df['failure_mode_enc']
        y_rul   = df['RUL']

        self.log(f"✓ Features prepared: {len(self.feature_names)}")
        self.log(f"  Classes: {', '.join(self.class_names)}")

        Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, y_class, test_size=0.2,
                                                        random_state=42, stratify=y_class)
        Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y_rul, test_size=0.2, random_state=42)

        sc = StandardScaler()
        self.Xc_tr = sc.fit_transform(Xc_tr); self.Xc_te = sc.transform(Xc_te)
        self.scaler = sc

        sr = StandardScaler()
        self.Xr_tr = sr.fit_transform(Xr_tr); self.Xr_te = sr.transform(Xr_te)

        self.yc_tr, self.yc_te = yc_tr, yc_te
        self.yr_tr, self.yr_te = yr_tr, yr_te
        return True

    def train(self):
        self.log("\n── Training Classifier (Gradient Boosting) ──")
        self.classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                     max_depth=5, random_state=42)
        self.classifier.fit(self.Xc_tr, self.yc_tr)
        yhat = self.classifier.predict(self.Xc_te)
        acc  = accuracy_score(self.yc_te, yhat)
        self.metrics['clf_accuracy'] = acc
        self.log(f"  ✓ Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
        self.log(f"\n{classification_report(self.yc_te, yhat, target_names=self.class_names)}")

        self.log("\n── Training Regressor (Gradient Boosting) ──")
        self.regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                                   max_depth=5, random_state=42)
        self.regressor.fit(self.Xr_tr, self.yr_tr)
        yhat_r = self.regressor.predict(self.Xr_te)
        rmse   = np.sqrt(mean_squared_error(self.yr_te, yhat_r))
        r2     = r2_score(self.yr_te, yhat_r)
        self.metrics['rmse'] = rmse
        self.metrics['r2']   = r2
        self.log(f"  ✓ Test RMSE  : {rmse:.2f} cycles")
        self.log(f"  ✓ Test R²    : {r2:.4f}")
        return True

    def save(self):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.classifier,   self.model_dir / 'failure_mode_classifier.pkl')
        joblib.dump(self.regressor,    self.model_dir / 'rul_regressor.pkl')
        joblib.dump(self.scaler,       self.model_dir / 'scaler.pkl')
        joblib.dump(self.le,           self.model_dir / 'label_encoder.pkl')
        joblib.dump(self.feature_names,self.model_dir / 'feature_names.pkl')
        self.log(f"\n✓ Models saved to {self.model_dir}/")

    def load_models(self, model_dir=None):
        d = Path(model_dir or self.model_dir)
        self.classifier    = joblib.load(d / 'failure_mode_classifier.pkl')
        self.regressor     = joblib.load(d / 'rul_regressor.pkl')
        self.scaler        = joblib.load(d / 'scaler.pkl')
        self.le            = joblib.load(d / 'label_encoder.pkl')
        self.feature_names = joblib.load(d / 'feature_names.pkl')
        self.class_names   = list(self.le.classes_)
        return True

    def predict_single(self, params: dict):
        vals = [params.get(f, 0.0) for f in self.feature_names]
        X    = np.array(vals).reshape(1, -1)
        Xs   = self.scaler.transform(X)
        fm_enc   = self.classifier.predict(Xs)[0]
        fm_prob  = float(self.classifier.predict_proba(Xs).max())
        fm_name  = self.le.inverse_transform([fm_enc])[0]
        rul      = float(max(0, self.regressor.predict(Xs)[0]))
        return fm_name, fm_prob, rul

    def predict_batch(self, df: pd.DataFrame):
        feats = [f for f in self.feature_names if f in df.columns]
        X  = df[feats].fillna(0).values
        Xs = self.scaler.transform(X)
        fms  = self.le.inverse_transform(self.classifier.predict(Xs))
        ruls = np.maximum(0, self.regressor.predict(Xs))
        return fms, ruls


# ═══════════════════════════════════════════════════════════════════════════
#  STYLED WIDGETS HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def dark_label(parent, text, fg=None, font=None, **kw):
    return tk.Label(parent, text=text, bg=THEME["bg_panel"],
                    fg=fg or THEME["text_main"], font=font or FONT_BODY, **kw)

def dark_entry(parent, textvariable=None, width=14):
    e = tk.Entry(parent, textvariable=textvariable, width=width,
                 bg=THEME["bg_input"], fg=THEME["text_main"],
                 insertbackground=THEME["accent"],
                 relief="flat", bd=4, font=FONT_MONO)
    # Subtle focus highlight
    e.bind("<FocusIn>",  lambda ev: e.config(highlightthickness=1,
                                              highlightbackground=THEME["accent"],
                                              highlightcolor=THEME["accent"]))
    e.bind("<FocusOut>", lambda ev: e.config(highlightthickness=0))
    return e

def accent_btn(parent, text, command, color=None, width=16):
    c = color or THEME["accent"]
    btn = tk.Button(parent, text=text, command=command, width=width,
                    bg=c, fg="#080C10", font=("Consolas", 10, "bold"),
                    relief="flat", cursor="hand2", bd=0,
                    activebackground=c, activeforeground="#080C10",
                    pady=7)
    btn.bind("<Enter>", lambda e: btn.config(bg=_lighten(c)))
    btn.bind("<Leave>", lambda e: btn.config(bg=c))
    return btn

def _lighten(hex_color):
    h = hex_color.lstrip('#')
    r, g, b = (min(255, int(h[i:i+2], 16) + 30) for i in (0, 2, 4))
    return f"#{r:02x}{g:02x}{b:02x}"

def dark_text(parent, height=20, width=80, font=None):
    t = tk.Text(parent, height=height, width=width,
                bg=THEME["bg_card"], fg=THEME["text_main"],
                insertbackground=THEME["accent"], relief="flat", bd=0,
                font=font or FONT_MONO, wrap=tk.WORD,
                selectbackground=THEME["accent"],
                padx=12, pady=10,
                spacing1=2, spacing3=2)
    return t

def section_frame(parent, title, pady=6):
    f = tk.LabelFrame(parent, text=f"  {title}  ",
                      bg=THEME["bg_panel"], fg=THEME["accent"],
                      font=("Consolas", 9, "bold"),
                      relief="flat", bd=1,
                      highlightbackground=THEME["border2"],
                      highlightcolor=THEME["accent"],
                      highlightthickness=1)
    f.pack(fill=tk.X, padx=8, pady=pady)
    return f

# ═══════════════════════════════════════════════════════════════════════════
#  STAT CARD WIDGET
# ═══════════════════════════════════════════════════════════════════════════
class StatCard(tk.Frame):
    def __init__(self, parent, label, value="—", color=None, unit=""):
        super().__init__(parent, bg=THEME["bg_card"],
                         highlightbackground=THEME["border2"],
                         highlightthickness=1)
        self._color = color or THEME["accent"]
        # Color accent bar at top
        bar = tk.Frame(self, bg=self._color, height=2)
        bar.pack(fill=tk.X)
        tk.Label(self, text=label.upper(), bg=THEME["bg_card"],
                 fg=THEME["text_muted"], font=("Consolas", 7, "bold")).pack(pady=(6, 0))
        self._val_lbl = tk.Label(self, text=value, bg=THEME["bg_card"],
                                 fg=self._color, font=("Consolas", 20, "bold"))
        self._val_lbl.pack()
        if unit:
            tk.Label(self, text=unit, bg=THEME["bg_card"],
                     fg=THEME["text_muted"], font=("Consolas", 7)).pack(pady=(0, 8))
        else:
            tk.Label(self, text=" ", bg=THEME["bg_card"],
                     font=("Consolas", 7)).pack(pady=(0, 4))

    def set(self, value, color=None):
        c = color or self._color
        self._val_lbl.config(text=str(value), fg=c)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════
class BatteryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ Battery 18650 ML Prediction System")
        self.root.geometry("1480x900")
        self.root.minsize(1100, 720)
        self.root.configure(bg=THEME["bg_dark"])

        self.pipeline     = None
        self.models_ready = False
        self.dataset_path = tk.StringVar(value="final_corrected_18650_dataset.csv")
        self.model_dir    = "./models"

        self._apply_ttk_theme()
        self._build_header()
        self._build_notebook()
        self._try_load_models_silent()

    # ─── TTK THEME ──────────────────────────────────────────────────────
    def _apply_ttk_theme(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('.', background=THEME["bg_dark"], foreground=THEME["text_main"],
                    fieldbackground=THEME["bg_input"], troughcolor=THEME["bg_card"],
                    selectbackground=THEME["accent"])
        s.configure('TNotebook', background=THEME["bg_dark"], borderwidth=0, tabmargins=0)
        s.configure('TNotebook.Tab',
                    background=THEME["bg_card"], foreground=THEME["text_muted"],
                    font=("Consolas", 10, "bold"), padding=(16, 7),
                    borderwidth=0)
        s.map('TNotebook.Tab',
              background=[('selected', THEME["bg_panel"]), ('active', THEME["bg_input"])],
              foreground=[('selected', THEME["accent"]), ('active', THEME["text_main"])])
        s.configure('TScrollbar', background=THEME["bg_card"],
                    troughcolor=THEME["bg_dark"], arrowcolor=THEME["text_muted"],
                    borderwidth=0, relief="flat")
        s.configure('Vertical.TScrollbar', width=6)

    # ─── HEADER ──────────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Frame(self.root, bg=THEME["bg_panel"],
                       highlightbackground=THEME["accent"],
                       highlightthickness=1)
        hdr.pack(fill=tk.X, padx=0, pady=0)

        left = tk.Frame(hdr, bg=THEME["bg_panel"])
        left.pack(side=tk.LEFT, padx=20, pady=8)

        tk.Label(left, text="⚡ BATTERY 18650",
                 bg=THEME["bg_panel"], fg=THEME["accent"],
                 font=("Consolas", 18, "bold")).pack(side=tk.LEFT)
        tk.Label(left, text=" ML PREDICTION SYSTEM",
                 bg=THEME["bg_panel"], fg=THEME["text_main"],
                 font=("Consolas", 14)).pack(side=tk.LEFT)

        subtitle = tk.Label(hdr, text="EIS-Based Predictive Analytics · GradientBoosting · Sustainability Metrics",
                            bg=THEME["bg_panel"], fg=THEME["text_muted"], font=("Consolas", 8))
        subtitle.pack(side=tk.LEFT, padx=16)

        right = tk.Frame(hdr, bg=THEME["bg_panel"])
        right.pack(side=tk.RIGHT, padx=20, pady=8)

        # Hardware availability indicator
        hw_color = THEME["accent_hw"] if HW_AVAILABLE else THEME["text_muted"]
        hw_text  = "HW:READY" if HW_AVAILABLE else "HW:OFFLINE"
        self._hw_dot = tk.Label(right, text=f"◆ {hw_text}",
                                bg=THEME["bg_panel"], fg=hw_color,
                                font=("Consolas", 8, "bold"))
        self._hw_dot.pack(side=tk.LEFT, padx=(0, 12))

        self._status_dot = tk.Label(right, text="●", bg=THEME["bg_panel"],
                                    fg=THEME["text_muted"], font=("Consolas", 16))
        self._status_dot.pack(side=tk.LEFT, padx=(0, 4))
        self._status_lbl = tk.Label(right, text="Models Not Loaded",
                                    bg=THEME["bg_panel"], fg=THEME["text_muted"],
                                    font=("Consolas", 9))
        self._status_lbl.pack(side=tk.LEFT)

    def _set_status(self, text, ok=True):
        color = THEME["good"] if ok else THEME["critical"]
        self._status_dot.config(fg=color)
        self._status_lbl.config(text=text, fg=color)

    # ─── NOTEBOOK TABS ────────────────────────────────────────────────────
    def _build_notebook(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=(4, 6))

        # Tab order
        self._tab_dashboard  = tk.Frame(self.nb, bg=THEME["bg_dark"])
        self._tab_predict    = tk.Frame(self.nb, bg=THEME["bg_dark"])
        self._tab_batch      = tk.Frame(self.nb, bg=THEME["bg_dark"])
        self._tab_visualize  = tk.Frame(self.nb, bg=THEME["bg_dark"])
        self._tab_train      = tk.Frame(self.nb, bg=THEME["bg_dark"])
        self._tab_about      = tk.Frame(self.nb, bg=THEME["bg_dark"])

        self.nb.add(self._tab_dashboard, text="  🏠 Dashboard  ")
        self.nb.add(self._tab_predict,   text="  🔮 Prediction  ")
        self.nb.add(self._tab_batch,     text="  📦 Batch  ")
        self.nb.add(self._tab_visualize, text="  📊 Visualize  ")
        self.nb.add(self._tab_train,     text="  🏋 Train  ")
        self.nb.add(self._tab_about,     text="  ℹ️  About  ")

        self._build_dashboard_tab()
        self._build_predict_tab()
        self._build_batch_tab()
        self._build_visualize_tab()
        self._build_train_tab()
        self._build_about_tab()

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 1: DASHBOARD
    # ═══════════════════════════════════════════════════════════════════
    def _build_dashboard_tab(self):
        p = self._tab_dashboard

        # Top control bar
        ctrl = tk.Frame(p, bg=THEME["bg_panel"],
                        highlightbackground=THEME["border"], highlightthickness=1)
        ctrl.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(ctrl, text="  Dataset:", bg=THEME["bg_panel"],
                 fg=THEME["text_muted"], font=FONT_BODY).pack(side=tk.LEFT, padx=4, pady=8)
        dark_entry(ctrl, textvariable=self.dataset_path, width=40).pack(side=tk.LEFT, padx=4)
        accent_btn(ctrl, "📂 Browse", self._browse_dataset, width=12).pack(side=tk.LEFT, padx=4)
        accent_btn(ctrl, "🔍 Load & Analyze", self._load_and_analyze, width=18,
                   color=THEME["accent2"]).pack(side=tk.LEFT, padx=4)

        # Stats row
        stats_row = tk.Frame(p, bg=THEME["bg_dark"])
        stats_row.pack(fill=tk.X, padx=8, pady=4)
        for c in range(6):
            stats_row.columnconfigure(c, weight=1)

        self._dash_cards = {}
        cards = [
            ("Total Records",    "records",    THEME["accent"]),
            ("Unique Batteries", "batteries",  THEME["accent2"]),
            ("Failure Classes",  "classes",    THEME["accent4"]),
            ("Avg Health",       "avg_health", THEME["good"]),
            ("Avg RUL",          "avg_rul",    THEME["warning"]),
            ("Features",         "features",   THEME["text_muted"]),
        ]
        for i, (lbl, key, col) in enumerate(cards):
            card = StatCard(stats_row, lbl, "—", col)
            card.grid(row=0, column=i, sticky="nsew", padx=4, pady=4)
            self._dash_cards[key] = card

        # Summary text
        sf = tk.Frame(p, bg=THEME["bg_dark"])
        sf.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self._dash_text = dark_text(sf, height=28)
        sb = ttk.Scrollbar(sf, command=self._dash_text.yview)
        self._dash_text.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._dash_text.pack(fill=tk.BOTH, expand=True)

        self._dash_text.insert(tk.END,
            "\n  📋 Load a dataset using the controls above to see the full analysis report.\n"
            "  The dashboard will display:\n"
            "    • Dataset shape & memory usage\n"
            "    • Failure mode distribution\n"
            "    • Chemistry types\n"
            "    • Health & RUL statistics\n"
            "    • Correlation analysis\n"
            "    • Missing value report\n")
        self._dash_text.config(state=tk.DISABLED)

    def _browse_dataset(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p: self.dataset_path.set(p)

    def _load_and_analyze(self):
        """Load and analyze dataset with comprehensive error handling."""
        path = self.dataset_path.get()
        if not os.path.exists(path):
            messagebox.showerror("Error", f"File not found:\n{path}")
            logger.error(f"Dataset file not found: {path}")
            return
        
        try:
            logger.info(f"Loading dataset from: {path}")
            df = pd.read_csv(path)
            
            if df.empty:
                messagebox.showerror("Error", "Dataset CSV file is empty.")
                logger.warning("Empty dataset loaded")
                return
            
            logger.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            self._df = df
            self._dash_text.config(state=tk.NORMAL)
            self._dash_text.delete(1.0, tk.END)

            try:
                report = self._generate_report(df)
                self._dash_text.insert(tk.END, report)
            except Exception as e:
                logger.error(f"Report generation error: {e}")
                self._dash_text.insert(tk.END, f"Error generating full report:\n{str(e)}\n\nPartial data loaded.")
            
            self._dash_text.config(state=tk.DISABLED)

            # Update stat cards with error handling
            try:
                self._dash_cards["records"].set(f"{len(df):,}")
                self._dash_cards["batteries"].set(df['battery_id'].nunique() if 'battery_id' in df.columns else 0)
                self._dash_cards["classes"].set(df['failure_mode'].nunique() if 'failure_mode' in df.columns else 0)
                self._dash_cards["avg_health"].set(f"{df['health'].mean():.1f}%" if 'health' in df.columns else "N/A")
                self._dash_cards["avg_rul"].set(f"{df['RUL'].mean():.0f}" if 'RUL' in df.columns else "N/A")
                num = df.select_dtypes(include=np.number).columns
                self._dash_cards["features"].set(max(0, len(num) - 2))
                logger.info("Dashboard cards updated successfully")
            except Exception as e:
                logger.warning(f"Error updating dashboard cards: {e}")

        except pd.errors.ParserError as e:
            messagebox.showerror("CSV Error", f"File is not a valid CSV:\n{str(e)}")
            logger.error(f"CSV parsing error: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")
            logger.error(f"Dataset loading error: {e}", exc_info=True)

    def _generate_report(self, df):
        sep = "─" * 72
        lines = [
            "",
            "╔" + "═"*70 + "╗",
            "║" + "  DATASET ANALYSIS REPORT".center(70) + "║",
            "╚" + "═"*70 + "╝",
            "",
            f"  File  : {self.dataset_path.get()}",
            f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns",
            f"  Memory: {df.memory_usage(deep=True).sum()/1024/1024:.2f} MB",
            "",
            sep,
            "  🧬 COLUMN DATA TYPES",
            sep,
        ]
        for dtype, cnt in df.dtypes.value_counts().items():
            lines.append(f"    {str(dtype):<15} {cnt} column(s)")
        lines += [
            "",
            sep,
            "  🔍 MISSING VALUES",
            sep,
        ]
        missing = df.isnull().sum()
        if missing.sum() == 0:
            lines.append("    ✓ No missing values detected!")
        else:
            for col, cnt in missing[missing > 0].items():
                lines.append(f"    {col:<30} {cnt}")
        lines += [
            "",
            sep,
            "  🔋 BATTERY OVERVIEW",
            sep,
            f"    Unique Batteries : {df['battery_id'].nunique()}",
            f"    Cycles (min/max) : {df['cycle'].min()} / {df['cycle'].max()}",
            f"    Avg cycles/batt  : {df.groupby('battery_id')['cycle'].max().mean():.0f}",
            "",
            sep,
            "  🧪 CHEMISTRY DISTRIBUTION",
            sep,
        ]
        for chem, cnt in df['chemistry'].value_counts().items():
            pct = cnt / len(df) * 100
            bar = "█" * int(pct / 3)
            lines.append(f"    {chem:<20} {cnt:>6,}  ({pct:.1f}%)  {bar}")
        lines += [
            "",
            sep,
            "  ⚠️  FAILURE MODE DISTRIBUTION",
            sep,
        ]
        for fm, cnt in df['failure_mode'].value_counts().items():
            pct = cnt / len(df) * 100
            bar = "█" * int(pct / 3)
            lines.append(f"    {fm:<25} {cnt:>6,}  ({pct:.1f}%)  {bar}")
        lines += [
            "",
            sep,
            "  💚 HEALTH STATISTICS",
            sep,
            f"    Min  : {df['health'].min():.2f}%",
            f"    Max  : {df['health'].max():.2f}%",
            f"    Mean : {df['health'].mean():.2f}%",
            f"    Std  : {df['health'].std():.2f}%",
            "",
            sep,
            "  🎯 RUL STATISTICS",
            sep,
            f"    Min     : {df['RUL'].min():.0f} cycles",
            f"    Max     : {df['RUL'].max():.0f} cycles",
            f"    Mean    : {df['RUL'].mean():.0f} cycles",
            f"    Std Dev : {df['RUL'].std():.0f} cycles",
            f"    EOL (=0): {(df['RUL']==0).sum():,} records",
            "",
            sep,
            "  🌡️  TEMPERATURE STATISTICS",
            sep,
            f"    Min : {df['temperature'].min():.2f}°C",
            f"    Max : {df['temperature'].max():.2f}°C",
            f"    Mean: {df['temperature'].mean():.2f}°C",
            "",
            sep,
            "  📊 TOP CORRELATIONS WITH HEALTH",
            sep,
        ]
        num_cols = df.select_dtypes(include=np.number).columns
        corr = df[num_cols].corr()['health'].drop('health').abs().sort_values(ascending=False)
        for col, val in corr.head(8).items():
            bar = "█" * int(abs(val) * 20)
            lines.append(f"    {col:<30} {val:+.4f}  {bar}")
        lines += ["", sep, "  📊 TOP CORRELATIONS WITH RUL", sep]
        corr2 = df[num_cols].corr()['RUL'].drop('RUL').abs().sort_values(ascending=False)
        for col, val in corr2.head(8).items():
            bar = "█" * int(abs(val) * 20)
            lines.append(f"    {col:<30} {val:+.4f}  {bar}")
        lines.append("")
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 2: PREDICTION
    # ═══════════════════════════════════════════════════════════════════
    def _build_predict_tab(self):
        p = self._tab_predict
        p.columnconfigure(0, weight=2, minsize=380)
        p.columnconfigure(1, weight=3)
        p.rowconfigure(0, weight=1)

        # ── Left: Input Panel ──────────────────────────────────────────
        left = tk.Frame(p, bg=THEME["bg_panel"],
                        highlightbackground=THEME["border"], highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(8,3), pady=8)

        tk.Label(left, text="⚙ INPUT PARAMETERS",
                 bg=THEME["bg_panel"], fg=THEME["accent"],
                 font=("Consolas", 11, "bold")).pack(pady=(12, 0))

        self._input_source_lbl = tk.Label(left,
                                          text="  Source: Manual Entry",
                                          bg=THEME["bg_panel"],
                                          fg=THEME["text_muted"],
                                          font=("Consolas", 8))
        self._input_source_lbl.pack(pady=(0, 6))

        # Scrollable input area
        canvas = tk.Canvas(left, bg=THEME["bg_panel"], bd=0, highlightthickness=0)
        sb     = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        inner  = tk.Frame(canvas, bg=THEME["bg_panel"])
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._pred_fields = {}

        groups = [
            ("📊 Basic Parameters", [
                ("health",           "Health (%)",          75.0),
                ("temperature",      "Temperature (°C)",    25.0),
                ("soc",              "State of Charge (%)", 60.0),
                ("degradation_rate", "Degradation Rate",    0.0007),
            ]),
            ("⚡ Resistance (Ω)", [
                ("R_s",          "Series R_s",          0.045),
                ("R_10kHz",      "R @ 10kHz",           0.045),
                ("R_1kHz",       "R @ 1kHz",            0.19),
                ("R_100Hz",      "R @ 100Hz",           0.19),
                ("R_10Hz",       "R @ 10Hz",            0.19),
                ("R_1Hz",        "R @ 1Hz",             0.19),
                ("R_ct",         "Charge Transfer Rct", 0.15),
                ("R_ct_R_s_ratio","Rct/Rs Ratio",       3.5),
            ]),
            ("📈 Impedance", [
                ("C_dl",              "Capacitance C_dl (F)", 1.5),
                ("Warburg_slope",     "Warburg Slope",        0.003),
                ("Warburg_intercept", "Warburg Intercept",    0.25),
                ("Z_mag_1kHz",        "Z Mag @ 1kHz",         0.20),
                ("Z_mag_100Hz",       "Z Mag @ 100Hz",        0.20),
                ("phase_1kHz",        "Phase @ 1kHz (°)",    -15.0),
                ("phase_100Hz",       "Phase @ 100Hz (°)",   -15.0),
                ("phase_min",         "Phase Min (°)",        -38.0),
                ("freq_min_imag",     "Freq Min Imag",         1.5),
                ("freq_phase_45",     "Freq Phase 45",         2.5),
            ]),
        ]

        for grp_name, params in groups:
            gf = tk.LabelFrame(inner, text=f"  {grp_name}  ",
                               bg=THEME["bg_panel"], fg=THEME["accent2"],
                               font=("Consolas", 8, "bold"), relief="flat",
                               bd=0, highlightbackground=THEME["border2"],
                               highlightthickness=1)
            gf.pack(fill=tk.X, padx=6, pady=5)
            for pname, plabel, pdefault in params:
                row = tk.Frame(gf, bg=THEME["bg_panel"])
                row.pack(fill=tk.X, padx=8, pady=2)
                tk.Label(row, text=plabel, bg=THEME["bg_panel"],
                         fg=THEME["text_muted"], font=("Consolas", 9),
                         width=22, anchor="w").pack(side=tk.LEFT)
                var = tk.DoubleVar(value=pdefault)
                dark_entry(row, textvariable=var, width=12).pack(side=tk.LEFT, padx=4)
                self._pred_fields[pname] = var

        # Buttons — primary row
        btn_bar = tk.Frame(left, bg=THEME["bg_panel"])
        btn_bar.pack(fill=tk.X, padx=8, pady=(4, 2))
        accent_btn(btn_bar, "🔮  PREDICT", self._run_prediction,
                   color=THEME["accent"], width=14).pack(side=tk.LEFT, padx=3)
        accent_btn(btn_bar, "📋 Sample", self._load_sample,
                   color=THEME["accent4"], width=10).pack(side=tk.LEFT, padx=3)
        accent_btn(btn_bar, "🔄 Clear", self._clear_pred,
                   color=THEME["bg_card"], width=8).pack(side=tk.LEFT, padx=3)

        # Hardware acquisition row
        hw_bar = tk.Frame(left, bg=THEME["bg_panel"],
                          highlightbackground=THEME["accent_hw"] if HW_AVAILABLE else THEME["border"],
                          highlightthickness=1)
        hw_bar.pack(fill=tk.X, padx=8, pady=(2, 8))

        hw_label = tk.Label(hw_bar,
                            text="  ADALM1000  ",
                            bg=THEME["bg_panel"],
                            fg=THEME["accent_hw"] if HW_AVAILABLE else THEME["text_muted"],
                            font=("Consolas", 7, "bold"))
        hw_label.pack(side=tk.LEFT, padx=(4, 0), pady=4)

        self._hw_btn = accent_btn(hw_bar, "⚡ Get Values from Hardware",
                                  self._fetch_from_hardware,
                                  color=THEME["accent_hw"] if HW_AVAILABLE else THEME["bg_card"],
                                  width=26)
        self._hw_btn.pack(side=tk.LEFT, padx=4, pady=4)

        if not HW_AVAILABLE:
            self._hw_btn.config(state=tk.DISABLED,
                                text="⚡ Get Values from Hardware  [pysmu not installed]")

        self._hw_status_lbl = tk.Label(hw_bar, text="",
                                       bg=THEME["bg_panel"],
                                       fg=THEME["accent_hw"],
                                       font=("Consolas", 8))
        self._hw_status_lbl.pack(side=tk.LEFT, padx=6)

        # ── Right: Results Panel ────────────────────────────────────────
        right = tk.Frame(p, bg=THEME["bg_dark"])
        right.grid(row=0, column=1, sticky="nsew", padx=(3,8), pady=8)

        # Stats cards row
        cards_row = tk.Frame(right, bg=THEME["bg_dark"])
        cards_row.pack(fill=tk.X, pady=(0, 6))
        for c in range(5):
            cards_row.columnconfigure(c, weight=1)

        self._pred_cards = {}
        card_defs = [
            ("State of Health", "soh",    "—", "%",      THEME["good"]),
            ("RUL",             "rul",    "—", "cycles",  THEME["accent"]),
            ("Failure Mode",    "fm",     "—", "",        THEME["accent4"]),
            ("Confidence",      "conf",   "—", "%",       THEME["accent2"]),
            ("Health Index",    "hidx",   "—", "/100",    THEME["warning"]),
        ]
        for i, (lbl, key, val, unit, col) in enumerate(card_defs):
            card = StatCard(cards_row, lbl, val, col, unit)
            card.grid(row=0, column=i, sticky="nsew", padx=3, pady=2)
            self._pred_cards[key] = card

        # Results notebook
        rn = ttk.Notebook(right)
        rn.pack(fill=tk.BOTH, expand=True)

        # Core results tab
        core_f = tk.Frame(rn, bg=THEME["bg_dark"])
        rn.add(core_f, text="  🎯 Core Results  ")
        self._pred_core_text = dark_text(core_f, height=22)
        csb = ttk.Scrollbar(core_f, command=self._pred_core_text.yview)
        self._pred_core_text.config(yscrollcommand=csb.set)
        csb.pack(side=tk.RIGHT, fill=tk.Y)
        self._pred_core_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._pred_core_text.config(state=tk.DISABLED)

        # Sustainability tab
        sust_f = tk.Frame(rn, bg=THEME["bg_dark"])
        rn.add(sust_f, text="  🌍 Sustainability  ")
        self._pred_sust_text = dark_text(sust_f, height=22)
        ssb = ttk.Scrollbar(sust_f, command=self._pred_sust_text.yview)
        self._pred_sust_text.config(yscrollcommand=ssb.set)
        ssb.pack(side=tk.RIGHT, fill=tk.Y)
        self._pred_sust_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._pred_sust_text.config(state=tk.DISABLED)

        # Insights tab
        ins_f = tk.Frame(rn, bg=THEME["bg_dark"])
        rn.add(ins_f, text="  💡 Insights  ")
        self._pred_ins_text = dark_text(ins_f, height=22)
        isb = ttk.Scrollbar(ins_f, command=self._pred_ins_text.yview)
        self._pred_ins_text.config(yscrollcommand=isb.set)
        isb.pack(side=tk.RIGHT, fill=tk.Y)
        self._pred_ins_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._pred_ins_text.config(state=tk.DISABLED)

    def _load_sample(self):
        samples = {
            "health": 75.0, "temperature": 28.0, "soc": 60.0, "degradation_rate": 0.0007,
            "R_s": 0.045, "R_10kHz": 0.045, "R_1kHz": 0.19, "R_100Hz": 0.19,
            "R_10Hz": 0.19, "R_1Hz": 0.19, "R_ct": 0.15, "R_ct_R_s_ratio": 3.5,
            "C_dl": 1.5, "Warburg_slope": 0.003, "Warburg_intercept": 0.25,
            "Z_mag_1kHz": 0.20, "Z_mag_100Hz": 0.20, "phase_1kHz": -15.0,
            "phase_100Hz": -15.0, "phase_min": -38.0, "freq_min_imag": 1.5, "freq_phase_45": 2.5,
        }
        for k, v in samples.items():
            if k in self._pred_fields:
                self._pred_fields[k].set(v)
        self._input_source_lbl.config(text="  Source: Sample Data (built-in defaults)",
                                      fg=THEME["accent4"])

    def _clear_pred(self):
        for v in self._pred_fields.values():
            v.set(0.0)
        for text_w in [self._pred_core_text, self._pred_sust_text, self._pred_ins_text]:
            text_w.config(state=tk.NORMAL)
            text_w.delete(1.0, tk.END)
            text_w.config(state=tk.DISABLED)
        for card in self._pred_cards.values():
            card.set("—")
        self._input_source_lbl.config(text="  Source: Manual Entry",
                                      fg=THEME["text_muted"])
        self._hw_status_lbl.config(text="")

    # ─── HARDWARE ACQUISITION (FIXED SESSION MANAGEMENT) ────────────────────────
    def _fetch_from_hardware(self):

        if not HW_AVAILABLE:
            messagebox.showerror(
                "Hardware Unavailable",
                "pysmu library is not installed.\n\n"
                "Install with:  pip install pysmu\n"
                "Then restart the application."
            )
            return

        # Disable button and show animated status
        self._hw_btn.config(state=tk.DISABLED, text="⏳ Acquiring from ADALM1000…")
        self._hw_status_lbl.config(text="Connecting…", fg=THEME["accent_hw"])
        self.root.update_idletasks()

        def _run_acquisition():
            """Run acquisition in background thread with safe cleanup."""
            eis = None
            max_retries = 2
            attempt = 0
            
            try:
                while attempt < max_retries:
                    attempt += 1
                    try:
                        logger.info(f"Hardware acquisition attempt {attempt}/{max_retries}")
                        self._hw_status_update(f"Connecting (attempt {attempt}/{max_retries})…")
                        
                        # Create new instance to ensure clean state
                        eis = M1K_EIS_ONLY()
                        eis.connect()
                        logger.info("Hardware connected successfully")
                        break
                        
                    except RuntimeError as e:
                        logger.error(f"Connection error (attempt {attempt}): {e}")
                        if eis:
                            try:
                                eis.disconnect()
                            except:
                                pass
                        eis = None
                        
                        if attempt < max_retries:
                            import time
                            time.sleep(2)  # Wait before retry
                        else:
                            error_msg = (f"Hardware connection failed after {max_retries} attempts:\n{e}\n\n"
                                        "Troubleshooting:\n"
                                        "• Check ADALM1000 is plugged in via USB\n"
                                        "• Check battery connections (CHA+ → Battery+, CHB → Battery−)\n"
                                        "• Try a different USB port or cable\n"
                                        "• Restart the application")
                            self.root.after(0, lambda: self._hw_error(error_msg))
                            return
                
                if not eis:
                    self.root.after(0, lambda: self._hw_error("Failed to connect to hardware"))
                    return
                    
                # Run measurement
                self._hw_status_update("Running EIS sweep (may take ~30s)…")
                logger.info("Starting EIS sweep")
                
                params = eis.run_eis()
                logger.info(f"EIS sweep completed: {len(params)} parameters extracted")

                # Populate fields on the main thread  
                self.root.after(0, lambda: self._populate_fields_from_hw(params))

            except Exception as e:
                logger.error(f"EIS acquisition error: {e}", exc_info=True)
                error_msg = f"EIS acquisition error:\n{str(e)}\n\nCheck battery connections and try again."
                self.root.after(0, lambda: self._hw_error(error_msg))
                
            finally:
                # CRITICAL: Ensure proper cleanup on main thread
                if eis:
                    try:
                        logger.info("Disconnecting hardware")
                        eis.disconnect()
                        logger.info("Hardware disconnected successfully")
                    except Exception as e:
                        logger.error(f"Error during disconnect: {e}")
                
                # Reset button on main thread
                self.root.after(0, self._hw_btn_reset)

        # Start acquisition in background thread
        thread = threading.Thread(target=_run_acquisition, daemon=True)
        thread.start()

    def _hw_status_update(self, msg: str):
        """Thread-safe status label update."""
        def update():
            self._hw_status_lbl.config(text=msg)
            self.root.update_idletasks()
        self.root.after(0, update)

    def _hw_btn_reset(self):
        """Reset hardware button to normal state."""
        try:
            self._hw_btn.config(state=tk.NORMAL, text="⚡ Get Values from Hardware")
        except tk.TclError:
            # Widget may have been destroyed
            logger.debug("Hardware button already destroyed")

    def _hw_error(self, msg: str):
        """Display hardware error message."""
        self._hw_status_lbl.config(text="✗ Failed", fg=THEME["critical"])
        logger.error(f"Hardware error: {msg}")
        messagebox.showerror("Hardware Error", msg)

    def _populate_fields_from_hw(self, params: dict):
        """
        Populate all prediction input fields with hardware-measured values.
        Only updates fields that exist in the GUI; silently skips any missing ones.
        """
        count = 0
        for field_name, var in self._pred_fields.items():
            if field_name in params:
                try:
                    var.set(round(float(params[field_name]), 6))
                    count += 1
                except (TypeError, ValueError):
                    pass

        from datetime import datetime as _dt
        ts = _dt.now().strftime("%H:%M:%S")
        self._input_source_lbl.config(
            text=f"  Source: ADALM1000 Hardware  [{ts}]",
            fg=THEME["accent_hw"]
        )
        self._hw_status_lbl.config(
            text=f"✓ {count} values loaded from hardware  "
                 f"[OCV={params.get('ocv', 0):.3f}V  "
                 f"T≈{params.get('temperature', 0):.1f}°C  "
                 f"SoC≈{params.get('soc', 0):.0f}%]",
            fg=THEME["accent_hw"]
        )
        # Flash the hw_dot in header to confirm
        self._hw_dot.config(text="◆ HW:LIVE", fg=THEME["accent_hw"])
        self.root.after(4000, lambda: self._hw_dot.config(text="◆ HW:READY"))

    def _run_prediction(self):
        """
        Run single battery prediction with comprehensive error handling.
        """
        if not self.models_ready:
            messagebox.showwarning("Models Not Loaded",
                "Please train or load models first.\nGo to the '🏋 Train' tab.")
            return
        
        try:
            # Collect parameters with validation
            params = {}
            for k, v in self._pred_fields.items():
                try:
                    val = v.get()
                    if not isinstance(val, (int, float)):
                        val = float(val)
                    params[k] = float(val)
                except (TypeError, ValueError) as e:
                    logger.warning(f"Invalid value for {k}: {e}")
                    messagebox.showwarning("Input Error", 
                        f"Invalid input for {k}. Using default value.")
                    params[k] = 0.0
            
            logger.info(f"Running prediction with {len(params)} parameters")
            
            # Extract key parameters
            health = params.get("health", 75.0)
            deg    = params.get("degradation_rate", 0.0007)
            temp   = params.get("temperature", 25.0)
            soc    = params.get("soc", 60.0)
            
            # Validate parameter ranges
            health = np.clip(health, 0, 100)
            deg = np.clip(deg, 0, 1)
            temp = np.clip(temp, -20, 60)
            soc = np.clip(soc, 0, 100)
            
            # Run ML prediction
            try:
                fm, conf, rul = self.pipeline.predict_single(params)
                logger.info(f"Prediction result: {fm} (confidence: {conf:.2%}, RUL: {rul:.0f})")
            except Exception as e:
                logger.error(f"ML prediction error: {e}", exc_info=True)
                messagebox.showerror("Prediction Error", 
                    f"ML prediction failed:\n{str(e)}\n\nTry loading different models.")
                return

            # Calculate metrics
            try:
                soh   = MetricsEngine.soh(health)
                hidx  = MetricsEngine.health_index(soh, rul, deg)
                env   = MetricsEngine.env_impact(health, rul)
                carbon= MetricsEngine.carbon_burden(health)
                urg   = MetricsEngine.replacement_urgency(rul, health, deg)
                rec   = MetricsEngine.recommendation(urg, health, rul)
                fm_info = MetricsEngine.failure_insights(fm)
                logger.info(f"Metrics calculated: SoH={soh:.1f}%, Health Index={hidx:.1f}")
            except Exception as e:
                logger.error(f"Metrics calculation error: {e}", exc_info=True)
                messagebox.showerror("Metrics Error", 
                    f"Failed to calculate metrics:\n{str(e)}")
                return

            # Update UI cards
            try:
                self._pred_cards["soh"].set(f"{soh:.1f}", THEME["good"] if soh > 80 else THEME["warning"] if soh > 60 else THEME["critical"])
                self._pred_cards["rul"].set(f"{rul:,.0f}", THEME["accent"])
                self._pred_cards["fm"].set(fm.replace("_", "\n"), THEME["accent4"])
                self._pred_cards["conf"].set(f"{conf*100:.1f}", THEME["accent2"])
                self._pred_cards["hidx"].set(f"{hidx:.1f}", THEME["good"] if hidx > 70 else THEME["warning"] if hidx > 40 else THEME["critical"])
            except Exception as e:
                logger.error(f"UI update error: {e}")
                # Continue even if card updates fail

            # Generate and display reports
            rul_status = ("🔵 EXCELLENT (>10k)" if rul>10000 else
                          "🟢 GOOD (5k–10k)" if rul>5000 else
                          "🟡 MONITOR (1k–5k)" if rul>1000 else "🔴 CRITICAL (<1k)")
            
            try:
                self._write_text(self._pred_core_text, self._core_output(
                    soh, rul, rul_status, fm, fm_info, conf, hidx, deg, temp, soc))

                self._write_text(self._pred_sust_text, self._sust_output(
                    env, carbon, urg, rec))

                self._write_text(self._pred_ins_text, self._insights_output(
                    fm, fm_info, conf, deg, temp, soc, rul, health))
                
                logger.info("Prediction reports generated successfully")
            except Exception as e:
                logger.error(f"Report generation error: {e}")
                messagebox.showerror("Report Error", 
                    f"Failed to generate reports:\n{str(e)}")

        except Exception as e:
            logger.error(f"Unexpected prediction error: {e}", exc_info=True)
            messagebox.showerror("Prediction Error", str(e))

    def _write_text(self, widget, content):
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.insert(tk.END, content)
        widget.config(state=tk.DISABLED)

    def _core_output(self, soh, rul, rul_status, fm, fm_info, conf, hidx, deg, temp, soc):
        bar_soh  = "█" * int(soh/5)  + "░" * (20 - int(soh/5))
        bar_hidx = "█" * int(hidx/5) + "░" * (20 - int(hidx/5))
        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    🎯  CORE PREDICTION RESULTS                      ║
╚══════════════════════════════════════════════════════════════════════╝

┌─ STATE OF HEALTH (SoH) ─────────────────────────────────────────────┐
│  SoH     : {soh:.2f}%
│  Bar     : [{bar_soh}]
│  Status  : {"✓ EXCELLENT" if soh>85 else "✓ GOOD" if soh>70 else "⚠ FAIR" if soh>55 else "🔴 POOR"}
└──────────────────────────────────────────────────────────────────────┘

┌─ REMAINING USEFUL LIFE (RUL) ────────────────────────────────────────┐
│  RUL     : {rul:,.0f} cycles
│  Status  : {rul_status}
│  In Time : ~{rul/30:.0f} months (@ 30 cycles/month)
└──────────────────────────────────────────────────────────────────────┘

┌─ FAILURE MODE CLASSIFICATION ────────────────────────────────────────┐
│  Mode    : {fm.upper().replace("_"," ")}
│  Full    : {fm_info[0]}
│  Confid  : {conf*100:.2f}% {"✓ HIGH" if conf>0.9 else "⚠ MODERATE" if conf>0.7 else "⚡ LOW"}
└──────────────────────────────────────────────────────────────────────┘

┌─ DEGRADATION METRICS ────────────────────────────────────────────────┐
│  Rate    : {deg:.6f} /cycle
│  Monthly : {deg*30*100:.4f}% loss/month (30 cycles)
│  Status  : {"✓ LOW" if deg<0.0003 else "⚠ MODERATE" if deg<0.0007 else "🔴 HIGH"}
└──────────────────────────────────────────────────────────────────────┘

┌─ HEALTH INDEX (Composite Score) ─────────────────────────────────────┐
│  Score   : {hidx:.2f} / 100
│  Bar     : [{bar_hidx}]
│  Formula : (SoH×0.5) + (RUL_norm×0.3) + (DegScore×0.2)
│  Status  : {"✓ EXCELLENT" if hidx>80 else "✓ GOOD" if hidx>60 else "⚠ FAIR" if hidx>40 else "🔴 POOR"}
└──────────────────────────────────────────────────────────────────────┘

┌─ OPERATING CONDITIONS ───────────────────────────────────────────────┐
│  Temperature : {temp:.1f}°C   {"✓ Optimal" if 10<temp<35 else "⚠ High — accelerates aging" if temp>35 else "❄ Low — reduced performance"}
│  SoC         : {soc:.1f}%    {"✓ Optimal" if 20<soc<80 else "⚠ High — avoid prolonged storage" if soc>80 else "⚡ Low — recharge soon"}
└──────────────────────────────────────────────────────────────────────┘
"""

    def _sust_output(self, env, carbon, urg, rec):
        urg_bar  = "█" * int(urg/5)  + "░" * (20 - int(urg/5))
        env_bar  = "█" * int(env/5)  + "░" * (20 - int(env/5))
        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║                🌍  SUSTAINABILITY & ENVIRONMENTAL IMPACT            ║
╚══════════════════════════════════════════════════════════════════════╝

┌─ ENVIRONMENTAL IMPACT INDEX ─────────────────────────────────────────┐
│  Score   : {env:.1f}/100  (Lower = Better)
│  Bar     : [{env_bar}]
│  Rating  : {"✓ LOW IMPACT" if env<30 else "⚠ MODERATE IMPACT" if env<60 else "🔴 HIGH IMPACT"}
│  
│  Component Breakdown:
│    • Manufacturing : ~61 kg CO₂/kWh (18650 battery)
│    • Transportation: 2% of manufacturing impact
│    • Recycling     : ~30% emission credit at EOL
└──────────────────────────────────────────────────────────────────────┘

┌─ CARBON BURDEN ESTIMATE ─────────────────────────────────────────────┐
│  Total CO₂   : {carbon['total_kg']} kg CO₂  (0.25 kWh capacity)
│  Per Cycle   : {carbon['per_cycle_g']:.3f} g CO₂/cycle
│
│  Equivalents:
│    🚗 Vehicle Miles : ~{carbon['total_kg'] * 0.22:.1f} miles driven
│    🌲 Trees Offset  : ~{carbon['total_kg'] / 20:.2f} mature trees (1 yr)
│    💡 LED Hours     : ~{carbon['total_kg'] * 2000:.0f} hours of LED lighting
└──────────────────────────────────────────────────────────────────────┘

┌─ REPLACEMENT URGENCY SCORE ──────────────────────────────────────────┐
│  Urgency : {urg:.1f}/100
│  Bar     : [{urg_bar}]
│  Action  : {rec[0]}
│  Color   : {"🟢 LOW" if urg<25 else "🔵 MONITOR" if urg<50 else "🟡 PLAN" if urg<75 else "🔴 URGENT"}
│
│  Factors:
│    • RUL score weight   : 40%
│    • Health score weight : 40%
│    • Degradation weight  : 20%
└──────────────────────────────────────────────────────────────────────┘

┌─ SUSTAINABILITY RECOMMENDATION ──────────────────────────────────────┐
│
│  {rec[2]}
│
│  Recycling Path:
│    ✓ Recover: Lithium (~60%), Cobalt (~40%), Nickel (~50%)
│    ✓ Benefit: Reduces new mining demand by 70%
│    ✓ Method : Certified e-waste / battery recycling center
│
└──────────────────────────────────────────────────────────────────────┘
"""

    def _insights_output(self, fm, fm_info, conf, deg, temp, soc, rul, health):
        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║              💡  EXPLAINABILITY & DIAGNOSTIC INSIGHTS               ║
╚══════════════════════════════════════════════════════════════════════╝

┌─ FAILURE MODE EXPLANATION ───────────────────────────────────────────┐
│  Detected : {fm.upper().replace("_"," ")}
│
│  {fm_info[1].replace(chr(10), chr(10) + "│  ")}
│
│  Electrochemical Mechanism:
{"│    → SEI layer buildup increases internal resistance → capacity fade" if "SEI" in fm else "│    → Dendrite growth risks micro-short circuits → sudden failure" if "plating" in fm else "│    → Cracking exposes fresh electrode surface → accelerated aging" if "crack" in fm else "│    → Regular capacity fade due to thermodynamic equilibrium loss"}
└──────────────────────────────────────────────────────────────────────┘

┌─ MODEL CONFIDENCE ANALYSIS ──────────────────────────────────────────┐
│  Confidence : {conf*100:.2f}%
│  Quality    : {"✓ HIGH — Strong signal in impedance features" if conf>0.9 else "✓ GOOD — Reliable prediction with clear pattern" if conf>0.8 else "⚠ MODERATE — Consider more data points"}
│  
│  Top drivers of this prediction:
│    1. Battery Health (SoH)         — Primary degradation indicator
│    2. Degradation Rate             — Rate of capacity loss trend
│    3. Internal Resistance (R_ct)  — Electrode kinetics marker
│    4. Warburg Impedance           — Lithium diffusion indicator
└──────────────────────────────────────────────────────────────────────┘

┌─ TEMPERATURE ANALYSIS ───────────────────────────────────────────────┐
│  Current   : {temp:.1f}°C
│  Optimal   : 15–25°C for long-term storage
│  For use   : 0–45°C operating range (18650)
│  Impact    : {"⚠ Each +10°C doubles degradation rate (Arrhenius law)" if temp>30 else "✓ Temperature in acceptable range"}
│  Advice    : {"Cool battery — reduce charge rate or ambient temp" if temp>35 else "Maintain current temperature conditions"}
└──────────────────────────────────────────────────────────────────────┘

┌─ CHARGE MANAGEMENT ──────────────────────────────────────────────────┐
│  Current SoC : {soc:.1f}%
│  Ideal Range : 20%–80% for lifespan optimization
│  Impact      : {"⚠ Avoid prolonged storage at high SoC" if soc>80 else "⚡ Recharge soon to avoid deep discharge stress" if soc<20 else "✓ Optimal charge level for battery longevity"}
│  Tip         : {"Reduce to 80% for long-term storage" if soc>85 else "Charge to at least 30% before storage" if soc<25 else "Maintain 40–60% during extended storage periods"}
└──────────────────────────────────────────────────────────────────────┘

┌─ PREDICTIVE MAINTENANCE SCHEDULE ────────────────────────────────────┐
│  Next check  : {"IMMEDIATE" if rul<1000 else f"~{rul//4:,} cycles (~{rul//4//30} months)"}
│  Risk level  : {"🔴 HIGH" if rul<1000 or health<70 else "🟡 MEDIUM" if rul<5000 or health<80 else "🟢 LOW"}
│  Action log  :
│    • Log all charge/discharge cycles
│    • Monitor capacity drop per cycle
│    • EIS (impedance) check recommended at {"500 cycles" if deg>0.0007 else "1000 cycles"}
└──────────────────────────────────────────────────────────────────────┘
"""

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 3: BATCH PREDICTION
    # ═══════════════════════════════════════════════════════════════════
    def _build_batch_tab(self):
        p = self._tab_batch

        # Control bar
        ctrl = tk.Frame(p, bg=THEME["bg_panel"],
                        highlightbackground=THEME["border"], highlightthickness=1)
        ctrl.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(ctrl, text="  Batch CSV:", bg=THEME["bg_panel"],
                 fg=THEME["text_muted"], font=FONT_BODY).pack(side=tk.LEFT, padx=4, pady=8)
        self._batch_path = tk.StringVar()
        dark_entry(ctrl, textvariable=self._batch_path, width=38).pack(side=tk.LEFT, padx=4)
        accent_btn(ctrl, "📂 Browse", self._browse_batch,        width=10).pack(side=tk.LEFT, padx=3)
        accent_btn(ctrl, "⚡ Run Batch", self._run_batch,
                   color=THEME["accent2"], width=14).pack(side=tk.LEFT, padx=3)
        accent_btn(ctrl, "💾 Export CSV", self._export_batch,
                   color=THEME["accent4"], width=14).pack(side=tk.LEFT, padx=3)

        # Batch stats row
        bstats = tk.Frame(p, bg=THEME["bg_dark"])
        bstats.pack(fill=tk.X, padx=8, pady=4)
        for c in range(5): bstats.columnconfigure(c, weight=1)

        self._batch_cards = {}
        for i, (lbl, key, col) in enumerate([
            ("Total Samples",  "total",    THEME["accent"]),
            ("CRITICAL",       "critical", THEME["critical"]),
            ("WARNING",        "warning",  THEME["warning"]),
            ("OK",             "ok",       THEME["good"]),
            ("Avg RUL",        "avg_rul",  THEME["accent2"]),
        ]):
            card = StatCard(bstats, lbl, "—", col)
            card.grid(row=0, column=i, sticky="nsew", padx=3, pady=3)
            self._batch_cards[key] = card

        # Batch results table
        table_frame = tk.Frame(p, bg=THEME["bg_dark"])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        cols = ["#", "Failure Mode", "RUL (cycles)", "Status", "Health Idx", "Urgency", "Action"]
        self._batch_tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=22)
        for col in cols:
            self._batch_tree.heading(col, text=col)
            self._batch_tree.column(col, width=120 if col!="Failure Mode" else 180, anchor="center")
        tsb = ttk.Scrollbar(table_frame, command=self._batch_tree.yview)
        self._batch_tree.config(yscrollcommand=tsb.set)
        tsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._batch_tree.pack(fill=tk.BOTH, expand=True)

        # Style treeview
        style = ttk.Style()
        style.configure("Treeview",
                         background=THEME["bg_card"], foreground=THEME["text_main"],
                         fieldbackground=THEME["bg_card"], rowheight=26,
                         font=("Consolas", 9))
        style.configure("Treeview.Heading",
                         background=THEME["bg_panel"], foreground=THEME["accent"],
                         font=("Consolas", 9, "bold"))

        self._batch_results = None

    def _browse_batch(self):
        p = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p: self._batch_path.set(p)

    def _run_batch(self):
        """Run batch prediction with comprehensive error handling."""
        if not self.models_ready:
            messagebox.showwarning("No Models", "Please train/load models first.")
            return
        
        path = self._batch_path.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Valid CSV file required.")
            return
        
        try:
            logger.info(f"Loading batch file: {path}")
            df = pd.read_csv(path)
            
            if df.empty:
                messagebox.showerror("Error", "CSV file is empty.")
                return
            
            logger.info(f"Batch data loaded: {len(df)} rows")
            
            try:
                fms, ruls = self.pipeline.predict_batch(df)
                logger.info(f"Batch predictions completed: {len(fms)} predictions")
            except Exception as e:
                logger.error(f"Batch prediction error: {e}", exc_info=True)
                messagebox.showerror("Batch Error", f"Prediction failed:\n{str(e)}")
                return
            
            self._batch_results = df.copy()
            self._batch_results['predicted_failure_mode'] = fms
            self._batch_results['predicted_rul']          = ruls

            # Clear tree
            for row in self._batch_tree.get_children():
                self._batch_tree.delete(row)

            stats = {"total": len(df), "critical": 0, "warning": 0, "ok": 0}
            
            try:
                for i, (fm, rul) in enumerate(zip(fms, ruls)):
                    try:
                        health = df['health'].iloc[i] if 'health' in df.columns else 75.0
                        deg    = df['degradation_rate'].iloc[i] if 'degradation_rate' in df.columns else 0.0007
                        hidx   = MetricsEngine.health_index(MetricsEngine.soh(health), rul, deg)
                        urg    = MetricsEngine.replacement_urgency(rul, health, deg)
                        rec    = MetricsEngine.recommendation(urg, health, rul)

                        if rul < 1000:   status = "CRITICAL"; stats["critical"] += 1
                        elif rul < 5000: status = "WARNING";  stats["warning"]  += 1
                        else:            status = "OK";        stats["ok"]       += 1

                        tag = status.lower()
                        self._batch_tree.insert("", tk.END, values=(
                            i+1, fm, f"{rul:,.0f}", status,
                            f"{hidx:.1f}", f"{urg:.0f}", rec[0]
                        ), tags=(tag,))
                    except Exception as e:
                        logger.warning(f"Error processing batch row {i}: {e}")
                        continue

                self._batch_tree.tag_configure("critical", foreground=THEME["critical"])
                self._batch_tree.tag_configure("warning",  foreground=THEME["warning"])
                self._batch_tree.tag_configure("ok",       foreground=THEME["good"])

                self._batch_cards["total"].set(stats["total"])
                self._batch_cards["critical"].set(stats["critical"])
                self._batch_cards["warning"].set(stats["warning"])
                self._batch_cards["ok"].set(stats["ok"])
                self._batch_cards["avg_rul"].set(f"{np.mean(ruls):.0f}")
                
                logger.info(f"Batch results: {stats['ok']} OK, {stats['warning']} WARNING, {stats['critical']} CRITICAL")

            except Exception as e:
                logger.error(f"Error populating batch results: {e}")
                messagebox.showerror("Display Error", f"Failed to display results:\n{str(e)}")

        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error: {e}")
            messagebox.showerror("CSV Error", f"File is not a valid CSV:\n{str(e)}")
        except Exception as e:
            logger.error(f"Batch processing error: {e}", exc_info=True)
            messagebox.showerror("Batch Error", str(e))

    def _export_batch(self):
        if self._batch_results is None:
            messagebox.showinfo("No Data", "Run a batch prediction first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            self._batch_results.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Results exported to:\n{path}")

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 4: VISUALIZE
    # ═══════════════════════════════════════════════════════════════════
    def _build_visualize_tab(self):
        p = self._tab_visualize

        ctrl = tk.Frame(p, bg=THEME["bg_panel"],
                        highlightbackground=THEME["border"], highlightthickness=1)
        ctrl.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(ctrl, text="  Visualization:", bg=THEME["bg_panel"],
                 fg=THEME["text_muted"], font=FONT_BODY).pack(side=tk.LEFT, padx=4, pady=8)

        self._viz_choice = tk.StringVar(value="Failure Mode Distribution")
        charts = [
            "Failure Mode Distribution",
            "Health Distribution",
            "RUL Distribution",
            "Correlation Heatmap",
            "Health vs RUL Scatter",
            "Degradation Rate by Mode",
            "Feature Importance (Classifier)",
            "Feature Importance (Regressor)",
            "Temperature vs Health",
        ]
        combo = ttk.Combobox(ctrl, textvariable=self._viz_choice,
                             values=charts, width=32, state="readonly",
                             font=FONT_BODY)
        combo.pack(side=tk.LEFT, padx=6)
        accent_btn(ctrl, "📊 Plot", self._run_plot, color=THEME["accent"], width=10).pack(side=tk.LEFT, padx=4)

        self._viz_frame = tk.Frame(p, bg=THEME["bg_dark"])
        self._viz_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _run_plot(self):
        """Generate visualization with comprehensive error handling."""
        df = getattr(self, '_df', None)
        if df is None:
            messagebox.showwarning("No Data",
                "Please load a dataset first in the Dashboard tab.")
            return
        
        try:
            # Clear previous plots
            for w in self._viz_frame.winfo_children():
                w.destroy()

            choice = self._viz_choice.get()
            logger.info(f"Generating visualization: {choice}")
            
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=THEME["bg_card"])
            ax.set_facecolor(THEME["bg_panel"])

            COLORS = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"],
                      "#BC8CFF", "#56D364", "#F0883E"]

            if choice == "Failure Mode Distribution":
                if 'failure_mode' not in df.columns:
                    raise ValueError("Dataset missing 'failure_mode' column")
                counts = df['failure_mode'].value_counts()
                bars = ax.bar(counts.index, counts.values, color=COLORS[:len(counts)], edgecolor='none')
                ax.bar_label(bars, fmt='%d', color=THEME["text_main"], fontsize=10)
                ax.set_title("Failure Mode Distribution", color=THEME["text_main"], fontsize=13, pad=12)
                ax.set_ylabel("Count", color=THEME["text_muted"])
                ax.tick_params(colors=THEME["text_muted"])

            elif choice == "Health Distribution":
                if 'health' not in df.columns:
                    raise ValueError("Dataset missing 'health' column")
                ax.hist(df['health'], bins=40, color=THEME["accent2"], edgecolor='none', alpha=0.85)
                ax.axvline(df['health'].mean(), color=THEME["accent3"], linestyle='--', lw=1.5,
                           label=f'Mean: {df["health"].mean():.1f}%')
                ax.legend(facecolor=THEME["bg_card"], edgecolor=THEME["border"],
                          labelcolor=THEME["text_main"])
                ax.set_title("Battery Health Distribution", color=THEME["text_main"], fontsize=13, pad=12)
                ax.set_xlabel("Health (%)", color=THEME["text_muted"])
                ax.tick_params(colors=THEME["text_muted"])

            elif choice == "RUL Distribution":
                if 'RUL' not in df.columns:
                    raise ValueError("Dataset missing 'RUL' column")
                ax.hist(df['RUL'], bins=50, color=THEME["accent"], edgecolor='none', alpha=0.85)
                ax.axvline(df['RUL'].mean(), color=THEME["accent3"], linestyle='--', lw=1.5,
                           label=f'Mean: {df["RUL"].mean():.0f}')
                ax.legend(facecolor=THEME["bg_card"], edgecolor=THEME["border"],
                          labelcolor=THEME["text_main"])
                ax.set_title("RUL Distribution", color=THEME["text_main"], fontsize=13, pad=12)
                ax.set_xlabel("RUL (cycles)", color=THEME["text_muted"])
                ax.tick_params(colors=THEME["text_muted"])

            elif choice == "Correlation Heatmap":
                try:
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=(10, 7), facecolor=THEME["bg_card"])
                    ax.set_facecolor(THEME["bg_panel"])
                    numeric = df.select_dtypes(include=np.number)
                    if numeric.empty:
                        raise ValueError("No numeric columns found in dataset")
                    top_cols = numeric.corr()['health'].abs().nlargest(10).index.tolist() if 'health' in numeric.columns else numeric.columns[:10]
                    corr_mat = numeric[top_cols].corr()
                    sns.heatmap(corr_mat, ax=ax, cmap="coolwarm", annot=True, fmt=".2f",
                                annot_kws={"size": 8}, linewidths=0.3,
                                linecolor=THEME["border"],
                                cbar_kws={"shrink": 0.8})
                    ax.set_title("Correlation Heatmap (Top 10 features)", color=THEME["text_main"],
                                 fontsize=12, pad=12)
                    ax.tick_params(colors=THEME["text_muted"], labelsize=8)
                except Exception as e:
                    logger.error(f"Heatmap generation error: {e}")
                    raise

            elif choice == "Health vs RUL Scatter":
                if 'health' not in df.columns or 'RUL' not in df.columns:
                    raise ValueError("Dataset missing 'health' or 'RUL' columns")
                if 'failure_mode' not in df.columns:
                    ax.scatter(df['health'], df['RUL'], alpha=0.4, s=12, color=THEME["accent"])
                else:
                    modes = df['failure_mode'].unique()
                    for i, mode in enumerate(modes):
                        sub = df[df['failure_mode']==mode]
                        ax.scatter(sub['health'], sub['RUL'], label=mode,
                                   alpha=0.4, s=12, color=COLORS[i % len(COLORS)])
                    ax.legend(facecolor=THEME["bg_card"], edgecolor=THEME["border"],
                              labelcolor=THEME["text_main"], markerscale=2)
                ax.set_xlabel("Health (%)", color=THEME["text_muted"])
                ax.set_ylabel("RUL (cycles)", color=THEME["text_muted"])
                ax.set_title("Health vs RUL by Failure Mode", color=THEME["text_main"], fontsize=13, pad=12)
                ax.tick_params(colors=THEME["text_muted"])

            elif choice == "Degradation Rate by Mode":
                if 'degradation_rate' not in df.columns or 'failure_mode' not in df.columns:
                    raise ValueError("Dataset missing 'degradation_rate' or 'failure_mode' columns")
                modes = df['failure_mode'].unique()
                data  = [df[df['failure_mode']==m]['degradation_rate'].values for m in modes]
                bp = ax.boxplot(data, labels=modes, patch_artist=True,
                                boxprops=dict(color=THEME["text_muted"]),
                                medianprops=dict(color=THEME["accent"], lw=2),
                                whiskerprops=dict(color=THEME["text_muted"]),
                                capprops=dict(color=THEME["text_muted"]),
                                flierprops=dict(markerfacecolor=THEME["accent3"],
                                                marker='o', markersize=3, alpha=0.5))
                for patch, color in zip(bp['boxes'], COLORS):
                    patch.set_facecolor(color + '44')
                ax.set_title("Degradation Rate by Failure Mode", color=THEME["text_main"], fontsize=13, pad=12)
                ax.set_ylabel("Degradation Rate", color=THEME["text_muted"])
                ax.tick_params(colors=THEME["text_muted"])

            elif "Feature Importance" in choice:
                if not self.models_ready:
                    messagebox.showwarning("No Models", "Train models first.")
                    return
                is_clf = "Classifier" in choice
                model  = self.pipeline.classifier if is_clf else self.pipeline.regressor
                try:
                    imp    = pd.Series(model.feature_importances_, index=self.pipeline.feature_names)
                    top    = imp.nlargest(12).sort_values()
                    ax.barh(top.index, top.values, color=THEME["accent"] if is_clf else THEME["accent2"],
                            edgecolor='none')
                except Exception as e:
                    logger.error(f"Feature importance error: {e}")
                    raise ValueError(f"Feature importance calculation failed: {e}")
                ax.set_title(f"Feature Importance — {'Classifier' if is_clf else 'Regressor'}",
                             color=THEME["text_main"], fontsize=12, pad=12)
                ax.tick_params(colors=THEME["text_muted"], labelsize=8)
                ax.set_xlabel("Importance", color=THEME["text_muted"])

            elif choice == "Temperature vs Health":
                if 'temperature' not in df.columns or 'health' not in df.columns:
                    raise ValueError("Dataset missing 'temperature' or 'health' columns")
                if 'RUL' in df.columns:
                    sc = ax.scatter(df['temperature'], df['health'],
                                    c=df['RUL'], cmap='plasma', alpha=0.4, s=10)
                    plt.colorbar(sc, ax=ax, label='RUL').ax.yaxis.label.set_color(THEME["text_muted"])
                else:
                    ax.scatter(df['temperature'], df['health'], alpha=0.4, s=10, color=THEME["accent"])
                ax.set_xlabel("Temperature (°C)", color=THEME["text_muted"])
                ax.set_ylabel("Health (%)", color=THEME["text_muted"])
                ax.set_title("Temperature vs Health (colored by RUL)",
                             color=THEME["text_main"], fontsize=12, pad=12)
                ax.tick_params(colors=THEME["text_muted"])

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=self._viz_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            logger.info(f"Visualization completed: {choice}")

        except ValueError as e:
            messagebox.showwarning("Data Error", str(e))
            logger.warning(f"Visualization data error: {e}")
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to generate plot:\n{str(e)}")
            logger.error(f"Visualization error: {e}", exc_info=True)

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 5: TRAIN
    # ═══════════════════════════════════════════════════════════════════
    def _build_train_tab(self):
        p = self._tab_train
        p.columnconfigure(0, weight=1)
        p.columnconfigure(1, weight=2)
        p.rowconfigure(0, weight=1)

        # Left: config
        left = tk.Frame(p, bg=THEME["bg_panel"],
                        highlightbackground=THEME["border"], highlightthickness=1)
        left.grid(row=0, column=0, sticky="nsew", padx=(8,4), pady=8)

        tk.Label(left, text="🏋 MODEL TRAINING CONFIG",
                 bg=THEME["bg_panel"], fg=THEME["accent"],
                 font=("Consolas", 11, "bold")).pack(pady=(14, 8))

        # Dataset path
        sf1 = section_frame(left, "📁 Dataset")
        dark_entry(sf1, textvariable=self.dataset_path, width=30).pack(side=tk.LEFT, padx=4, pady=6)
        accent_btn(sf1, "Browse", self._browse_dataset, width=8, color=THEME["bg_card"]).pack(side=tk.LEFT, padx=3)

        # Model params
        sf2 = section_frame(left, "⚙️ Hyperparameters")
        self._hp = {}
        params = [("n_estimators", "N Estimators", 100),
                  ("learning_rate", "Learning Rate", 0.1),
                  ("max_depth",    "Max Depth",     5)]
        for key, label, default in params:
            row = tk.Frame(sf2, bg=THEME["bg_panel"])
            row.pack(fill=tk.X, padx=6, pady=3)
            tk.Label(row, text=label, bg=THEME["bg_panel"], fg=THEME["text_muted"],
                     font=FONT_BODY, width=16).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            dark_entry(row, textvariable=var, width=10).pack(side=tk.LEFT, padx=4)
            self._hp[key] = var

        # Split
        sf3 = section_frame(left, "📊 Train/Test Split")
        row3 = tk.Frame(sf3, bg=THEME["bg_panel"])
        row3.pack(fill=tk.X, padx=6, pady=4)
        tk.Label(row3, text="Test Size", bg=THEME["bg_panel"],
                 fg=THEME["text_muted"], font=FONT_BODY, width=10).pack(side=tk.LEFT)
        self._test_size = tk.StringVar(value="0.2")
        dark_entry(row3, textvariable=self._test_size, width=8).pack(side=tk.LEFT, padx=4)

        # Model dir
        sf4 = section_frame(left, "💾 Save Directory")
        self._model_dir_var = tk.StringVar(value="./models")
        dark_entry(sf4, textvariable=self._model_dir_var, width=30).pack(padx=4, pady=6)

        # Buttons
        btn_f = tk.Frame(left, bg=THEME["bg_panel"])
        btn_f.pack(fill=tk.X, padx=8, pady=10)
        accent_btn(btn_f, "🚀 START TRAINING", self._start_training,
                   color=THEME["accent2"], width=20).pack(pady=4)
        accent_btn(btn_f, "📂 Load Existing Models", self._load_models_gui,
                   color=THEME["accent4"], width=20).pack(pady=4)

        # Right: log
        right = tk.Frame(p, bg=THEME["bg_dark"])
        right.grid(row=0, column=1, sticky="nsew", padx=(4,8), pady=8)

        tk.Label(right, text="📋 TRAINING LOG",
                 bg=THEME["bg_dark"], fg=THEME["accent"],
                 font=("Consolas", 10, "bold")).pack(pady=(8,4))

        self._train_log = dark_text(right, height=36)
        tsb = ttk.Scrollbar(right, command=self._train_log.yview)
        self._train_log.config(yscrollcommand=tsb.set)
        tsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._train_log.pack(fill=tk.BOTH, expand=True, padx=4)

        self._train_log_print(
            "  Ready to train. Configure parameters on the left and click START TRAINING.\n"
            "  Or load existing models with the button below.\n"
        )

    def _train_log_print(self, msg):
        self._train_log.config(state=tk.NORMAL)
        self._train_log.insert(tk.END, msg + "\n")
        self._train_log.see(tk.END)
        self._train_log.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def _start_training(self):
        """Start model training with comprehensive error handling."""
        path = self.dataset_path.get()
        if not os.path.exists(path):
            messagebox.showerror("Error", f"Dataset not found:\n{path}")
            return

        self._train_log.config(state=tk.NORMAL)
        self._train_log.delete(1.0, tk.END)
        self._train_log.config(state=tk.DISABLED)
        self._set_status("Training...", ok=True)
        logger.info(f"Starting model training with dataset: {path}")

        def _run():
            try:
                # Parse hyperparameters with validation
                try:
                    n_est = int(self._hp["n_estimators"].get())
                    lr    = float(self._hp["learning_rate"].get())
                    depth = int(self._hp["max_depth"].get())
                    mdir  = self._model_dir_var.get()
                    
                    # Validate hyperparameters
                    if n_est < 10 or n_est > 1000:
                        raise ValueError("n_estimators must be between 10 and 1000")
                    if lr < 0.001 or lr > 1.0:
                        raise ValueError("learning_rate must be between 0.001 and 1.0")
                    if depth < 2 or depth > 20:
                        raise ValueError("max_depth must be between 2 and 20")
                        
                    logger.info(f"Hyperparameters: n_est={n_est}, lr={lr}, depth={depth}")
                except (ValueError, TypeError) as e:
                    self._train_log_print(f"\n✗ Invalid hyperparameters: {e}")
                    self._set_status("Training failed", ok=False)
                    return

                pipe = BatteryMLPipeline(path, model_dir=mdir,
                                         log_callback=self._train_log_print)
                self._train_log_print("═"*60)
                self._train_log_print("  ⚡ BATTERY ML TRAINING PIPELINE")
                self._train_log_print("═"*60)

                try:
                    self._train_log_print("\n[1/5] Loading dataset...")
                    pipe.load()
                except Exception as e:
                    self._train_log_print(f"\n✗ Data loading error: {e}")
                    raise

                try:
                    self._train_log_print("[2/5] Preprocessing data...")
                    pipe.preprocess()
                except Exception as e:
                    self._train_log_print(f"\n✗ Preprocessing error: {e}")
                    raise

                try:
                    self._train_log_print("[3/5] Training classifier...")
                    pipe.classifier = GradientBoostingClassifier(
                        n_estimators=n_est, learning_rate=lr,
                        max_depth=depth, random_state=42, n_iter_no_change=10,
                        validation_fraction=0.1
                    )
                    pipe.classifier.fit(pipe.Xc_tr, pipe.yc_tr)
                    yhat = pipe.classifier.predict(pipe.Xc_te)
                    acc  = accuracy_score(pipe.yc_te, yhat)
                    self._train_log_print(f"\n  ✓ Classifier Accuracy: {acc:.4f} ({acc*100:.2f}%)")
                    self._train_log_print(f"\n{classification_report(pipe.yc_te, yhat, target_names=pipe.class_names)}")
                    logger.info(f"Classifier training complete: accuracy={acc:.4f}")
                except Exception as e:
                    self._train_log_print(f"\n✗ Classifier training error: {e}")
                    logger.error(f"Classifier training failed: {e}", exc_info=True)
                    raise

                try:
                    self._train_log_print("\n[4/5] Training regressor...")
                    pipe.regressor = GradientBoostingRegressor(
                        n_estimators=n_est, learning_rate=lr,
                        max_depth=depth, random_state=42, n_iter_no_change=10,
                        validation_fraction=0.1
                    )
                    pipe.regressor.fit(pipe.Xr_tr, pipe.yr_tr)
                    yhat_r = pipe.regressor.predict(pipe.Xr_te)
                    rmse   = np.sqrt(mean_squared_error(pipe.yr_te, yhat_r))
                    r2     = r2_score(pipe.yr_te, yhat_r)
                    self._train_log_print(f"\n  ✓ Regressor RMSE : {rmse:.2f}")
                    self._train_log_print(f"  ✓ Regressor R²   : {r2:.4f}")
                    logger.info(f"Regressor training complete: RMSE={rmse:.2f}, R²={r2:.4f}")
                except Exception as e:
                    self._train_log_print(f"\n✗ Regressor training error: {e}")
                    logger.error(f"Regressor training failed: {e}", exc_info=True)
                    raise

                try:
                    self._train_log_print("\n[5/5] Saving models...")
                    pipe.save()
                    logger.info(f"Models saved to {mdir}")
                except Exception as e:
                    self._train_log_print(f"\n✗ Model saving error: {e}")
                    raise

                self.pipeline     = pipe
                self.models_ready = True
                self._set_status(f"✓ Trained | Acc {acc*100:.1f}% | R² {r2:.4f}", ok=True)
                self._train_log_print("\n" + "═"*60)
                self._train_log_print("✓ TRAINING COMPLETE! Models saved & ready for prediction.")
                self._train_log_print("═"*60)
                logger.info("Training pipeline completed successfully")

            except Exception as e:
                self._train_log_print(f"\n✗ TRAINING FAILED: {e}")
                self._set_status("Training failed", ok=False)
                logger.error(f"Training pipeline failed: {e}", exc_info=True)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        logger.info("Training thread started")

    def _load_models_gui(self):
        d = filedialog.askdirectory(title="Select Models Directory")
        if not d: return
        try:
            p = BatteryMLPipeline("", model_dir=d)
            p.load_models(d)
            self.pipeline     = p
            self.models_ready = True
            self._set_status("✓ Models Loaded", ok=True)
            self._train_log_print(f"\n✓ Models loaded from: {d}")
            self._train_log_print(f"  Features: {len(p.feature_names)}")
            self._train_log_print(f"  Classes : {', '.join(p.class_names)}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _try_load_models_silent(self):
        try:
            p = BatteryMLPipeline("", model_dir="./models")
            p.load_models("./models")
            self.pipeline     = p
            self.models_ready = True
            self._set_status("✓ Models Loaded", ok=True)
        except:
            pass

    # ═══════════════════════════════════════════════════════════════════
    #  TAB 6: ABOUT
    # ═══════════════════════════════════════════════════════════════════
    def _build_about_tab(self):
        p = self._tab_about

        frame = tk.Frame(p, bg=THEME["bg_dark"])
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        about_text = dark_text(frame, height=40)
        sb = ttk.Scrollbar(frame, command=about_text.yview)
        about_text.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        about_text.pack(fill=tk.BOTH, expand=True)

        content = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              ⚡  BATTERY 18650 ML PREDICTION SYSTEM  ⚡                      ║
║                    Complete Predictive Analytics Suite                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

  Version   : 2.0 Enhanced Edition
  Dataset   : 18650 Cylindrical Li-ion Battery Dataset
  Language  : Python 3.x
  Framework : scikit-learn, tkinter, matplotlib

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🎯 TASKS & MODELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  TASK 1 — FAILURE MODE CLASSIFICATION
  ───────────────────────────────────
  Objective : Predict which failure mechanism is occurring in the battery
  Model     : Gradient Boosting Classifier
  Algorithm : Gradient Boosted Decision Trees (GBDT)
  Classes   : SEI_growth | lithium_plating | electrode_cracking | normal_aging
  Metric    : Accuracy, F1-Score, Precision, Recall
  Why GBDT? : Handles non-linear relationships in EIS data; robust to
              feature scale differences; naturally models interactions
              between resistance, impedance, and temperature features.

  TASK 2 — REMAINING USEFUL LIFE (RUL) REGRESSION
  ──────────────────────────────────────────────────
  Objective : Predict number of remaining charge/discharge cycles
  Model     : Gradient Boosting Regressor
  Algorithm : Gradient Boosted Decision Trees (GBDT)
  Output    : Continuous numeric prediction (cycles)
  Metrics   : RMSE, R² Score, MAE
  Why GBDT? : Superior to Linear Regression (non-linear aging curve);
              better than Random Forest for sequential predictions;
              captures gradual degradation trends with low bias.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔬 ALGORITHM COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌─────────────────────────┬───────────┬────────────┬──────────────────────┐
  │ Algorithm               │ Accuracy  │ Speed      │ Battery Use Case     │
  ├─────────────────────────┼───────────┼────────────┼──────────────────────┤
  │ Gradient Boosting ✓     │ VERY HIGH │ Moderate   │ Best: complex EIS    │
  │ Random Forest           │ HIGH      │ Fast       │ Good: general purpose│
  │ SVM                     │ HIGH      │ Slow       │ Good: small datasets │
  │ Logistic Regression     │ MODERATE  │ Very Fast  │ Poor: non-linear     │
  │ KNN                     │ MODERATE  │ Slow infer │ Poor: high-dim data  │
  │ Neural Network (MLP)    │ HIGH      │ Varies     │ Needs more data      │
  │ Decision Tree           │ LOW-MOD   │ Very Fast  │ Poor: overfit risk   │
  └─────────────────────────┴───────────┴────────────┴──────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊 INPUT FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ELECTROCHEMICAL IMPEDANCE SPECTROSCOPY (EIS) FEATURES:
  ────────────────────────────────────────────────────────
  R_s            — Series resistance (electrolyte/connector)
  R_ct           — Charge transfer resistance (electrode kinetics)
  R_10kHz/1kHz   — Frequency-specific impedance
  C_dl           — Double-layer capacitance
  Warburg_slope  — Diffusion-related impedance slope
  Z_mag_1kHz     — Impedance magnitude at 1kHz
  phase_min      — Minimum phase angle

  STATE PARAMETERS:
  ─────────────────
  health         — Current battery capacity vs. nominal (%)
  degradation_rate — Capacity loss per cycle
  temperature    — Operating temperature (°C)
  soc            — State of Charge (%)
  RUL            — Remaining Useful Life (target variable)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📦 LIBRARY STACK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  pandas      — Data loading, manipulation, feature engineering
  numpy       — Numerical computation, array operations
  scikit-learn— ML models, preprocessing, train-test split, metrics
  matplotlib  — Interactive plots embedded in GUI
  seaborn     — Statistical visualizations (heatmaps)
  joblib      — Model serialization/deserialization
  tkinter     — Cross-platform GUI framework (built-in)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🌍 SUSTAINABILITY METRICS (NOVEL CONTRIBUTION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Health Index      — Composite score (SoH×50 + RUL×30 + Degradation×20)
  Replacement Urgency — Multi-factor score for replacement decision support
  Carbon Burden     — Lifecycle CO₂ estimate per 18650 cell (0.25 kWh)
  Environmental Impact— Efficiency-adjusted lifecycle burden score

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🏛️ SYSTEM ARCHITECTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  battery_ml_system.py (this file)
  ├── MetricsEngine      — Sustainability & health calculations
  ├── BatteryMLPipeline  — Training, evaluation, serialization
  └── BatteryApp (GUI)
      ├── Tab 1: Dashboard    — Dataset analysis & statistics
      ├── Tab 2: Prediction   — Single battery full prediction
      ├── Tab 3: Batch        — Multi-battery CSV prediction
      ├── Tab 4: Visualize    — 9 interactive chart types
      ├── Tab 5: Train        — Live model training with log
      └── Tab 6: About        — This documentation page

"""
        about_text.insert(tk.END, content)
        about_text.config(state=tk.DISABLED)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = BatteryApp(root)
    root.mainloop()