import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datetime import datetime
import logging
from typing import Tuple, Dict, Optional, List
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

try:
    from pysmu import Session, Mode
    PYSMU_AVAILABLE = True
except ImportError:
    PYSMU_AVAILABLE = False
    logger.warning("pysmu library not installed. Hardware acquisition will not be available.")


# ============================================================
# CONFIGURATION for Hongli IMR 18650 – 2600mAh
# ============================================================

RATED_CAPACITY_MAH  = 2600.0
VOLT_FULL           = 4.20
VOLT_EMPTY          = 3.00
EIS_FREQ_START      = 1.0
EIS_FREQ_STOP       = 5000.0
EIS_FREQ_POINTS     = 30
EIS_AMPLITUDE_V     = 0.005       # 5 mV AC perturbation

SAMPLE_RATE         = 100000      # 100 kSPS fixed on ADALM1000
SAMPLES_PER_READ    = 1000


# OCV–SOC approx tables (LiPo)
OCV_TABLES = {
    "LIPO": {
        "ocv": [3.00, 3.30, 3.50, 3.60, 3.70, 3.75, 3.80, 3.85,
                3.90, 3.95, 4.00, 4.10, 4.20],
        "soc": [0,    5,   10,   20,   30,   40,   50,   60,
                70,   80,   90,   95,  100]
    }
}

def voltage_to_soc(v, battery_type="LIPO"):
    t = OCV_TABLES[battery_type]
    if v <= t["ocv"][0]:
        return 0.0
    if v >= t["ocv"][-1]:
        return 100.0
    return float(interp1d(t["ocv"], t["soc"], kind='linear')(v))


class M1K_EIS_ONLY:
    """
    ADALM1000 Hardware Interface for EIS (Electrochemical Impedance Spectroscopy)
    
    Features:
    - Safe session management with proper cleanup
    - Retry logic for hardware operations
    - Comprehensive error handling
    - Type hints for better code clarity
    """
    
    def __init__(self):
        self.session: Optional[Session] = None
        self.dev     = None
        self.cha     = None
        self.chb     = None
        self._is_connected = False
        self._max_retries = 3
        logger.info("M1K_EIS_ONLY instance created")

    # ─────────────────────────── ADALM1000 connection ────────────────
    
    def connect(self) -> bool:
        """
        Connect to ADALM1000 hardware device.
        
        Returns:
            bool: True if connection successful
            
        Raises:
            RuntimeError: If no ADALM1000 device found or connection fails
        """
        if self._is_connected:
            logger.warning("Already connected. Skipping connection.")
            return True
            
        try:
            # Ensure old session is properly closed
            self._cleanup_session()
            
            self.session = Session()
            if not self.session or not self.session.devices:
                raise RuntimeError("No ADALM1000 found. Check USB connection.")
            
            self.dev = self.session.devices[0]
            self.cha = self.dev.channels['A']
            self.chb = self.dev.channels['B']
            self._is_connected = True
            
            logger.info(f"✓ Connected to ADALM1000 | SN: {self.dev.serial}")
            logger.info(f"  Firmware: {self.dev.fwver}")
            logger.info(f"  Hardware: {self.dev.hwver}")
            
            print(f"[OK] Connected: {self.dev.serial}")
            print(f"[OK] Firmware : {self.dev.fwver}")
            print(f"[OK] Hardware : {self.dev.hwver}")
            return True
            
        except Exception as e:
            logger.error(f"Hardware connection failed: {e}")
            self._is_connected = False
            raise RuntimeError(f"ADALM1000 connection error: {e}")

    def disconnect(self) -> None:
        """
        Safely disconnect from ADALM1000 and cleanup resources.
        """
        self._cleanup_session()
        self._is_connected = False
        logger.info("[EXIT] Session closed and resources cleaned up")
        print("[EXIT] Session closed.")

    def _cleanup_session(self) -> None:
        """
        Internal method to properly cleanup session and reset device state.
        """
        try:
            if self.cha or self.chb:
                try:
                    if self.cha:
                        self.cha.mode = Mode.HI_Z
                    if self.chb:
                        self.chb.mode = Mode.HI_Z
                    time.sleep(0.1)  # Allow time for mode change
                except Exception as e:
                    logger.debug(f"Channel reset issue (non-critical): {e}")
            
            if self.session:
                try:
                    if self.session:
                        self.session.end()
                except Exception as e:
                    logger.debug(f"Session end issue (non-critical): {e}")
                finally:
                    self.session = None
                    
        except Exception as e:
            logger.debug(f"Cleanup exception (non-critical): {e}")
        finally:
            self.cha = None
            self.chb = None

    # ─────────────────────────── OCV measurement ────────────────────
    
    def read_ocv(self, n_samples: int = 3000) -> Tuple[float, float]:
        """
        Measure Open Circuit Voltage (OCV) and leakage current.
        
        Args:
            n_samples: Number of samples to read
            
        Returns:
            Tuple of (battery_voltage_V, leakage_current_mA)
            
        Raises:
            RuntimeError: If not connected or measurement fails
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to hardware. Call connect() first.")
        
        try:
            self.cha.mode = Mode.HI_Z
            self.chb.mode = Mode.HI_Z
            self.session.start(n_samples)
            data = self.dev.read(n_samples, -1)
            self.session.cancel()
            
            if not data or len(data) == 0:
                logger.warning("No data returned from hardware")
                return 0.0, 0.0
            
            v_chb = float(np.mean([s[1][0] for s in data]))  # battery V
            i_cha = float(np.mean([s[0][1] for s in data]))  # leakage
            
            logger.debug(f"OCV measurement: V={v_chb:.4f}V, I={i_cha*1000:.2f}mA")
            return v_chb, i_cha * 1000.0  # V, mA
            
        except Exception as e:
            logger.error(f"OCV measurement failed: {e}")
            raise RuntimeError(f"OCV measurement error: {e}")

    # ─────────────────────────── DFT helper ────────────────────────
    
    @staticmethod
    def dft_at(signal: np.ndarray, freq: float, fs: float) -> complex:
        """
        Calculate DFT at specific frequency.
        
        Args:
            signal: Input signal array
            freq: Target frequency (Hz)
            fs: Sampling frequency (Hz)
            
        Returns:
            Complex DFT value
        """
        N = len(signal)
        if N == 0:
            return 0.0 + 0.0j
        
        t = np.arange(N) / fs
        ref = np.exp(-2j * np.pi * freq * t)
        return np.dot(signal.astype(complex), ref) / (N / 2)

    # ─────────────────────────── EIS Only sweep ────────────────────
    
    def run_eis(self, max_attempts: int = 1) -> Dict:
        """
        Run complete EIS (Electrochemical Impedance Spectroscopy) sweep.
        
        Args:
            max_attempts: Number of retry attempts if measurement fails
            
        Returns:
            Dictionary containing 24 EIS parameters and diagnostics
            
        Raises:
            ValueError: If insufficient valid EIS points collected
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to hardware. Call connect() first.")
        
        print("\n" + "═" * 60)
        print("  ADALM1000 EIS Only (Hongli 18650)")
        print("═" * 60)
        
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                logger.info(f"EIS sweep attempt {attempt + 1}/{max_attempts}")
                
                print("[EIS] Measuring OCV for DC bias...")
                v_ocv, _i = self.read_ocv(3000)
                dc_bias = v_ocv
                
                logger.info(f"OCV measurement success: {v_ocv:.5f}V")
                
                print(f"  OCV          : {v_ocv:.5f} V")
                print(f"  DC Bias      : {dc_bias:.5f} V")
                print(f"  AC Amplitude : {EIS_AMPLITUDE_V*1000.0:.2f} mV")
                print(f"  Freq Range   : {EIS_FREQ_START}–{EIS_FREQ_STOP} Hz "
                      f"({EIS_FREQ_POINTS} pts)")

                freqs = np.logspace(np.log10(EIS_FREQ_START),
                                    np.log10(EIS_FREQ_STOP),
                                    EIS_FREQ_POINTS)

                Z_real_arr = []
                Z_imag_arr = []
                valid_points = 0

                for idx, f in enumerate(freqs):
                    try:
                        n_cycles = max(10, int(SAMPLE_RATE / f * 10))
                        n_samp   = min(n_cycles, 100000)
                        t        = np.arange(n_samp) / SAMPLE_RATE
                        omega    = 2 * np.pi * f

                        v_wave = dc_bias + EIS_AMPLITUDE_V * np.sin(omega * t)
                        v_wave = np.clip(v_wave, 0.0, 5.0)

                        self.cha.mode = Mode.SVMI
                        self.chb.mode = Mode.HI_Z
                        self.cha.write(v_wave.tolist())

                        self.session.start(n_samp)
                        data = self.dev.read(n_samp, -1)
                        self.session.cancel()

                        v_cha_raw = np.array([s[0][0] for s in data])
                        i_cha_raw = np.array([s[0][1] for s in data])
                        v_chb_raw = np.array([s[1][0] for s in data])

                        v_cha_ac = v_cha_raw - np.mean(v_cha_raw)
                        v_chb_ac = v_chb_raw - np.mean(v_chb_raw)

                        v_bat_ac = v_chb_ac
                        i_ac     = i_cha_raw

                        V_ph = self.dft_at(v_bat_ac, f, SAMPLE_RATE)
                        I_ph = self.dft_at(i_ac,     f, SAMPLE_RATE)

                        if abs(I_ph) < 1e-9:
                            Z_real_arr.append(np.nan)
                            Z_imag_arr.append(np.nan)
                            print(f"  [{idx+1:02d}/{EIS_FREQ_POINTS}] "
                                  f"f={f:8.1f} Hz | zero current – skipped")
                            continue

                        Z = V_ph / I_ph
                        Z_real_arr.append(float(Z.real))
                        Z_imag_arr.append(float(Z.imag))
                        valid_points += 1

                        Z_mag = abs(Z)
                        phase_deg = np.degrees(np.angle(Z))
                        print(f"  [{idx+1:02d}/{EIS_FREQ_POINTS}] "
                              f"f={f:8.1f} Hz | |Z|={Z_mag:8.4f} Ω | φ={phase_deg:+7.2f}°")
                              
                    except Exception as e:
                        logger.debug(f"Single frequency measurement error at {f}Hz: {e}")
                        Z_real_arr.append(np.nan)
                        Z_imag_arr.append(np.nan)
                        continue

                Z_real_arr = np.array(Z_real_arr)
                Z_imag_arr = np.array(Z_imag_arr)
                mask = ~np.isnan(Z_real_arr)
                freqs = freqs[mask]
                Z_real = Z_real_arr[mask]
                Z_imag = Z_imag_arr[mask]

                if len(freqs) < 5:
                    raise ValueError(f"[ERROR] Only {len(freqs)} valid EIS points (need ≥5). Check wiring.")

                logger.info(f"EIS sweep complete: {valid_points} valid points collected")

                params = self._extract_all_eis_params(freqs, Z_real, Z_imag)
                params["ocv"] = float(v_ocv)
                params["soc"] = float(voltage_to_soc(v_ocv, "LIPO"))

                R_s = params["R_s"]
                R_ct = params["R_ct"]
                temp = self.estimate_temperature(R_s, R_ct, params["phase_min"], v_ocv)
                health = self.estimate_health(R_ct, R_s)
                deg_rate = self.estimate_degradation_rate(health)

                params["temperature"]        = float(temp)
                params["health"]             = float(health)
                params["degradation_rate"]   = float(deg_rate)

                self._save_eis(freqs, Z_real, Z_imag, params)
                self._print_eis_results(params)

                logger.info("EIS measurement and processing successful")
                return params
                
            except ValueError as e:
                last_error = e
                logger.error(f"EIS validation error (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying in 2 seconds...")
                    time.sleep(2)
            except Exception as e:
                last_error = e
                logger.error(f"EIS measurement error (attempt {attempt + 1}): {e}")
                if attempt < max_attempts - 1:
                    logger.info(f"Retrying in 2 seconds...")
                    time.sleep(2)
        
        # All attempts failed
        raise RuntimeError(f"EIS measurement failed after {max_attempts} attempt(s): {last_error}")

    # ─────────────────────────── Extract all 24 EIS parameters ────────
    
    def _extract_all_eis_params(self, freq: np.ndarray, Z_real: np.ndarray, 
                                Z_imag: np.ndarray) -> Dict[str, float]:
        """
        Extract comprehensive EIS parameters (24 core metrics).
        
        Args:
            freq: Frequency array (Hz)
            Z_real: Real impedance components
            Z_imag: Imaginary impedance components
            
        Returns:
            Dictionary of 24 EIS parameters with proper validation
        """
        Z     = Z_real + 1j * Z_imag
        Z_mag = np.abs(Z)
        phase = np.angle(Z, deg=True)
        omega = 2 * np.pi * freq

        def at_f(ft: float, arr: np.ndarray) -> float:
            """Safely interpolate impedance at target frequency."""
            ft = float(np.clip(ft, freq.min(), freq.max()))
            return float(interp1d(freq, arr, kind='linear', bounds_error=False, 
                                fill_value='extrapolate')(ft))

        p = {}

        # Frequency-specific resistance
        p["R_10kHz"]       = at_f(min(10000.0, freq.max()), Z_real)
        p["R_1kHz"]        = at_f(min(1000.0,  freq.max()), Z_real)
        p["R_100Hz"]       = at_f(min(100.0,   freq.max()), Z_real)
        p["R_10Hz"]        = at_f(min(10.0,    freq.max()), Z_real)
        p["R_1Hz"]         = at_f(max(1.0,     freq.min()), Z_real)

        p["Z_mag_1kHz"]    = at_f(min(1000.0, freq.max()), Z_mag)
        p["Z_mag_100Hz"]   = at_f(min(100.0,  freq.max()), Z_mag)

        p["phase_1kHz"]    = at_f(min(1000.0, freq.max()), phase)
        p["phase_100Hz"]   = at_f(min(100.0,  freq.max()), phase)

        p["phase_min"]     = float(np.min(phase))

        # Series resistance from high-frequency region
        hf_mask = freq >= np.percentile(freq, 75)
        if np.sum(hf_mask) > 0:
            hf_Zreal = Z_real[hf_mask]
            hf_Zimag = Z_imag[hf_mask]
            i_min = np.argmin(np.abs(hf_Zimag))
            p["R_s"] = float(hf_Zreal[i_min])
        else:
            i_min = np.argmin(np.abs(Z_imag))
            p["R_s"] = float(Z_real[i_min])

        # Constraint: R_s must be positive and reasonable
        p["R_s"] = max(0.0001, float(p["R_s"]))

        # Frequencies of interest
        p["freq_min_imag"] = float(freq[np.argmin(Z_imag)])
        p["freq_phase_45"] = float(freq[np.argmin(np.abs(phase + 45.0))])

        # Warburg impedance (low-frequency tail)
        wf_mask = freq < p["freq_phase_45"]
        if wf_mask.sum() >= 3:
            x = 1.0 / np.sqrt(omega[wf_mask])
            try:
                mr, br = np.polyfit(x, Z_real[wf_mask], 1)
                mi, bi = np.polyfit(x, -Z_imag[wf_mask], 1)
                p["Warburg_slope"]     = float((mr + mi) / 2.0)
                p["Warburg_intercept"] = float((br + bi) / 2.0)
            except Exception as e:
                logger.debug(f"Warburg fit warning: {e}")
                p["Warburg_slope"] = 0.0
                p["Warburg_intercept"] = 0.0
        else:
            p["Warburg_slope"] = 0.0
            p["Warburg_intercept"] = 0.0

        # Charge transfer resistance & double-layer capacitance
        R_ct = float(max(np.max(Z_real) - p["R_s"], 0.001))
        f_pk = p["freq_min_imag"]
        if R_ct > 0 and f_pk > 0:
            C_dl = 1.0 / (2.0 * np.pi * R_ct * f_pk)
        else:
            C_dl = 1e-4

        p["R_ct"] = R_ct
        p["C_dl"] = float(max(1e-6, C_dl))  # Constrain to positive value
        p["R_ct_R_s_ratio"] = (p["R_ct"] / p["R_s"] if p["R_s"] > 0 else 0.0)

        return p

    # ─────────────────────────── Health / temp heuristics ────────────
    
    def estimate_temperature(self, R_s_measured: float, R_ct_measured: Optional[float] = None,
                           phase_min: Optional[float] = None, 
                           voltage_ocv: Optional[float] = None) -> float:
        """
        Estimate battery temperature from EIS parameters.
        
        Args:
            R_s_measured: Series resistance (Ω)
            R_ct_measured: Charge transfer resistance (Ω)
            phase_min: Minimum phase angle (°)
            voltage_ocv: Open circuit voltage (V)
            
        Returns:
            Estimated temperature in Celsius
        """
        temperatures = []
        BATTERY_TEMP_REF = 25.0
        R_CT_TEMP_COEFF = 0.0015
        R_S_TEMP_COEFF  = 0.004
        R_s_ref = 0.060  # typical 18650 R_s

        # R_ct method
        if R_ct_measured is not None and 0.001 < R_ct_measured < 1.0:
            R_ct_ref_25c = 0.08
            temp = BATTERY_TEMP_REF + (R_ct_measured - R_ct_ref_25c) \
                                 / (R_ct_ref_25c * R_CT_TEMP_COEFF)
            temperatures.append(np.clip(temp, 0.0, 50.0))

        # R_s method
        if 0.8 < R_s_measured < 1.2:
            temp = BATTERY_TEMP_REF + (R_s_measured - 1.0) / (1.0 * R_S_TEMP_COEFF)
            temperatures.append(np.clip(temp, 0.0, 50.0))

        # Phase min heuristic
        if phase_min is not None and phase_min < -175.0:
            temp = BATTERY_TEMP_REF + (phase_min + 179.98) / 0.1
            temperatures.append(np.clip(temp, 0.0, 50.0))

        # OCV heuristic
        if voltage_ocv is not None and voltage_ocv > 2.5:
            ocv_ref_25c = 3.70
            ocv_temp_coeff = -0.0015
            temp = BATTERY_TEMP_REF + (voltage_ocv - ocv_ref_25c) / ocv_temp_coeff
            temperatures.append(np.clip(temp, 0.0, 50.0))

        return float(np.mean(temperatures)) if temperatures else 25.0

    def estimate_health(self, R_ct: float, R_s: float, R_ct_ref: float = 0.05) -> float:
        """
        Estimate State of Health (SoH) from impedance parameters.
        
        Args:
            R_ct: Charge transfer resistance (Ω)
            R_s: Series resistance (Ω)
            R_ct_ref: Reference R_ct at 100% health
            
        Returns:
            Health percentage (0-100%)
        """
        if R_ct <= 0 or R_s <= 0:
            return 80.0
        ratio = R_ct / (R_s * 1.0)
        health = float(np.clip(100.0 - (ratio * 50.0), 20.0, 100.0))
        return health

    def estimate_degradation_rate(self, health: float, 
                                 baseline_health: float = 100.0) -> float:
        """
        Estimate battery degradation rate.
        
        Args:
            health: Current health percentage
            baseline_health: Reference health at start-of-life
            
        Returns:
            Degradation rate (fraction per cycle)
        """
        degradation = baseline_health - health
        rate = float(np.clip(degradation / 50.0, 0.0, 5.0))
        return rate

    # ─────────────────────────── Print 24 parameters ─────────────────
    
    def _print_eis_results(self, p: Dict[str, float]) -> None:
        """Print comprehensive EIS parameter results."""
        cols = [
            "R_10kHz", "R_1kHz", "R_100Hz", "R_10Hz", "R_1Hz",
            "Z_mag_1kHz", "Z_mag_100Hz",
            "phase_1kHz", "phase_100Hz", "phase_min",
            "R_s", "freq_min_imag", "freq_phase_45",
            "Warburg_slope", "Warburg_intercept",
            "R_ct", "C_dl", "R_ct_R_s_ratio",
            "temperature", "health", "soc", "degradation_rate"
        ]
        units = {
            "R_10kHz": "Ω", "R_1kHz": "Ω", "R_100Hz": "Ω",
            "R_10Hz": "Ω", "R_1Hz": "Ω",
            "Z_mag_1kHz": "Ω", "Z_mag_100Hz": "Ω",
            "phase_1kHz": "°", "phase_100Hz": "°", "phase_min": "°",
            "R_s": "Ω",
            "freq_min_imag": "Hz", "freq_phase_45": "Hz",
            "Warburg_slope": "Ω·s^0.5", "Warburg_intercept": "Ω",
            "R_ct": "Ω",
            "C_dl": "F",
            "R_ct_R_s_ratio": "",
            "temperature": "°C",
            "health": "%",
            "soc": "%",
            "degradation_rate": "%/cycle"
        }

        print("\n" + "═" * 52)
        print("  EIS PARAMETER RESULTS (24 core parameters)")
        print("═" * 52)
        for k in cols:
            v = p.get(k, np.nan)
            u = units.get(k, "")
            print(f"  {k:<25s} = {v:>14.6f} {u}")
        print("═" * 52)

    # ─────────────────────────── EIS plot (Nyquist + Bode) ──────────
    
    def _plot_eis(self, freq: np.ndarray, Z_real: np.ndarray, 
                  Z_imag: np.ndarray, params: Dict[str, float]) -> None:
        """Generate EIS visualization plots."""
        Z_mag = np.abs(Z_real + 1j * Z_imag)
        phase = np.degrees(np.arctan2(Z_imag, Z_real))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"ADALM1000 EIS | OCV={params['ocv']:.3f}V | "
            f"R_s={params['R_s']:.4f}Ω | "
            f"T≈{params['temperature']:.1f}°C",
            fontsize=12, fontweight='bold'
        )

        # Nyquist
        axes[0].plot(Z_real, -Z_imag, 'bo-', ms=5, lw=1.5)
        axes[0].axvline(params["R_s"], color='red', ls='--', lw=1.5,
                        label=f'R_s={params["R_s"]:.4f}Ω')
        axes[0].set_xlabel('Z_real (Ω)')
        axes[0].set_ylabel('−Z_imag (Ω)')
        axes[0].set_title('Nyquist Plot')
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)
        axes[0].set_aspect('equal', adjustable='datalim')

        # Bode |Z|
        axes[1].semilogx(freq, Z_mag, 'b.-', lw=1.5)
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('|Z| (Ω)')
        axes[1].set_title('Bode – Magnitude')
        axes[1].grid(True, which='both', alpha=0.3)

        # Bode phase
        axes[2].semilogx(freq, phase, 'r.-', lw=1.5)
        axes[2].axhline(-45, color='gray', ls='--', lw=1,
                        label='−45°')
        axes[2].set_xlabel('Frequency (Hz)')
        axes[2].set_ylabel('Phase (°)')
        axes[2].set_title('Bode – Phase')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        plt.savefig("eis_plot.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("EIS plot saved to eis_plot.png")
        print("[SAVED] eis_plot.png")

    # ─────────────────────────── CSV saving (EIS‑only) ───────────────
    
    def _save_eis(self, freq: np.ndarray, Z_real: np.ndarray, 
                  Z_imag: np.ndarray, params: Dict[str, float]) -> None:
        """Save EIS data and parameters to CSV files."""
        try:
            # Raw spectrum
            pd.DataFrame({
                "freq_Hz":   freq,
                "Z_real":    Z_real,
                "Z_imag":    Z_imag,
                "Z_mag":     np.abs(Z_real + 1j * Z_imag),
                "phase_deg": np.degrees(np.arctan2(Z_imag, Z_real))
            }).to_csv("eis_raw.csv", index=False)

            # Parameters
            cols = [
                "R_10kHz", "R_1kHz", "R_100Hz", "R_10Hz", "R_1Hz",
                "Z_mag_1kHz", "Z_mag_100Hz",
                "phase_1kHz", "phase_100Hz", "phase_min",
                "R_s", "freq_min_imag", "freq_phase_45",
                "Warburg_slope", "Warburg_intercept",
                "R_ct", "C_dl", "R_ct_R_s_ratio",
                "temperature", "health", "soc", "degradation_rate"
            ]

            row = {k: params.get(k, np.nan) for k in cols}
            pd.DataFrame([row]).to_csv("eis_params.csv", index=False)
            logger.info("EIS data saved: eis_raw.csv, eis_params.csv")
            print("[SAVED] eis_raw.csv | eis_params.csv")
            
        except Exception as e:
            logger.error(f"Failed to save EIS data: {e}")
            raise





# ============================================================
# MAIN (EIS Only – your Mode 2)
# ============================================================
if __name__ == "__main__":
    """Main execution block with comprehensive error handling."""
    logger.info("Starting ADALM1000 EIS measurement system")
    
    bms = M1K_EIS_ONLY()

    try:
        if not bms.connect():
            logger.error("Failed to connect to ADALM1000")
            exit(1)

        params = bms.run_eis()
        logger.info("EIS measurement completed successfully")
        
        # Optional: uncomment to generate plots
        # bms._plot_eis(freq, Z_real, Z_imag, params)

    except KeyboardInterrupt:
        logger.info("Measurement interrupted by user")
        print("\n[STOP] Interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n[ERROR] {e}")
        exit(1)
    finally:
        try:
            bms.disconnect()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
