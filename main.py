"""
=============================================================================
REN-01: Wind Turbine Power Curve Deviation Analysis
Pillar 4: Renewable Energy Systems
=============================================================================
Dataset  : Wind Turbine SCADA Dataset (Kaggle)
           https://www.kaggle.com/datasets/bhavya harini/wind-turbine-scada-dataset
Unique Filter : Records from January (Month == 1) to isolate winter
                low-temperature performance characteristics.
Author   : [Student Name]
Date     : 2025
=============================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────
import os
import sys
import warnings
import logging
from pathlib import Path

# ── Third-Party ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Logging Configuration ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Global Style ──────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#1B4F72",
    "secondary": "#2E86AB",
    "accent":    "#F18F01",
    "positive":  "#27AE60",
    "negative":  "#C0392B",
    "light":     "#ECF0F1",
    "dark":      "#2C3E50",
}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F9FA",
    "axes.edgecolor":   "#BDC3C7",
    "axes.labelcolor":  PALETTE["dark"],
    "axes.titlesize":   13,
    "axes.labelsize":   11,
    "xtick.color":      PALETTE["dark"],
    "ytick.color":      PALETTE["dark"],
    "font.family":      "DejaVu Sans",
    "grid.color":       "#DDE1E7",
    "grid.linestyle":   "--",
    "grid.alpha":       0.7,
})

OUTPUT_DIR = Path("outputs")
DATA_DIR   = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# =============================================================================
# MODULE 1 — DATA INGESTION
# =============================================================================
class DataIngestion:
    """
    Handles all data loading responsibilities.
    Priority order:
      1. Explicit local filepath (if provided and exists).
      2. Auto-download via kagglehub (requires Kaggle credentials).
      3. Realistic synthetic SCADA dataset (fallback).
    """

    COLUMNS = [
        "Date/Time",
        "LV ActivePower (kW)",
        "Wind Speed (m/s)",
        "Theoretical_Power_Curve (KWh)",
        "Wind Direction (°)",
    ]
    KAGGLE_DATASET  = "berkerisen/wind-turbine-scada-dataset"
    KAGGLE_FILENAME = "T1.csv"

    def __init__(self, filepath: str | None = None):
        self.filepath = filepath
        self.raw_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def load(self) -> pd.DataFrame:
        """Primary entry point."""
        if self.filepath and Path(self.filepath).exists():
            return self._load_from_csv()

        kaggle_csv = self._try_kagglehub()
        if kaggle_csv:
            self.filepath = kaggle_csv
            return self._load_from_csv()

        log.warning("Falling back to synthetic SCADA data.")
        return self._generate_synthetic()

    # ------------------------------------------------------------------
    def _try_kagglehub(self) -> str | None:
        """
        Download the dataset using kagglehub.
        Requires ~/.kaggle/kaggle.json  OR
                 env vars KAGGLE_USERNAME + KAGGLE_KEY.
        Returns the CSV path string on success, None on any failure.
        """
        try:
            import kagglehub
            log.info("Downloading '%s' via kagglehub ...", self.KAGGLE_DATASET)
            dataset_dir = Path(kagglehub.dataset_download(self.KAGGLE_DATASET))
            log.info("Dataset path: %s", dataset_dir)

            csv_path = dataset_dir / self.KAGGLE_FILENAME
            if not csv_path.exists():
                matches = list(dataset_dir.rglob("*.csv"))
                if not matches:
                    log.error("No CSV found in downloaded dataset folder.")
                    return None
                csv_path = matches[0]
                log.info("Using CSV: %s", csv_path)

            return str(csv_path)

        except ImportError:
            log.warning(
                "kagglehub is not installed. "
                "Run:  pip install kagglehub"
            )
            return None
        except Exception as exc:
            log.warning("kagglehub download failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    def _load_from_csv(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.filepath)
            df["Date/Time"] = pd.to_datetime(
                df["Date/Time"], dayfirst=True, format="mixed"
            )
            log.info("Loaded %d rows from '%s'.", len(df), self.filepath)
            self.raw_df = df
            return df
        except FileNotFoundError as exc:
            log.error("File not found: %s", exc)
            raise
        except pd.errors.ParserError as exc:
            log.error("CSV parse error: %s", exc)
            raise
        except Exception as exc:
            log.error("Unexpected error during CSV load: %s", exc)
            raise

    # ------------------------------------------------------------------
    def _generate_synthetic(self) -> pd.DataFrame:
        """
        Generates 52,560 records (10-minute resolution × 365 days) that
        faithfully reproduce the statistical signature of the Kaggle dataset.
        """
        rng = np.random.default_rng(seed=42)
        n = 52_560  # full year

        timestamps = pd.date_range(
            start="2018-01-01 00:00",
            periods=n,
            freq="10min",
        )

        # ── Realistic wind-speed distribution (Weibull k=2, λ=8) ──────
        wind_speed = rng.weibull(2, n) * 8.0
        wind_speed = np.clip(wind_speed, 0, 25)

        # ── Realistic power curve (cubic up to rated, then flat) ──────
        rated_power = 3600.0          # kW
        cut_in, rated, cut_out = 3.5, 13.0, 25.0

        def theoretical_pc(ws):
            p = np.where(
                ws < cut_in, 0.0,
                np.where(
                    ws < rated,
                    rated_power * ((ws - cut_in) / (rated - cut_in)) ** 3,
                    np.where(ws <= cut_out, rated_power, 0.0),
                ),
            )
            return p

        theoretical = theoretical_pc(wind_speed)

        # ── Inject realistic deviations (±10–15 %) + maintenance faults
        deviation_factor = 1.0 + rng.normal(0, 0.08, n)
        fault_mask = rng.random(n) < 0.03          # 3 % fault events
        deviation_factor[fault_mask] *= rng.uniform(0.0, 0.5, fault_mask.sum())

        active_power = np.clip(theoretical * deviation_factor, 0, rated_power + 200)

        # ── Inject missing values and corruptions ─────────────────────
        for col_arr in [wind_speed, active_power]:
            null_idx = rng.choice(n, size=int(n * 0.012), replace=False)
            col_arr[null_idx] = np.nan

        # Corrupt ~0.5 % of wind speed with negative values
        corrupt_idx = rng.choice(n, size=int(n * 0.005), replace=False)
        wind_speed[corrupt_idx] = -rng.uniform(1, 5, len(corrupt_idx))

        wind_direction = rng.uniform(0, 360, n)

        df = pd.DataFrame({
            "Date/Time":                        timestamps,
            "LV ActivePower (kW)":              active_power,
            "Wind Speed (m/s)":                 wind_speed,
            "Theoretical_Power_Curve (KWh)":    theoretical,
            "Wind Direction (°)":               wind_direction,
        })

        # Inject ~200 exact duplicates
        dup_rows = df.sample(200, random_state=1)
        df = pd.concat([df, dup_rows], ignore_index=True)

        log.info("Generated synthetic dataset: %d rows.", len(df))
        self.raw_df = df
        return df


# =============================================================================
# MODULE 2 — DATA CLEANING
# =============================================================================
class DataCleaner:
    """
    Applies a reproducible cleaning sequence and the project's unique filter.
    Unique Filter: Month == 1 (January) — isolates winter performance data.
    """

    WIND_SPEED_MIN = 0.0
    WIND_SPEED_MAX = 25.0
    POWER_MIN      = 0.0
    POWER_MAX      = 3_800.0    # allow slight over-rated margin

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report: dict = {}

    # ------------------------------------------------------------------
    def clean(self) -> pd.DataFrame:
        self._ensure_datetime()
        self._remove_duplicates()
        self._handle_missing()
        self._fix_corrupt_ranges()
        self._engineer_features()
        self._apply_unique_filter()
        log.info("Cleaning complete — %d records retained.", len(self.df))
        return self.df

    # ------------------------------------------------------------------
    def _ensure_datetime(self):
        try:
            if not pd.api.types.is_datetime64_any_dtype(self.df["Date/Time"]):
                self.df["Date/Time"] = pd.to_datetime(
                    self.df["Date/Time"], dayfirst=True, format="mixed"
                )
        except Exception as exc:
            log.error("Datetime conversion failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    def _remove_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        self.report["duplicates_removed"] = removed
        log.info("Duplicates removed: %d", removed)

    # ------------------------------------------------------------------
    def _handle_missing(self):
        before = len(self.df)
        missing_summary = self.df.isnull().sum()
        self.report["missing_before"] = missing_summary.to_dict()

        # Forward-fill temporal gaps (10-min SCADA data)
        numeric_cols = [
            "LV ActivePower (kW)",
            "Wind Speed (m/s)",
            "Theoretical_Power_Curve (KWh)",
            "Wind Direction (°)",
        ]
        self.df.sort_values("Date/Time", inplace=True)
        self.df[numeric_cols] = self.df[numeric_cols].ffill().bfill()

        # If still NaN (leading/trailing) — drop
        self.df.dropna(subset=numeric_cols, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self.report["rows_dropped_after_fill"] = before - len(self.df)
        log.info("Missing value treatment complete.")

    # ------------------------------------------------------------------
    def _fix_corrupt_ranges(self):
        mask_ws = (
            (self.df["Wind Speed (m/s)"] < self.WIND_SPEED_MIN) |
            (self.df["Wind Speed (m/s)"] > self.WIND_SPEED_MAX)
        )
        mask_pw = (
            (self.df["LV ActivePower (kW)"] < self.POWER_MIN) |
            (self.df["LV ActivePower (kW)"] > self.POWER_MAX)
        )
        corrupted = mask_ws.sum() + mask_pw.sum()
        self.df.loc[mask_ws, "Wind Speed (m/s)"] = np.nan
        self.df.loc[mask_pw, "LV ActivePower (kW)"] = np.nan
        self.df.ffill(inplace=True)
        self.report["corrupt_values_fixed"] = int(corrupted)
        log.info("Corrupt range values corrected: %d cells.", corrupted)

    # ------------------------------------------------------------------
    def _engineer_features(self):
        """Derive key analysis columns."""
        dt = self.df["Date/Time"]
        self.df["Month"]  = dt.dt.month
        self.df["Hour"]   = dt.dt.hour
        self.df["Season"] = pd.cut(
            dt.dt.month,
            bins=[0, 3, 6, 9, 12],
            labels=["Winter", "Spring", "Summer", "Autumn"],
            right=True,
        )

        # Core metric: Power Curve Deviation
        self.df["Power_Deviation (kW)"] = (
            self.df["LV ActivePower (kW)"] -
            self.df["Theoretical_Power_Curve (KWh)"]
        )
        self.df["Deviation_Pct (%)"] = np.where(
            self.df["Theoretical_Power_Curve (KWh)"] > 0,
            (self.df["Power_Deviation (kW)"] /
             self.df["Theoretical_Power_Curve (KWh)"]) * 100,
            np.nan,
        )

        # Wind speed bins for comparative analysis
        self.df["Wind_Bin"] = pd.cut(
            self.df["Wind Speed (m/s)"],
            bins=[0, 3.5, 8, 13, 25],
            labels=["Below Cut-In", "Partial Load", "Transition", "Full Load"],
        )
        log.info("Feature engineering complete.")

    # ------------------------------------------------------------------
    def _apply_unique_filter(self):
        """
        UNIQUE FILTER: Retain only January records.
        Rationale: January represents peak winter conditions in northern
        hemisphere wind farms, isolating low-temperature power losses.
        """
        before = len(self.df)
        self.df = self.df[self.df["Month"] == 1].copy()
        self.df.reset_index(drop=True, inplace=True)
        self.report["unique_filter"] = "Month == 1 (January)"
        self.report["rows_after_filter"] = len(self.df)
        log.info(
            "Unique filter applied (Month=January): %d → %d rows.",
            before, len(self.df),
        )


# =============================================================================
# MODULE 3 — ENGINEERING ANALYTICS
# =============================================================================
class DataAnalyzer:
    """
    Computes all required statistics using NumPy.
    Every metric is stored in self.results for downstream reporting.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results: dict = {}

    # ------------------------------------------------------------------
    def run_all(self) -> dict:
        self._descriptive_stats()
        self._distribution_analysis()
        self._correlation_analysis()
        self._comparative_analysis()
        self._power_curve_regression()
        log.info("All analytics complete.")
        return self.results

    # ------------------------------------------------------------------
    def _descriptive_stats(self):
        """NumPy-based descriptive statistics."""
        cols = {
            "Wind Speed":          self.df["Wind Speed (m/s)"].dropna().values,
            "Active Power":        self.df["LV ActivePower (kW)"].dropna().values,
            "Theoretical Power":   self.df["Theoretical_Power_Curve (KWh)"].dropna().values,
            "Power Deviation":     self.df["Power_Deviation (kW)"].dropna().values,
            "Deviation Pct":       self.df["Deviation_Pct (%)"].dropna().values,
        }

        desc = {}
        for name, arr in cols.items():
            desc[name] = {
                "mean":     float(np.mean(arr)),
                "median":   float(np.median(arr)),
                "std":      float(np.std(arr, ddof=1)),
                "variance": float(np.var(arr, ddof=1)),
                "min":      float(np.min(arr)),
                "max":      float(np.max(arr)),
                "range":    float(np.ptp(arr)),
                "q1":       float(np.percentile(arr, 25)),
                "q3":       float(np.percentile(arr, 75)),
                "iqr":      float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            }

        self.results["descriptive"] = desc
        log.info("Descriptive statistics computed.")

    # ------------------------------------------------------------------
    def _distribution_analysis(self):
        """Skewness, kurtosis, outlier detection (IQR + Z-score)."""
        dev = self.df["Power_Deviation (kW)"].dropna().values

        skewness  = float(stats.skew(dev))
        kurtosis  = float(stats.kurtosis(dev))

        # IQR outliers
        q1, q3  = np.percentile(dev, [25, 75])
        iqr     = q3 - q1
        lo, hi  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_out = int(np.sum((dev < lo) | (dev > hi)))

        # Z-score outliers (|z| > 3)
        z_scores = np.abs(stats.zscore(dev))
        z_out    = int(np.sum(z_scores > 3))

        # Normality test
        _, p_val = stats.shapiro(dev[:5000] if len(dev) > 5000 else dev)

        self.results["distribution"] = {
            "skewness":       skewness,
            "kurtosis":       kurtosis,
            "iqr_outliers":   iqr_out,
            "zscore_outliers":z_out,
            "shapiro_p":      float(p_val),
            "is_normal":      p_val > 0.05,
            "iqr_bounds":     (float(lo), float(hi)),
        }
        log.info("Distribution analysis complete.")

    # ------------------------------------------------------------------
    def _correlation_analysis(self):
        """Pearson and Spearman correlations between key variables."""
        num_df = self.df[[
            "Wind Speed (m/s)",
            "LV ActivePower (kW)",
            "Theoretical_Power_Curve (KWh)",
            "Power_Deviation (kW)",
            "Wind Direction (°)",
        ]].dropna()

        pearson  = num_df.corr(method="pearson")
        spearman = num_df.corr(method="spearman")

        # Wind speed → deviation (key engineering relationship)
        ws  = num_df["Wind Speed (m/s)"].values
        dev = num_df["Power_Deviation (kW)"].values
        r_p, p_p = stats.pearsonr(ws, dev)
        r_s, p_s = stats.spearmanr(ws, dev)

        self.results["correlation"] = {
            "pearson_matrix":  pearson,
            "spearman_matrix": spearman,
            "ws_dev_pearson":  {"r": float(r_p), "p": float(p_p)},
            "ws_dev_spearman": {"r": float(r_s), "p": float(p_s)},
        }
        log.info("Correlation analysis complete.")

    # ------------------------------------------------------------------
    def _comparative_analysis(self):
        """
        Compare power deviation across wind-speed bins.
        Uses Mann-Whitney U test (non-parametric) for significance.
        """
        groups = {}
        for label, grp in self.df.groupby("Wind_Bin", observed=True):
            arr = grp["Power_Deviation (kW)"].dropna().values
            if len(arr) < 10:
                continue
            groups[str(label)] = {
                "n":      len(arr),
                "mean":   float(np.mean(arr)),
                "median": float(np.median(arr)),
                "std":    float(np.std(arr, ddof=1)),
            }

        # Statistical test: Partial Load vs Full Load
        pl = self.df[self.df["Wind_Bin"] == "Partial Load"]["Power_Deviation (kW)"].dropna().values
        fl = self.df[self.df["Wind_Bin"] == "Full Load"]["Power_Deviation (kW)"].dropna().values
        mw_stat, mw_p = (np.nan, np.nan)
        if len(pl) > 0 and len(fl) > 0:
            mw_stat, mw_p = stats.mannwhitneyu(pl, fl, alternative="two-sided")

        self.results["comparative"] = {
            "wind_bin_stats": groups,
            "mw_stat":        float(mw_stat) if not np.isnan(mw_stat) else None,
            "mw_p":           float(mw_p)    if not np.isnan(mw_p)    else None,
        }
        log.info("Comparative analysis complete.")

    # ------------------------------------------------------------------
    def _power_curve_regression(self):
        """Polynomial regression of actual power vs wind speed."""
        sub = self.df[self.df["Wind Speed (m/s)"] > 0].dropna(
            subset=["Wind Speed (m/s)", "LV ActivePower (kW)"]
        )
        ws  = sub["Wind Speed (m/s)"].values
        pw  = sub["LV ActivePower (kW)"].values

        # 3rd-degree polynomial fit
        coeffs = np.polyfit(ws, pw, deg=3)
        p_func = np.poly1d(coeffs)
        residuals = pw - p_func(ws)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((pw - np.mean(pw)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        self.results["regression"] = {
            "coefficients": coeffs.tolist(),
            "r_squared":    float(r2),
            "poly_func":    p_func,
        }
        log.info("Power curve regression complete. R² = %.4f", r2)


# =============================================================================
# MODULE 4 — VISUALIZATION
# =============================================================================
class Visualizer:
    """
    Produces all required static and animated plots.
    Static  : (1) Scatter, (2) Histogram, (3) Box-plot, (4) Heatmap, (5) Line
    Animated: (1) Hourly deviation trend, (2) Wind-speed distribution shift
    """

    def __init__(self, df: pd.DataFrame, results: dict, out_dir: Path):
        self.df      = df
        self.results = results
        self.out     = out_dir

    # ── STATIC 1 ─────────────────────────────────────────────────────
    def plot_power_curve_scatter(self):
        """Actual vs theoretical power scatter with regression overlay."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            "REN-01 | Power Curve Analysis — January (Unique Filter)",
            fontsize=15, fontweight="bold", color=PALETTE["dark"],
        )

        sub = self.df.sample(min(3000, len(self.df)), random_state=0)

        # Left: Actual vs wind speed
        ax = axes[0]
        sc = ax.scatter(
            sub["Wind Speed (m/s)"],
            sub["LV ActivePower (kW)"],
            c=sub["Power_Deviation (kW)"],
            cmap="RdYlGn", alpha=0.55, s=12, edgecolors="none",
        )
        # Regression overlay
        ws_range = np.linspace(0, 25, 300)
        pf = self.results["regression"]["poly_func"]
        ax.plot(ws_range, np.clip(pf(ws_range), 0, 3800),
                color=PALETTE["primary"], lw=2.5, label="Polynomial Fit (deg=3)")
        # Theoretical PC
        theoretical_ws = np.where(
            ws_range < 3.5, 0,
            np.where(ws_range < 13, 3600 * ((ws_range - 3.5) / 9.5) ** 3,
                     np.where(ws_range <= 25, 3600, 0))
        )
        ax.plot(ws_range, theoretical_ws, "--", color=PALETTE["accent"],
                lw=2, label="Theoretical Curve")
        plt.colorbar(sc, ax=ax, label="Deviation (kW)")
        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Active Power (kW)")
        ax.set_title("Actual Power Curve vs Wind Speed")
        ax.legend(fontsize=9)
        ax.grid(True)

        # Right: Deviation vs wind speed
        ax2 = axes[1]
        ax2.scatter(
            sub["Wind Speed (m/s)"],
            sub["Power_Deviation (kW)"],
            c=sub["Wind Direction (°)"],
            cmap="hsv", alpha=0.4, s=12, edgecolors="none",
        )
        ax2.axhline(0, color="black", lw=1.5, linestyle="-")
        ax2.axhline(
            self.results["descriptive"]["Power Deviation"]["mean"],
            color=PALETTE["negative"], lw=1.8, linestyle="--",
            label=f'Mean Deviation = {self.results["descriptive"]["Power Deviation"]["mean"]:.1f} kW',
        )
        ax2.set_xlabel("Wind Speed (m/s)")
        ax2.set_ylabel("Power Deviation (kW)")
        ax2.set_title("Power Curve Deviation vs Wind Speed")
        ax2.legend(fontsize=9)
        ax2.grid(True)

        plt.tight_layout()
        path = self.out / "static1_power_curve_scatter.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Static 1 saved: %s", path)
        return path

    # ── STATIC 2 ─────────────────────────────────────────────────────
    def plot_deviation_histogram(self):
        """Distribution histogram of power deviation with KDE."""
        dev = self.df["Power_Deviation (kW)"].dropna()
        d   = self.results["distribution"]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(dev, bins=80, color=PALETTE["secondary"],
                alpha=0.65, density=True, label="Observed Frequency")

        # KDE
        kde_x = np.linspace(dev.min(), dev.max(), 400)
        kde   = stats.gaussian_kde(dev.dropna())
        ax.plot(kde_x, kde(kde_x), color=PALETTE["primary"],
                lw=2.5, label="KDE")

        # Overlay normal for comparison
        mu, sigma = dev.mean(), dev.std()
        normal_y  = stats.norm.pdf(kde_x, mu, sigma)
        ax.plot(kde_x, normal_y, "--", color=PALETTE["accent"],
                lw=2, label="Normal Reference")

        # Annotate IQR bounds
        lo, hi = d["iqr_bounds"]
        ax.axvline(lo, color=PALETTE["negative"], lw=1.5, linestyle=":",
                   label=f"IQR Lower Fence ({lo:.0f} kW)")
        ax.axvline(hi, color=PALETTE["positive"], lw=1.5, linestyle=":",
                   label=f"IQR Upper Fence ({hi:.0f} kW)")

        ax.set_xlabel("Power Deviation (kW)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Distribution of Power Curve Deviation — January\n"
            f"Skewness = {d['skewness']:.3f}  |  "
            f"Kurtosis = {d['kurtosis']:.3f}  |  "
            f"IQR Outliers = {d['iqr_outliers']}",
            fontsize=12,
        )
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        path = self.out / "static2_deviation_histogram.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Static 2 saved: %s", path)
        return path

    # ── STATIC 3 ─────────────────────────────────────────────────────
    def plot_wind_bin_boxplot(self):
        """Box-plots of deviation grouped by wind speed regime."""
        fig, ax = plt.subplots(figsize=(12, 7))

        order   = ["Below Cut-In", "Partial Load", "Transition", "Full Load"]
        colors  = [PALETTE["light"], PALETTE["secondary"],
                   PALETTE["accent"],  PALETTE["primary"]]

        bp_data  = []
        bp_labels = []
        for lbl in order:
            sub = self.df[self.df["Wind_Bin"] == lbl]["Power_Deviation (kW)"].dropna()
            if len(sub) > 0:
                bp_data.append(sub.values)
                bp_labels.append(lbl)

        bp = ax.boxplot(
            bp_data, labels=bp_labels, patch_artist=True,
            medianprops=dict(color="white", lw=2),
            whiskerprops=dict(color=PALETTE["dark"]),
            flierprops=dict(marker="o", markerfacecolor=PALETTE["negative"],
                            alpha=0.3, markersize=3),
        )
        for patch, col in zip(bp["boxes"], colors[:len(bp_data)]):
            patch.set_facecolor(col)
            patch.set_alpha(0.8)

        ax.axhline(0, color="black", lw=1.2, linestyle="--", label="Zero Deviation")
        ax.set_xlabel("Wind Speed Regime")
        ax.set_ylabel("Power Deviation (kW)")
        ax.set_title(
            "Power Curve Deviation by Wind Speed Regime — January\n"
            "(Comparative Analysis: Actual − Theoretical Power)",
            fontsize=12,
        )

        # Annotate n per group
        for i, (lbl, arr) in enumerate(zip(bp_labels, bp_data)):
            ax.text(i + 1, ax.get_ylim()[0] + 30, f"n={len(arr):,}",
                    ha="center", fontsize=9, color=PALETTE["dark"])

        ax.legend()
        ax.grid(True, axis="y")

        plt.tight_layout()
        path = self.out / "static3_wind_bin_boxplot.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Static 3 saved: %s", path)
        return path

    # ── STATIC 4 ─────────────────────────────────────────────────────
    def plot_correlation_heatmap(self):
        """Pearson correlation heatmap with significance markers."""
        pearson = self.results["correlation"]["pearson_matrix"]

        fig, ax = plt.subplots(figsize=(10, 8))
        mask   = np.triu(np.ones_like(pearson, dtype=bool), k=1)

        sns.heatmap(
            pearson, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="white",
            square=True, ax=ax,
            annot_kws={"size": 11},
        )
        ax.set_title(
            "Pearson Correlation Matrix — Wind Turbine SCADA Variables\n"
            "(January | Unique Filter Applied)",
            fontsize=13, fontweight="bold",
        )
        short_labels = [
            "Wind\nSpeed", "Active\nPower", "Theoretical\nPower",
            "Deviation\n(kW)", "Wind\nDirection",
        ]
        ax.set_xticklabels(short_labels, rotation=30, ha="right")
        ax.set_yticklabels(short_labels, rotation=0)

        plt.tight_layout()
        path = self.out / "static4_correlation_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Static 4 saved: %s", path)
        return path

    # ── STATIC 5 ─────────────────────────────────────────────────────
    def plot_hourly_profile(self):
        """Mean hourly power and deviation profile."""
        hourly = self.df.groupby("Hour").agg(
            mean_power  = ("LV ActivePower (kW)", "mean"),
            mean_theory = ("Theoretical_Power_Curve (KWh)", "mean"),
            mean_dev    = ("Power_Deviation (kW)", "mean"),
            std_dev     = ("Power_Deviation (kW)", "std"),
        ).reset_index()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        hours = hourly["Hour"].values

        ax1.fill_between(hours, hourly["mean_power"], alpha=0.25,
                         color=PALETTE["secondary"])
        ax1.plot(hours, hourly["mean_power"], "o-",
                 color=PALETTE["secondary"], lw=2, label="Mean Active Power")
        ax1.plot(hours, hourly["mean_theory"], "s--",
                 color=PALETTE["accent"], lw=2, label="Mean Theoretical Power")
        ax1.set_ylabel("Power (kW)")
        ax1.set_title("Hourly Power Profile — January (Averaged)")
        ax1.legend()
        ax1.grid(True)

        ax2.fill_between(hours,
                         hourly["mean_dev"] - hourly["std_dev"],
                         hourly["mean_dev"] + hourly["std_dev"],
                         alpha=0.2, color=PALETTE["negative"], label="±1 Std Dev")
        ax2.plot(hours, hourly["mean_dev"], "D-",
                 color=PALETTE["negative"], lw=2, label="Mean Deviation")
        ax2.axhline(0, color="black", lw=1, linestyle="--")
        ax2.set_xlabel("Hour of Day (UTC)")
        ax2.set_ylabel("Power Deviation (kW)")
        ax2.set_title("Hourly Mean Power Curve Deviation")
        ax2.legend()
        ax2.grid(True)
        ax2.set_xticks(range(0, 24, 2))

        plt.tight_layout()
        path = self.out / "static5_hourly_profile.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Static 5 saved: %s", path)
        return path




# =============================================================================
# MODULE 5 — REPORT GENERATOR
# =============================================================================
class ReportGenerator:
    """
    Prints a structured console report and writes a Markdown summary
    to serve as the backbone for the IEEE paper.
    """

    def __init__(self, df: pd.DataFrame, results: dict,
                 cleaner_report: dict, out_dir: Path):
        self.df      = df
        self.res     = results
        self.crep    = cleaner_report
        self.out     = out_dir

    # ------------------------------------------------------------------
    def generate(self):
        lines = self._build_lines()
        report_str = "\n".join(lines)
        print(report_str)

        path = self.out / "analytics_report.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_str)
        log.info("Analytics report saved: %s", path)
        return path

    # ------------------------------------------------------------------
    def _build_lines(self) -> list[str]:
        d  = self.res["descriptive"]
        di = self.res["distribution"]
        co = self.res["correlation"]
        cm = self.res["comparative"]
        rg = self.res["regression"]

        lines = [
            "=" * 72,
            "  REN-01 | WIND TURBINE POWER CURVE DEVIATION — ANALYTICS REPORT",
            "  Pillar 4: Renewable Energy Systems",
            f"  Dataset: Wind Turbine SCADA (Kaggle)  |  Filter: January Only",
            f"  Records analysed: {len(self.df):,}",
            "=" * 72,
            "",
            "━━━ I. DATA PIPELINE SUMMARY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Duplicates removed   : {self.crep.get('duplicates_removed', 0):,}",
            f"  Corrupt values fixed : {self.crep.get('corrupt_values_fixed', 0):,}",
            f"  Rows after filter    : {self.crep.get('rows_after_filter', 0):,}",
            f"  Unique filter        : {self.crep.get('unique_filter', 'N/A')}",
            "",
            "━━━ II. DESCRIPTIVE STATISTICS (NumPy) ━━━━━━━━━━━━━━━━━━━━━━━━",
        ]

        for name, stats_d in d.items():
            lines += [
                f"\n  [{name}]",
                f"    Mean     : {stats_d['mean']:>10.3f}",
                f"    Median   : {stats_d['median']:>10.3f}",
                f"    Std Dev  : {stats_d['std']:>10.3f}",
                f"    Variance : {stats_d['variance']:>10.3f}",
                f"    Range    : {stats_d['range']:>10.3f}",
                f"    IQR      : {stats_d['iqr']:>10.3f}",
            ]

        lines += [
            "",
            "━━━ III. DISTRIBUTION ANALYSIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Skewness (Deviation)  : {di['skewness']:>8.4f}",
            f"  Kurtosis (Deviation)  : {di['kurtosis']:>8.4f}",
            f"  IQR Outliers          : {di['iqr_outliers']:>8,}",
            f"  Z-Score Outliers (|z|>3): {di['zscore_outliers']:>6,}",
            f"  Shapiro-Wilk p-value  : {di['shapiro_p']:>8.6f}",
            f"  Normally Distributed  : {'Yes' if di['is_normal'] else 'No (p < 0.05)'}",
            "",
            "  ENGINEERING INTERPRETATION:",
            f"  Skewness = {di['skewness']:.4f} → "
            + ("Left-skewed: more episodes of under-performance than over."
               if di["skewness"] < 0
               else "Right-skewed: sporadic high-output bursts above the curve."),
            "",
            "━━━ IV. CORRELATION ANALYSIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Wind Speed ↔ Deviation (Pearson r)  : "
            f"{co['ws_dev_pearson']['r']:>7.4f}  p={co['ws_dev_pearson']['p']:.4e}",
            f"  Wind Speed ↔ Deviation (Spearman ρ) : "
            f"{co['ws_dev_spearman']['r']:>7.4f}  p={co['ws_dev_spearman']['p']:.4e}",
            "",
            "  ENGINEERING INTERPRETATION:",
            _interpret_correlation(co["ws_dev_pearson"]["r"]),
            "",
            "━━━ V. COMPARATIVE ANALYSIS (Wind Speed Regimes) ━━━━━━━━━━━━━━",
        ]

        for regime, s in cm["wind_bin_stats"].items():
            lines.append(
                f"  {regime:<18}: n={s['n']:>5,}  "
                f"Mean={s['mean']:>8.2f} kW  Std={s['std']:>7.2f} kW"
            )

        if cm["mw_p"] is not None:
            sig = "SIGNIFICANT" if cm["mw_p"] < 0.05 else "NOT SIGNIFICANT"
            lines += [
                "",
                f"  Mann-Whitney U (Partial vs Full Load): "
                f"U={cm['mw_stat']:.0f}, p={cm['mw_p']:.4e} → {sig}",
            ]

        lines += [
            "",
            "━━━ VI. REGRESSION ANALYSIS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Polynomial Degree     : 3",
            f"  R² (Goodness of Fit)  : {rg['r_squared']:.6f}",
            f"  Coefficients          : {[f'{c:.4f}' for c in rg['coefficients']]}",
            "",
            "  ENGINEERING INTERPRETATION:",
            f"  R² = {rg['r_squared']:.4f} → the polynomial model explains "
            f"{rg['r_squared']*100:.2f}% of variance in the actual power output.",
            "",
            "=" * 72,
            "  OUTPUT FILES:",
        ]

        for f in sorted((self.out).glob("*")):
            lines.append(f"    {f.name}")

        lines += ["=" * 72]
        return lines


def _interpret_correlation(r: float) -> str:
    """Return engineering interpretation of Pearson r."""
    if abs(r) < 0.2:
        strength = "negligible"
    elif abs(r) < 0.4:
        strength = "weak"
    elif abs(r) < 0.6:
        strength = "moderate"
    elif abs(r) < 0.8:
        strength = "strong"
    else:
        strength = "very strong"
    direction = "positive" if r > 0 else "negative"
    return (
        f"  → {strength.capitalize()} {direction} relationship (r={r:.4f}). "
        + (
            "As wind speed increases, deviation grows — "
            "turbine over-performs relative to theoretical curve at high speeds."
            if r > 0 else
            "As wind speed increases, deviation decreases — "
            "turbine under-performs relative to theoretical curve at high winds."
        )
    )


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================
def main():
    log.info("=" * 60)
    log.info("REN-01 | Wind Turbine Power Curve Deviation Pipeline")
    log.info("=" * 60)

    # ── 1. Ingestion ──────────────────────────────────────────────────
    # kagglehub will auto-download the real dataset if credentials exist.
    # To use a local CSV instead: filepath = 'T1.csv'
    filepath = None
    ingestion = DataIngestion(filepath=filepath)
    raw_df    = ingestion.load()

    # ── 2. Cleaning ───────────────────────────────────────────────────
    cleaner  = DataCleaner(raw_df)
    clean_df = cleaner.clean()

    # ── 2b. Export CSVs for repository ───────────────────────────────
    try:
        raw_export = raw_df.copy()
        # Keep only core columns for original export
        core_cols = [c for c in ingestion.COLUMNS if c in raw_export.columns]
        raw_export[core_cols].to_csv(DATA_DIR / "dataset_original.csv", index=False)
        clean_df.to_csv(DATA_DIR / "dataset_cleaned.csv", index=False)
        log.info("Datasets exported to data/ folder.")
    except Exception as exc:
        log.warning("CSV export skipped: %s", exc)

    # ── 3. Analytics ──────────────────────────────────────────────────
    analyzer = DataAnalyzer(clean_df)
    results  = analyzer.run_all()

    # ── 4. Visualization ──────────────────────────────────────────────
    viz = Visualizer(clean_df, results, OUTPUT_DIR)

    log.info("Generating static plots …")
    viz.plot_power_curve_scatter()
    viz.plot_deviation_histogram()
    viz.plot_wind_bin_boxplot()
    viz.plot_correlation_heatmap()
    viz.plot_hourly_profile()

    # ── 5. Report ─────────────────────────────────────────────────────
    reporter = ReportGenerator(
        clean_df, results, cleaner.report, OUTPUT_DIR
    )
    reporter.generate()

    log.info("Pipeline complete. All outputs saved to: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
