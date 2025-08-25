"""
Raman Viewer - Streamlit app for viewing & preprocessing Raman spectra
"""

from __future__ import annotations

# ---- stdlib
import io
import json
import zipfile
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

# ---- third-party
import numpy as np
import pandas as pd
import streamlit as st
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve

try:
    import plotly.graph_objects as go
except Exception:
    go = None

# ----------------------------
# Parsing & utilities
# ----------------------------

X_CANDIDATE_NAMES = {"x", "wavenumber", "raman_shift", "raman shift", "wavelength", "cm-1"}
Y_CANDIDATE_NAMES = {"y", "intensity", "counts", "a.u.", "a.u", "signal"}


def _strip_esp_header(file_bytes: bytes, lines_to_skip: int) -> bytes:
    """Drop first N lines (instrument params) for .esp files."""
    try:
        text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        text = file_bytes.decode("cp1251", errors="ignore")
    lines = text.splitlines()
    if len(lines) > lines_to_skip:
        text = "\n".join(lines[lines_to_skip:])
    else:
        text = "\n".join(lines[1:]) if len(lines) >= 2 else ""
    return text.encode("utf-8")


def _read_text_df(file_bytes: bytes, filename: Optional[str] = None, esp_skip_lines: int = 2) -> pd.DataFrame:
    """Read a 2-column spectrum robustly (.txt/.csv/.esp). For .esp skip header lines."""
    if filename and filename.lower().endswith(".esp"):
        file_bytes = _strip_esp_header(file_bytes, esp_skip_lines)

    attempts = [
        {"sep": ",", "decimal": ".", "header": 0},
        {"sep": ";", "decimal": ".", "header": 0},
        {"sep": "\t", "decimal": ".", "header": 0},
        {"sep": None, "decimal": ".", "header": None},  # whitespace
        {"sep": ",", "decimal": ",", "header": 0},
        {"sep": ";", "decimal": ",", "header": 0},
        {"sep": "\t", "decimal": ",", "header": 0},
        {"sep": None, "decimal": ",", "header": None},
    ]

    for opt in attempts:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), **opt, engine="python")
            if opt.get("header") is None:
                if df.shape[1] < 2:
                    continue
                df = df.iloc[:, :2].copy()
                df.columns = ["x", "y"]
            else:
                cols_norm = {c: str(c).strip().lower() for c in df.columns}
                x_col, y_col = None, None
                for c, norm in cols_norm.items():
                    if any(key in norm for key in X_CANDIDATE_NAMES):
                        x_col = c
                        break
                for c, norm in cols_norm.items():
                    if any(key in norm for key in Y_CANDIDATE_NAMES):
                        y_col = c
                        break
                if x_col is None or y_col is None:
                    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                    if len(num_cols) >= 2:
                        x_col, y_col = num_cols[:2]
                    else:
                        df2 = df.apply(pd.to_numeric, errors="coerce")
                        num_cols = [c for c in df2.columns if df2[c].notna().sum() > 0]
                        if len(num_cols) >= 2:
                            x_col, y_col = num_cols[:2]
                            df = df2
                        else:
                            continue
                df = df[[x_col, y_col]].rename(columns={x_col: "x", y_col: "y"})
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            df = df.drop_duplicates(subset=["x"]).sort_values("x")
            if df.shape[0] >= 5 and df["y"].abs().sum() > 0:
                return df
        except Exception:
            continue
    raise ValueError("Не удалось распознать формат файла (нужны 2 числовые колонки: x и y).")


# ----------------------------
# Preprocessing
# ----------------------------

@dataclass
class Settings:
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    resample_step: Optional[float] = None          # disabled by default
    invert_x: bool = True                           # show axis 4000->0 when True

    # S-G smoothing (off by default)
    smooth_enable: bool = False
    smooth_window: int = 11
    smooth_poly: int = 3

    # AsLS baseline (off by default)
    baseline_enable: bool = False
    baseline_lam: float = 1e5
    baseline_p: float = 0.001
    baseline_iter: int = 10

    # Normalization (off by default)
    norm_mode: str = "none"                         # none|max|area|vector

    # Peaks
    peaks_enable: bool = False
    peaks_prominence: float = 0.02                  # relative to max(y)
    peaks_distance: int = 5                         # points

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(s: str) -> "Settings":
        return Settings(**json.loads(s))


def asls_baseline(y: np.ndarray, lam: float, p: float, niter: int) -> np.ndarray:
    """Asymmetric Least Squares baseline (Eilers & Boelens)."""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * (D @ D.T)
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def resample_uniform(x: np.ndarray, y: np.ndarray, step: float) -> Tuple[np.ndarray, np.ndarray]:
    if step is None or step <= 0:
        return x, y
    new_x = np.arange(float(np.nanmin(x)), float(np.nanmax(x)) + step, step)
    new_y = np.interp(new_x, x, y)
    return new_x, new_y


def normalize(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return y
    if mode == "max":
        m = np.nanmax(np.abs(y))
        return y / m if m else y
    if mode == "area":
        a = np.trapz(np.abs(y))
        return y / a if a else y
    if mode == "vector":
        n = np.linalg.norm(y)
        return y / n if n else y
    return y


@dataclass
class Spectrum:
    name: str
    raw: pd.DataFrame
    processed: Optional[pd.DataFrame] = None

    def apply(self, s: Settings) -> None:
        df = self.raw.copy()
        x = df["x"].to_numpy(float)
        y = df["y"].to_numpy(float)

        # Crop
        if s.x_min is not None:
            mask_min = x >= s.x_min
        else:
            mask_min = np.ones_like(x, bool)
        if s.x_max is not None:
            mask_max = x <= s.x_max
        else:
            mask_max = np.ones_like(x, bool)
        m = mask_min & mask_max
        x, y = x[m], y[m]

        # Resample (off by default)
        if s.resample_step and s.resample_step > 0 and len(x) > 1:
            x, y = resample_uniform(x, y, s.resample_step)

        # Smoothing (off by default)
        if s.smooth_enable and len(y) > s.smooth_window:
            win = s.smooth_window + (1 - s.smooth_window % 2)  # ensure odd
            try:
                y = signal.savgol_filter(y, int(win), int(s.smooth_poly))
            except Exception:
                pass

        # Baseline (off by default)
        if s.baseline_enable and len(y) > 5:
            try:
                base = asls_baseline(y, s.baseline_lam, s.baseline_p, s.baseline_iter)
                y = y - base
            except Exception:
                pass

        # Normalization (off by default)
        y = normalize(y, s.norm_mode)

        # Do NOT reverse arrays here — axis direction is handled in Plotly layout
        self.processed = pd.DataFrame({"x": x, "y": y})


# ----------------------------
# UI
# ----------------------------

st.set_page_config(page_title="Raman Viewer", layout="wide")
st.title("Raman Viewer - viewing and preprocessing of spectra")

with st.sidebar:
    st.header("Settings")

    # Range
    st.subheader("Range (cm-1)")
    x_min = st.number_input("Min", value=200.0, step=10.0)
    x_max = st.number_input("Max", value=4000.0, step=10.0)
    resample_step = st.number_input("Resample step (cm-1)", value=0.0, min_value=0.0, step=0.5)
    invert_x = st.checkbox("Invert X (high->low)", value=False)  # controls axis orientation only

    # Smoothing
    st.subheader("Smoothing (Savitzky-Golay)")
    smooth_enable = st.checkbox("Enable smoothing", value=False)
    smooth_window = st.slider("Window", min_value=5, max_value=101, value=11, step=2)
    smooth_poly = st.slider("Polyorder", min_value=1, max_value=5, value=3, step=1)

    # Baseline
    st.subheader("Baseline (AsLS)")
    baseline_enable = st.checkbox("Enable baseline", value=False)
    baseline_lam = st.number_input("lambda (stiffness)", value=1e5, format="%.0f")
    baseline_p = st.number_input("p (asymmetry)", value=0.001, min_value=0.0001, max_value=0.05, step=0.0001, format="%.4f")
    baseline_iter = st.number_input("Iterations", value=10, min_value=1, step=1)

    # Normalization
    st.subheader("Normalization")
    norm_mode = st.selectbox("Mode", ["none", "max", "area", "vector"], index=0)

    # Peaks
    st.subheader("Peaks")
    peaks_enable = st.checkbox("Show peaks", value=False)
    peaks_prom_rel = st.slider("Prominence (rel)", min_value=0.0, max_value=1.0, value=0.02, step=0.01)
    peaks_distance = st.number_input("Min distance (points)", value=5, min_value=1, step=1)

    # ESP
    st.subheader("ESP import")
    esp_skip_lines = st.number_input("ESP: header lines to skip", value=2, min_value=0, step=1)

    # Pack settings
    settings = Settings(
        x_min=x_min,
        x_max=x_max,
        resample_step=float(resample_step) if resample_step > 0 else None,
        invert_x=invert_x,
        smooth_enable=smooth_enable,
        smooth_window=int(smooth_window),
        smooth_poly=int(smooth_poly),
        baseline_enable=baseline_enable,
        baseline_lam=float(baseline_lam),
        baseline_p=float(baseline_p),
        baseline_iter=int(baseline_iter),
        norm_mode=norm_mode,
        peaks_enable=peaks_enable,
        peaks_prominence=float(peaks_prom_rel),
        peaks_distance=int(peaks_distance),
    )

st.markdown("Upload one or more .txt/.csv/.esp files.")

uploaded_files = st.file_uploader("Spectra files", type=["txt", "csv", "esp"], accept_multiple_files=True)

spectra: List[Spectrum] = []
if uploaded_files:
    for f in uploaded_files:
        try:
            raw = _read_text_df(f.getvalue(), f.name, esp_skip_lines=int(esp_skip_lines))
            spectra.append(Spectrum(f.name, raw))
        except Exception as e:
            st.warning(f"{f.name}: {e}")

if not spectra:
    st.info("Upload files to start.")
    st.stop()

# Apply preprocessing
for sp in spectra:
    sp.apply(settings)

# Plotting
if go is None:
    st.error("Plotly not installed. Run: pip install plotly")
else:
    fig = go.Figure()
    peak_table_rows = []

    for sp in spectra:
        dfp = sp.processed
        if dfp is None or dfp.empty:
            continue
        x = dfp["x"].to_numpy()
        y = dfp["y"].to_numpy()

        peaks_idx = []
        if settings.peaks_enable:
            try:
                ymax = float(np.nanmax(y)) if y.size else 1.0
                abs_prom = settings.peaks_prominence * (ymax if ymax else 1.0)
                peaks_idx, _ = signal.find_peaks(y, prominence=abs_prom, distance=settings.peaks_distance)
            except Exception:
                peaks_idx = []
            if len(peaks_idx):
                fig.add_trace(
                    go.Scatter(
                        x=x[peaks_idx],
                        y=y[peaks_idx],
                        mode="markers",
                        name=f"peaks: {sp.name}",
                        marker=dict(size=6),
                    )
                )
                for pi in peaks_idx:
                    peak_table_rows.append({"file": sp.name, "x": float(x[pi]), "y": float(y[pi])})

        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=sp.name))

    # --- axis orientation controlled here ---
    fig.update_layout(
        xaxis=dict(
            title="Raman shift, cm-1",
            autorange="reversed" if settings.invert_x else True
        ),
        yaxis_title=("Intensity (norm.)" if settings.norm_mode != "none" else "Intensity"),
        template="plotly_white",
        legend_title_text="Files",
        height=550,
        dragmode="pan",
        uirevision="keep-zoom",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": [
                "zoomIn2d",
                "zoomOut2d",
                "pan2d",
                "zoom2d",
                "autoScale2d",
                "resetScale2d",
            ],
        },
    )

    if settings.peaks_enable and peak_table_rows:
        st.subheader("Detected peaks")
        df_peaks = pd.DataFrame(peak_table_rows)
        st.dataframe(df_peaks, use_container_width=True)
        csv_peaks = df_peaks.to_csv(index=False).encode("utf-8")
        st.download_button("Download peaks (CSV)", csv_peaks, file_name="peaks.csv", mime="text/csv")

# Preview tables & exports
st.subheader("Processed preview")
tabs = st.tabs([sp.name for sp in spectra])
for tab, sp in zip(tabs, spectra):
    with tab:
        st.dataframe(sp.processed, use_container_width=True)
        csv = sp.processed.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, file_name=f"{sp.name}_processed.csv", mime="text/csv")

# ZIP export
with io.BytesIO() as bio:
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        for sp in spectra:
            if sp.processed is not None:
                zf.writestr(f"{sp.name}_processed.csv", sp.processed.to_csv(index=False))
    zip_bytes = bio.getvalue()

st.download_button("Download all (ZIP)", zip_bytes, "processed_spectra.zip", "application/zip")
