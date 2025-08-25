# Raman Viewer

An interactive web application for viewing and preprocessing Raman spectra (Streamlit + Plotly).

## Features
- Load spectra (`.txt`, `.csv`, `.esp`), automatic delimiter/decimal detection
- Interactive visualization (mouse wheel zoom, pan, modebar buttons)
- Preprocessing:
  - cropping by Raman shift range (cm⁻¹)
  - resampling to a uniform grid
  - smoothing (Savitzky–Golay)
  - baseline correction (AsLS)
  - normalization (max, area, vector)
- Peak detection (`scipy.signal.find_peaks`)
- Export processed spectra (CSV / ZIP)
- Save/Load settings (JSON)

---

## Run with Docker

1. Install [Docker](https://docs.docker.com/get-docker/).
2. Pull the prebuilt image from GitHub Container Registry:
   ```bash
   docker pull ghcr.io/sl0n77/raman_viewer:latest
