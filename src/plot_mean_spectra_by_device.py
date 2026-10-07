"""Quick check: mean fluorescence spectrum per spectrometer device in patients_database.xlsx."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

df = pd.read_excel(os.path.join(DATA, "patients_database.xlsx"), sheet_name="patients", header=0)

all_wave_cols = df.columns[70:582]
all_waves = np.array([float(str(w).replace(",", ".")) for w in all_wave_cols])

# Common analysis window (matches the legacy 150-point grid used elsewhere in this
# project, notebooks/sfm25_preprocessing.ipynb: df.columns[-221:-221+150], which spans
# 401.73-682.67 nm): all three devices report real (non-NaN) signal in this band, so
# it is the range that is actually comparable across devices.
WAVE_MIN, WAVE_MAX = 401.7, 682.7
band_mask = (all_waves >= WAVE_MIN) & (all_waves <= WAVE_MAX)
wave_cols = all_wave_cols[band_mask]
waves = all_waves[band_mask]
spectra = df[wave_cols].apply(pd.to_numeric, errors="coerce")

device_labels = {
    "Spectometer: Подключены - контроллер: USB24A_PG_001; Спектрометр: STS s/n S07279 b/n 1200 API v 2.46": "STS S07279",
    "Подключены: |USB24A_PG_001|Спектрометр:Optosky|ATP10100000|s/n 25081X37ABX23233": "Optosky 25081X37ABX23233",
    "Подключены: |USB24A_PG_001|Спектрометр:Optosky|ATP10100000|s/n 24361X39ABX23130": "Optosky 24361X39ABX23130",
}

fig, ax = plt.subplots(figsize=(11, 6))
for device, g in df.groupby("Оборудование"):
    sub = spectra.loc[g.index]
    mean = sub.mean(axis=0, skipna=True)
    sd = sub.std(axis=0, skipna=True)
    n = sub.notna().any(axis=1).sum()
    label = device_labels.get(device, device[:40])
    ax.plot(waves, mean.values, label=f"{label} (N={n})", linewidth=1.5)
    ax.fill_between(waves, (mean - sd).values, (mean + sd).values, alpha=0.15)

ax.set_xlabel("Wavelength, nm")
ax.set_ylabel("Intensity, a.u. (mean ± SD)")
ax.set_title(f"Mean spectra by spectrometer device — patients_database.xlsx\n"
             f"({WAVE_MIN:.0f}-{WAVE_MAX:.0f} nm, {len(waves)} points, common window)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()

out_path = os.path.join(DATA, "..", "mean_spectra_by_device.png")
out_path = os.path.abspath(out_path)
fig.savefig(out_path, dpi=150)
print("saved:", out_path)
