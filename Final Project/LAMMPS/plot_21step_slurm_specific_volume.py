#!/usr/bin/env python3
"""
plot_21step_slurm_specific_volume.py

Parse independent 21-step LAMMPS runs from SLURM output files in folders like:
    500K/, 600K/, ..., 1300K/

For each folder, the script:
  1. finds the SLURM/LAMMPS output file
  2. extracts the final Step 21 thermo block
  3. averages density over the last 400 ps of Step 21
  4. computes specific volume = 1 / density
  5. plots specific volume vs temperature
  6. fits:
       low-T  = 500–800 K
       high-T = 950–1300 K

Plot style:
- black points with error bars
- no line connecting data points
- blue fit lines
- low fit line drawn from 400 K to intersection
- high fit line drawn from a little before intersection to a little past 1300 K
- R^2 values and Tg displayed on the plot
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="Root directory containing temperature folders like 500K/")
    p.add_argument("--tail-ps", type=float, default=400.0,
                   help="How many ps from the end of Step 21 to average (default: 400 ps)")
    p.add_argument("--dt-fs", type=float, default=1.0,
                   help="LAMMPS timestep in fs used in Step 21 (default: 1.0 fs)")
    p.add_argument("--outcsv", default="specific_volume_from_21step_slurm.csv",
                   help="Output CSV summary filename")
    p.add_argument("--outpng", default="specific_volume_from_21step_slurm.png",
                   help="Output PNG plot filename")
    p.add_argument("--title", default="PIM-1",
                   help="Text shown in upper-left of plot (default: PIM-1)")
    return p.parse_args()


def extract_temp_from_dir(dirname):
    m = re.match(r"(\d+(?:\.\d+)?)K$", os.path.basename(dirname))
    if not m:
        return None
    return float(m.group(1))


def find_slurm_file(temp_dir):
    candidates = sorted(glob.glob(os.path.join(temp_dir, "slurm*.out")))
    if candidates:
        return candidates[0]

    candidates = sorted(glob.glob(os.path.join(temp_dir, "*.out")))
    if candidates:
        return candidates[0]

    for path in sorted(glob.glob(os.path.join(temp_dir, "*"))):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", errors="ignore") as fh:
                chunk = fh.read(5000)
            if "LAMMPS (" in chunk:
                return path
        except Exception:
            pass

    return None


def is_numeric_thermo_row(line):
    s = line.strip()
    if not s:
        return False
    parts = s.split()
    if len(parts) < 5:
        return False
    if not re.fullmatch(r"[+-]?\d+", parts[0]):
        return False
    for tok in parts[1:5]:
        if not re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?", tok):
            return False
    return True


def parse_last_step21_block(path):
    with open(path, "r", errors="ignore") as fh:
        lines = fh.readlines()

    zero_step_indices = [
        i for i, line in enumerate(lines)
        if "Current step" in line and re.search(r"Current step\s*:\s*0\b", line)
    ]
    if not zero_step_indices:
        raise RuntimeError(f"No 'Current step : 0' found in {path}")

    start_search = zero_step_indices[-1]

    header_idx = None
    for i in range(start_search, min(start_search + 50, len(lines))):
        if re.match(r"\s*Step\s+Temp\s+Press\s+Volume\s+Density", lines[i]):
            header_idx = i
            break

    if header_idx is None:
        raise RuntimeError(f"Could not find thermo header after final 'Current step : 0' in {path}")

    data_rows = []
    for i in range(header_idx + 1, len(lines)):
        line = lines[i]
        if line.startswith("Loop time of"):
            break
        if is_numeric_thermo_row(line):
            parts = line.split()
            step = int(parts[0])
            temp = float(parts[1])
            press = float(parts[2])
            volume = float(parts[3])
            density = float(parts[4])
            data_rows.append((step, temp, press, volume, density))

    if not data_rows:
        raise RuntimeError(f"No thermo data rows found for final Step 21 block in {path}")

    return np.array(data_rows, dtype=float)


def linear_fit(x, y):
    m, b = np.polyfit(x, y, 1)
    return m, b


def calc_r2(x, y, m, b):
    y_pred = m * x + b
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if abs(ss_tot) < 1e-15:
        return np.nan
    return 1.0 - ss_res / ss_tot


def main():
    args = parse_args()

    fit_low = (500.0, 800.0)
    fit_high = (1200.0, 1300.0)

    temp_dirs = sorted(glob.glob(os.path.join(args.root, "*K")))
    temp_dirs = [d for d in temp_dirs if os.path.isdir(d)]

    if not temp_dirs:
        print("[ERROR] No temperature folders found.", flush=True)
        sys.exit(1)

    tail_steps = int(round(args.tail_ps * 1000.0 / args.dt_fs))
    step_threshold = 800000 - tail_steps

    print(f"[INFO] Averaging over last {args.tail_ps:.1f} ps of Step 21", flush=True)
    print(f"[INFO] Using Step 21 step threshold: step >= {step_threshold}", flush=True)
    print(f"[INFO] Low-T fit range: {fit_low[0]} to {fit_low[1]} K", flush=True)
    print(f"[INFO] High-T fit range: {fit_high[0]} to {fit_high[1]} K", flush=True)

    rows = []

    for d in temp_dirs:
        T_nom = extract_temp_from_dir(d)
        if T_nom is None:
            continue

        slurm_file = find_slurm_file(d)
        if slurm_file is None:
            print(f"[WARN] No SLURM/LAMMPS output file found in {d}", flush=True)
            continue

        try:
            arr = parse_last_step21_block(slurm_file)
        except Exception as e:
            print(f"[WARN] Failed parsing {slurm_file}: {e}", flush=True)
            continue

        step = arr[:, 0]
        temp = arr[:, 1]
        press = arr[:, 2]
        volume = arr[:, 3]
        density = arr[:, 4]

        mask = step >= step_threshold
        if np.sum(mask) < 2:
            print(f"[WARN] Not enough Step 21 data in tail window for {d}", flush=True)
            continue

        mean_temp = np.mean(temp[mask])
        mean_press = np.mean(press[mask])
        mean_volume = np.mean(volume[mask])
        mean_density = np.mean(density[mask])
        std_density = np.std(density[mask], ddof=1) if np.sum(mask) > 1 else 0.0

        specific_volume = 1.0 / mean_density
        specific_volume_std = std_density / (mean_density ** 2) if mean_density > 0 else np.nan

        rows.append({
            "directory": os.path.basename(d),
            "slurm_file": os.path.basename(slurm_file),
            "T_nominal_K": T_nom,
            "T_avg_K": mean_temp,
            "P_avg_atm": mean_press,
            "V_avg_A3": mean_volume,
            "rho_avg_gcm3": mean_density,
            "rho_std_gcm3": std_density,
            "vspec_avg_cm3g": specific_volume,
            "vspec_std_cm3g": specific_volume_std,
            "n_points": int(np.sum(mask)),
        })

        print(
            f"[INFO] {os.path.basename(d)} | "
            f"rho = {mean_density:.6f} g/cm^3 | "
            f"vspec = {specific_volume:.6f} cm^3/g | "
            f"points = {np.sum(mask)}",
            flush=True
        )

    if not rows:
        print("[ERROR] No valid data extracted.", flush=True)
        sys.exit(1)

    rows = sorted(rows, key=lambda r: r["T_nominal_K"])

    with open(args.outcsv, "w") as fh:
        fh.write("directory,slurm_file,T_nominal_K,T_avg_K,P_avg_atm,V_avg_A3,rho_avg_gcm3,rho_std_gcm3,vspec_avg_cm3g,vspec_std_cm3g,n_points\n")
        for r in rows:
            fh.write(
                f"{r['directory']},{r['slurm_file']},{r['T_nominal_K']:.6f},{r['T_avg_K']:.6f},"
                f"{r['P_avg_atm']:.6f},{r['V_avg_A3']:.6f},{r['rho_avg_gcm3']:.8f},{r['rho_std_gcm3']:.8f},"
                f"{r['vspec_avg_cm3g']:.8f},{r['vspec_std_cm3g']:.8f},{r['n_points']}\n"
            )

    print(f"[INFO] Wrote CSV summary: {args.outcsv}", flush=True)

    T = np.array([r["T_nominal_K"] for r in rows])
    vspec = np.array([r["vspec_avg_cm3g"] for r in rows])
    vspec_err = np.array([r["vspec_std_cm3g"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(
        T, vspec, yerr=vspec_err,
        fmt='o',
        linestyle='none',
        color='black',
        ecolor='black',
        ms=5,
        lw=1.2,
        capsize=3
    )

    # Low fit
    low_mask = (T >= fit_low[0]) & (T <= fit_low[1])
    if np.sum(low_mask) >= 2:
        m_low, b_low = linear_fit(T[low_mask], vspec[low_mask])
        r2_low = calc_r2(T[low_mask], vspec[low_mask], m_low, b_low)
        print(f"[INFO] Low-T fit: vspec = {m_low:.8e} * T + {b_low:.8e}", flush=True)
        print(f"[INFO] Low-T R^2 = {r2_low:.6f}", flush=True)
    else:
        print("[WARN] Not enough points for low-T fit.", flush=True)
        m_low = b_low = r2_low = None

    # High fit
    high_mask = (T >= fit_high[0]) & (T <= fit_high[1])
    if np.sum(high_mask) >= 2:
        m_high, b_high = linear_fit(T[high_mask], vspec[high_mask])
        r2_high = calc_r2(T[high_mask], vspec[high_mask], m_high, b_high)
        print(f"[INFO] High-T fit: vspec = {m_high:.8e} * T + {b_high:.8e}", flush=True)
        print(f"[INFO] High-T R^2 = {r2_high:.6f}", flush=True)
    else:
        print("[WARN] Not enough points for high-T fit.", flush=True)
        m_high = b_high = r2_high = None

    Tg = None

    if m_low is not None and m_high is not None and abs(m_low - m_high) > 1e-15:
        Tg = (b_high - b_low) / (m_low - m_high)
        print(f"[INFO] Estimated Tg from line intersection: {Tg:.3f} K", flush=True)

        # Low line: 400 K to Tg
        xfit_low = np.linspace(400.0, Tg, 200)
        yfit_low = m_low * xfit_low + b_low
        ax.plot(xfit_low, yfit_low, color='blue', lw=1.2)

        # High line: little before Tg to a little past 1300 K
        xfit_high = np.linspace(Tg - 40.0, 1325.0, 200)
        yfit_high = m_high * xfit_high + b_high
        ax.plot(xfit_high, yfit_high, color='blue', lw=1.2)

    else:
        if m_low is not None:
            xfit_low = np.linspace(400.0, 1100.0, 200)
            yfit_low = m_low * xfit_low + b_low
            ax.plot(xfit_low, yfit_low, color='blue', lw=1.2)

        if m_high is not None:
            xfit_high = np.linspace(900.0, 1325.0, 200)
            yfit_high = m_high * xfit_high + b_high
            ax.plot(xfit_high, yfit_high, color='blue', lw=1.2)

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r"Specific volume [cm$^3$ g$^{-1}$]")

    ax.text(0.03, 0.97, args.title, transform=ax.transAxes,
            ha='left', va='top', fontsize=13)

    stats_lines = []
    if r2_low is not None:
        stats_lines.append(rf"Low $R^2$ = {r2_low:.3f}")
    if r2_high is not None:
        stats_lines.append(rf"High $R^2$ = {r2_high:.3f}")
    if Tg is not None:
        stats_lines.append(rf"$T_g$ = {Tg:.1f} K")

    if stats_lines:
        stats_text = "\n".join(stats_lines)
        ax.text(
            0.62, 0.18, stats_text,
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8)
        )

    ax.grid(False)
    ax.set_xlim(400, 1400)

    fig.tight_layout()
    fig.savefig(args.outpng, dpi=300)
    print(f"[INFO] Wrote plot: {args.outpng}", flush=True)


if __name__ == "__main__":
    main()
