#!/usr/bin/env python3
"""
plot_21step_density_profile.py

Make a 21-step bulk density vs time plot from a LAMMPS 21-step output
(SLURM output or log file), similar to the paper figure.

Usage
-----
If this script is in the same folder as your 21-step output:
    python plot_21step_density_profile.py
"""

import glob
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt


def find_input_file():
    candidates = sorted(glob.glob("slurm*.out"))
    if candidates:
        return candidates[0]

    candidates = sorted(glob.glob("log*.lammps"))
    if candidates:
        return candidates[0]

    candidates = sorted(glob.glob("*.out"))
    if candidates:
        return candidates[0]

    for path in sorted(glob.glob("*")):
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


def parse_all_thermo_blocks(path):
    with open(path, "r", errors="ignore") as fh:
        lines = fh.readlines()

    blocks = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if re.match(r"\s*Step\s+Temp\s+Press\s+Volume\s+Density", line):
            block_rows = []
            i += 1
            while i < n:
                if lines[i].startswith("Loop time of"):
                    break
                if is_numeric_thermo_row(lines[i]):
                    parts = lines[i].split()
                    block_rows.append((
                        int(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                        float(parts[4])
                    ))
                i += 1

            if block_rows:
                arr = np.array(block_rows, dtype=float)
                blocks.append({
                    "step": arr[:, 0],
                    "temp": arr[:, 1],
                    "press": arr[:, 2],
                    "vol": arr[:, 3],
                    "density": arr[:, 4],
                })
        i += 1

    return blocks


def main():
    input_file = find_input_file()

    if input_file is None:
        print("[ERROR] No suitable 21-step output file found in the current directory.", flush=True)
        sys.exit(1)

    print(f"[INFO] Using input file: {input_file}", flush=True)

    run_lengths = {
        1: 50000,
        2: 50000,
        3: 50000,
        4: 50000,
        5: 100000,
        6: 50000,
        7: 50000,
        8: 100000,
        9: 50000,
        10: 50000,
        11: 100000,
        12: 5000,
        13: 5000,
        14: 10000,
        15: 5000,
        16: 5000,
        17: 10000,
        18: 5000,
        19: 5000,
        20: 10000,
        21: 800000,
    }

    blocks = parse_all_thermo_blocks(input_file)

    if len(blocks) < 22:
        print(f"[ERROR] Expected minimization + 21 run blocks, but found only {len(blocks)} thermo blocks.", flush=True)
        sys.exit(1)

    run_blocks = blocks[1:22]

    if len(run_blocks) != 21:
        print(f"[ERROR] Could not isolate exactly 21 run blocks. Found {len(run_blocks)}.", flush=True)
        sys.exit(1)

    offsets = {}
    cumulative = 0
    for step_id in range(1, 22):
        offsets[step_id] = cumulative
        cumulative += run_lengths[step_id]

    all_time_ns = []
    all_density = []
    stitched = {}

    for step_id, block in enumerate(run_blocks, start=1):
        raw_step = block["step"]
        block_step = raw_step - raw_step[0]
        global_step = offsets[step_id] + block_step
        time_ns = global_step * 1.0e-6  # 1 fs = 1e-6 ns

        stitched[step_id] = {
            "time_ns": time_ns,
            "step_in_block": block_step,
            "density": block["density"],
            "temp": block["temp"],
            "press": block["press"],
            "vol": block["vol"],
        }

        all_time_ns.append(time_ns)
        all_density.append(block["density"])

    all_time_ns = np.concatenate(all_time_ns)
    all_density = np.concatenate(all_density)

    # Average density over last 400 ps of Step 21
    step21_steps = stitched[21]["step_in_block"]
    step21_density = stitched[21]["density"]
    mask21 = step21_steps >= 400000

    if np.sum(mask21) < 1:
        print("[WARN] Could not find data for the last 400 ps of Step 21; using full Step 21 average instead.", flush=True)
        avg_density_21 = np.mean(step21_density)
    else:
        avg_density_21 = np.mean(step21_density[mask21])

    print(f"[INFO] Average density over last 400 ps of Step 21 = {avg_density_21:.4f} g/cm^3", flush=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(all_time_ns, all_density, color="#2c4aa5", lw=0.8)

    ax.set_xlabel("Time [ns]")
    ax.set_ylabel(r"Bulk density [g cm$^{-3}$]")

    ax.text(0.03, 0.95, "PIM-1", transform=ax.transAxes,
            ha="left", va="top", fontsize=13)

    pressure_labels = {
        3: r"0.02 $P_{\max}$",
        6: r"0.6 $P_{\max}$",
        9: r"$P_{\max}$",
        12: r"0.5 $P_{\max}$",
        15: r"0.1 $P_{\max}$",
        18: r"0.01 $P_{\max}$",
        21: r"$P_{\mathrm{final}}$",
    }

    label_frac = {
        3: 0.55,
        6: 0.55,
        9: 0.45,
        12: 0.35,
        15: 0.35,
        18: 0.30,
        21: 0.55,
    }

    x_shift = {
        3: -0.015,
        6: -0.010,
        9: -0.005,
        12: 0.010,
        15: 0.010,
        18: 0.012,
        21: 0.030,
    }

    y_shift = {
        3: 0.025,
        6: 0.040,
        9: 0.040,
        12: 0.035,
        15: 0.055,
        18: 0.030,
        21: -0.095,
    }

    for step_id, label in pressure_labels.items():
        t = stitched[step_id]["time_ns"]
        d = stitched[step_id]["density"]

        frac = label_frac.get(step_id, 0.5)
        idx = min(len(t) - 1, max(0, int(frac * (len(t) - 1))))

        i0 = max(0, idx - 2)
        i1 = min(len(d), idx + 3)

        tx = t[idx]
        ty = np.mean(d[i0:i1])

        ax.text(
            tx + x_shift.get(step_id, 0.0),
            ty + y_shift.get(step_id, 0.0),
            label,
            fontsize=11
        )

    # Add average density from last 400 ps of Step 21
    ax.text(
        0.63, 0.08,
        rf"$\langle \rho \rangle_{{21,\mathrm{{last\ 400\ ps}}}}$ = {avg_density_21:.3f} g cm$^{{-3}}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.8)
    )

    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)

    ax.set_xlim(0.0, np.max(all_time_ns) * 1.02)
    ax.set_ylim(np.min(all_density) - 0.05, np.max(all_density) + 0.08)

    fig.tight_layout()
    outpng = "density_21step_profile.png"
    fig.savefig(outpng, dpi=300)
    print(f"[INFO] Wrote plot: {outpng}", flush=True)


if __name__ == "__main__":
    main()