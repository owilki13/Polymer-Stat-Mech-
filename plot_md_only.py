#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_one_md_file(filename, skip_fraction=0.2):
    rg2_vals = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                # LAMMPS ave/time output: timestep rg2 pe
                rg2_vals.append(float(parts[1]))
            except ValueError:
                continue

    if len(rg2_vals) == 0:
        return None

    rg2_vals = np.array(rg2_vals)

    # discard first fraction as equilibration
    start = int(skip_fraction * len(rg2_vals))
    rg2_vals = rg2_vals[start:]

    if len(rg2_vals) == 0:
        return None

    return {
        "rg2_mean": np.mean(rg2_vals),
        "rg2_var": np.var(rg2_vals, ddof=1) if len(rg2_vals) > 1 else 0.0,
        "nsamples": len(rg2_vals)
    }


def load_md(md_root="md_runs", skip_fraction=0.2):
    rows = []

    pattern = os.path.join(md_root, "N*_eps*", "md_observables.dat")
    files = sorted(glob.glob(pattern))

    for f in files:
        m = re.search(r"N(\d+)_eps([0-9.]+)", f)
        if m is None:
            continue

        N = int(m.group(1))
        eps = float(m.group(2))

        result = load_one_md_file(f, skip_fraction=skip_fraction)
        if result is None:
            continue

        rows.append({
            "N": N,
            "epsilon": eps,
            "rg2_mean": result["rg2_mean"],
            "rg2_var": result["rg2_var"],
            "nsamples": result["nsamples"],
            "file": f
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    return df.sort_values(["N", "epsilon"]).reset_index(drop=True)


def plot_rg2(df, outfile="md_rg2_vs_epsilon.png"):
    plt.figure(figsize=(8, 6))

    for N in sorted(df["N"].unique()):
        d = df[df["N"] == N].sort_values("epsilon")
        plt.plot(d["epsilon"], d["rg2_mean"], "o-", label="N={}".format(N))

    plt.xlabel(r"Interaction parameter $\epsilon$")
    plt.ylabel(r"$\langle R_g^2 \rangle$")
    plt.title(r"MD: $\langle R_g^2 \rangle$ vs. interaction parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()


def plot_rg2_over_N(df, outfile="md_rg2_over_N_vs_epsilon.png"):
    plt.figure(figsize=(8, 6))

    for N in sorted(df["N"].unique()):
        d = df[df["N"] == N].sort_values("epsilon")
        plt.plot(d["epsilon"], d["rg2_mean"] / float(N), "o-", label="N={}".format(N))

    plt.xlabel(r"Interaction parameter $\epsilon$")
    plt.ylabel(r"$\langle R_g^2 \rangle / N$")
    plt.title(r"MD: $\langle R_g^2 \rangle / N$ vs. interaction parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()


def main():
    md_root = "md_runs"
    df = load_md(md_root=md_root, skip_fraction=0.2)

    if len(df) == 0:
        print("No md_observables.dat files found under {}".format(md_root))
        return

    print(df)
    df.to_csv("md_processed.csv", index=False)

    plot_rg2(df, outfile="md_rg2_vs_epsilon.png")
    plot_rg2_over_N(df, outfile="md_rg2_over_N_vs_epsilon.png")

    print("\nWrote:")
    print("  md_processed.csv")
    print("  md_rg2_vs_epsilon.png")
    print("  md_rg2_over_N_vs_epsilon.png")


if __name__ == "__main__":
    main()