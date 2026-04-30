#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_md_observables(filename, skip_fraction=0.2):
    rg2_vals = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                # format: timestep rg2 pe
                rg2_vals.append(float(parts[1]))
            except ValueError:
                continue

    if len(rg2_vals) == 0:
        return None

    rg2_vals = np.array(rg2_vals)

    # discard initial fraction as equilibration
    start = int(skip_fraction * len(rg2_vals))
    rg2_vals = rg2_vals[start:]

    if len(rg2_vals) < 2:
        return None

    rg2_mean = np.mean(rg2_vals)
    rg2_var = np.var(rg2_vals, ddof=1)
    rg4_mean = np.mean(rg2_vals ** 2)

    return {
        "rg2_mean": rg2_mean,
        "rg2_var": rg2_var,
        "rg4_mean": rg4_mean,
        "nsamples": len(rg2_vals),
    }


def load_all_md(md_root="md_runs", skip_fraction=0.2):
    rows = []

    pattern = os.path.join(md_root, "N*_eps*", "md_observables.dat")
    files = sorted(glob.glob(pattern))

    for f in files:
        m = re.search(r"N(\d+)_eps([0-9.]+)", f)
        if m is None:
            continue

        N = int(m.group(1))
        eps = float(m.group(2))

        result = read_md_observables(f, skip_fraction=skip_fraction)
        if result is None:
            continue

        rows.append({
            "N": N,
            "epsilon": eps,
            "rg2_mean": result["rg2_mean"],
            "rg2_var": result["rg2_var"],
            "rg4_mean": result["rg4_mean"],
            "nsamples": result["nsamples"],
            "file": f
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    return df.sort_values(["N", "epsilon"]).reset_index(drop=True)


def plot_fluctuation_vs_epsilon(df, outfile="md_fluctuation_vs_epsilon.png"):
    plt.figure(figsize=(8, 6))

    for N in sorted(df["N"].unique()):
        d = df[df["N"] == N].sort_values("epsilon")
        plt.plot(d["epsilon"], d["rg2_var"], "o-", label="N={}".format(N))

    plt.xlabel(r"Interaction parameter $\epsilon$")
    plt.ylabel(r"$\mathrm{Var}(R_g^2)$")
    plt.title(r"MD fluctuations: $\mathrm{Var}(R_g^2)$ vs. $\epsilon$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()


def plot_scaled_fluctuation_vs_epsilon(df, outfile="md_N_var_rg2_vs_epsilon.png"):
    plt.figure(figsize=(8, 6))

    for N in sorted(df["N"].unique()):
        d = df[df["N"] == N].sort_values("epsilon")
        plt.plot(d["epsilon"], N * d["rg2_var"], "o-", label="N={}".format(N))

    plt.xlabel(r"Interaction parameter $\epsilon$")
    plt.ylabel(r"$N\,\mathrm{Var}(R_g^2)$")
    plt.title(r"MD scaled fluctuations: $N\,\mathrm{Var}(R_g^2)$ vs. $\epsilon$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()


def plot_theta_region_vs_N(df, theta_eps=(0.35, 0.40), outfile="md_fluctuation_theta_vs_N.png"):
    plt.figure(figsize=(8, 6))

    # pick points closest to requested epsilons
    for target_eps in theta_eps:
        rows = []
        for N in sorted(df["N"].unique()):
            dN = df[df["N"] == N].copy()
            if len(dN) == 0:
                continue
            idx = np.argmin(np.abs(dN["epsilon"].values - target_eps))
            row = dN.iloc[idx]
            rows.append((row["N"], row["epsilon"], row["rg2_var"]))

        if len(rows) > 0:
            rows = sorted(rows, key=lambda x: x[0])
            Ns = [r[0] for r in rows]
            vars_ = [r[2] for r in rows]
            actual_eps = rows[0][1]
            plt.plot(Ns, vars_, "o-", label=r"$\epsilon \approx {:.2f}$".format(actual_eps))

    plt.xlabel(r"Chain length $N$")
    plt.ylabel(r"$\mathrm{Var}(R_g^2)$")
    plt.title(r"MD fluctuations near $\theta$: $\mathrm{Var}(R_g^2)$ vs. $N$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()


def main():
    md_root = "md_runs"
    skip_fraction = 0.2

    df = load_all_md(md_root=md_root, skip_fraction=skip_fraction)

    if len(df) == 0:
        print("No MD files found under {}".format(md_root))
        return

    print(df)
    df.to_csv("md_fluctuations_processed.csv", index=False)

    plot_fluctuation_vs_epsilon(df, outfile="md_fluctuation_vs_epsilon.png")
    plot_scaled_fluctuation_vs_epsilon(df, outfile="md_N_var_rg2_vs_epsilon.png")
    plot_theta_region_vs_N(df, theta_eps=(0.35, 0.40), outfile="md_fluctuation_theta_vs_N.png")

    print("\nWrote:")
    print("  md_fluctuations_processed.csv")
    print("  md_fluctuation_vs_epsilon.png")
    print("  md_N_var_rg2_vs_epsilon.png")
    print("  md_fluctuation_theta_vs_N.png")


if __name__ == "__main__":
    main()