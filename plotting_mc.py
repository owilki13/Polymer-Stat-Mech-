#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt


def main():
    summary_file = "mc_results/summary.csv"
    df = pd.read_csv(summary_file)

    df = df.sort_values(["N", "epsilon"]).copy()
    df["rg2_over_N"] = df["rg2_mean"] / df["N"]

    # Q(1): <Rg^2> vs epsilon
    plt.figure(figsize=(8, 6))
    for N in sorted(df["N"].unique()):
        d = df[df["N"] == N]
        plt.plot(d["epsilon"], d["rg2_mean"], "o-", label="N={}".format(N))
    plt.xlabel(r"Interaction parameter $\epsilon$")
    plt.ylabel(r"$\langle R_g^2 \rangle$")
    plt.title(r"MC: $\langle R_g^2 \rangle$ vs. interaction parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mc_rg2_vs_epsilon.png", dpi=200)
    plt.show()

    # Q(2): <Rg^2>/N vs epsilon
    plt.figure(figsize=(8, 6))
    for N in sorted(df["N"].unique()):
        d = df[df["N"] == N]
        plt.plot(d["epsilon"], d["rg2_over_N"], "o-", label="N={}".format(N))
    plt.xlabel(r"Interaction parameter $\epsilon$")
    plt.ylabel(r"$\langle R_g^2 \rangle / N$")
    plt.title(r"MC: $\langle R_g^2 \rangle/N$ vs. interaction parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mc_rg2_over_N_vs_epsilon.png", dpi=200)
    plt.show()

    # Q(3): fluctuations vs epsilon
    plt.figure(figsize=(8, 6))
    for N in sorted(df["N"].unique()):
        d = df[df["N"] == N]
        plt.plot(d["epsilon"], d["rg2_var"], "o-", label="N={}".format(N))
    plt.xlabel(r"Interaction parameter $\epsilon$")
    plt.ylabel(r"$\mathrm{Var}(R_g^2)$")
    plt.title(r"MC: Fluctuations $\mathrm{Var}(R_g^2)$ vs. interaction parameter")
    plt.legend()
    plt.tight_layout()
    plt.savefig("mc_fluctuations_vs_epsilon.png", dpi=200)
    plt.show()

    df.to_csv("mc_processed.csv", index=False)

    print("Wrote:")
    print("  mc_rg2_vs_epsilon.png")
    print("  mc_rg2_over_N_vs_epsilon.png")
    print("  mc_fluctuations_vs_epsilon.png")
    print("  mc_processed.csv")


if __name__ == "__main__":
    main()