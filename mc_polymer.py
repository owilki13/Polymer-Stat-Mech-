#!/usr/bin/env python3
import numpy as np
import argparse
import csv
import os
import sys


def radius_of_gyration_squared(pos):
    com = np.mean(pos, axis=0)
    dr = pos - com
    return np.mean(np.sum(dr * dr, axis=1))


def make_initial_chain(N, mode="coil", bond_length=1.0, seed=1234):
    rng = np.random.RandomState(seed)
    pos = np.zeros((N, 3), dtype=float)

    for i in range(1, N):
        step = rng.normal(size=3)
        step /= np.linalg.norm(step)
        pos[i] = pos[i - 1] + bond_length * step
        if mode == "compact":
            pos[i] *= 0.98

    return pos


def lj_shift(epsilon, sigma, rc):
    sr6 = (sigma / rc) ** 6
    return 4.0 * epsilon * (sr6 * sr6 - sr6)


def lj_pair(r, epsilon, sigma=1.0, rc=2.5):
    if r >= rc:
        return 0.0
    sr6 = (sigma / r) ** 6
    return 4.0 * epsilon * (sr6 * sr6 - sr6) - lj_shift(epsilon, sigma, rc)


def bond_energy(r, k_bond=50.0, r0=1.0):
    return 0.5 * k_bond * (r - r0) ** 2


def total_energy(pos, epsilon, sigma=1.0, rc=2.5, k_bond=50.0, r0=1.0):
    N = len(pos)
    E = 0.0

    for i in range(N - 1):
        r = np.linalg.norm(pos[i + 1] - pos[i])
        E += bond_energy(r, k_bond, r0)

    for i in range(N - 1):
        for j in range(i + 2, N):
            r = np.linalg.norm(pos[j] - pos[i])
            if r < 1e-10:
                return np.inf
            E += lj_pair(r, epsilon, sigma, rc)

    return E


def local_energy_of_bead(pos, i, epsilon, sigma=1.0, rc=2.5, k_bond=50.0, r0=1.0):
    N = len(pos)
    E = 0.0

    if i > 0:
        r = np.linalg.norm(pos[i] - pos[i - 1])
        E += bond_energy(r, k_bond, r0)
    if i < N - 1:
        r = np.linalg.norm(pos[i + 1] - pos[i])
        E += bond_energy(r, k_bond, r0)

    for j in range(N):
        if j == i or abs(j - i) == 1:
            continue
        r = np.linalg.norm(pos[j] - pos[i])
        if r < 1e-10:
            return np.inf
        E += lj_pair(r, epsilon, sigma, rc)

    return E


def pivot_move(pos, rng):
    N = len(pos)
    if N < 4:
        return pos.copy()

    pivot = rng.randint(1, N - 1)
    rotate_tail = rng.rand() < 0.5
    new_pos = pos.copy()

    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-np.pi, np.pi)

    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * np.dot(K, K)

    anchor = pos[pivot].copy()

    if rotate_tail:
        idx = range(pivot + 1, N)
    else:
        idx = range(0, pivot)

    for ii in idx:
        rel = pos[ii] - anchor
        new_pos[ii] = anchor + np.dot(R, rel)

    return new_pos


def run_mc(
    N,
    epsilon,
    steps,
    burn,
    sample_every,
    max_disp,
    temperature,
    init_mode,
    seed,
    pivot_prob,
    save_samples=False,
    sigma=1.0,
    rc=2.5,
    k_bond=50.0,
    r0=1.0
):
    rng = np.random.RandomState(seed)
    beta = 1.0 / temperature

    pos = make_initial_chain(N, mode=init_mode, bond_length=r0, seed=seed)
    E = total_energy(pos, epsilon, sigma, rc, k_bond, r0)

    accepted = 0
    attempted = 0

    rg2_samples = []
    energy_samples = []

    for step in range(steps):
        attempted += 1

        if rng.rand() < pivot_prob:
            trial = pivot_move(pos, rng)
            E_new = total_energy(trial, epsilon, sigma, rc, k_bond, r0)
            dE = E_new - E
            if dE <= 0.0 or rng.rand() < np.exp(-beta * dE):
                pos = trial
                E = E_new
                accepted += 1
        else:
            i = rng.randint(0, N)
            old_pos = pos[i].copy()
            old_local = local_energy_of_bead(pos, i, epsilon, sigma, rc, k_bond, r0)

            pos[i] += rng.uniform(-max_disp, max_disp, size=3)
            new_local = local_energy_of_bead(pos, i, epsilon, sigma, rc, k_bond, r0)

            dE = new_local - old_local
            if np.isfinite(new_local) and (dE <= 0.0 or rng.rand() < np.exp(-beta * dE)):
                E += dE
                accepted += 1
            else:
                pos[i] = old_pos

        if step >= burn and ((step - burn) % sample_every == 0):
            rg2 = radius_of_gyration_squared(pos)
            rg2_samples.append(rg2)
            if save_samples:
                energy_samples.append(E)

        if step > 0 and step % 50000 == 0:
            print(
                "Progress: N={}, epsilon={:.3f}, step={}/{}, acc={:.3f}".format(
                    N, epsilon, step, steps, float(accepted) / attempted
                ),
                flush=True
            )

    rg2_samples = np.array(rg2_samples)

    results = {
        "N": N,
        "epsilon": epsilon,
        "temperature": temperature,
        "steps": steps,
        "burn": burn,
        "sample_every": sample_every,
        "acceptance": float(accepted) / attempted if attempted > 0 else 0.0,
        "rg2_mean": np.mean(rg2_samples),
        "rg2_var": np.var(rg2_samples, ddof=1) if len(rg2_samples) > 1 else 0.0,
        "rg4_mean": np.mean(rg2_samples ** 2),
        "nsamples": len(rg2_samples),
    }

    return results, rg2_samples, energy_samples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--Ns", type=int, nargs="+", required=True)
    p.add_argument("--epsilons", type=float, nargs="+", required=True)
    p.add_argument("--steps", type=int, default=600000)
    p.add_argument("--burn", type=int, default=100000)
    p.add_argument("--sample-every", type=int, default=2000)
    p.add_argument("--max-disp", type=float, default=0.25)
    p.add_argument("--pivot-prob", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--init", type=str, default="coil", choices=["coil", "compact"])
    p.add_argument("--outdir", type=str, default="mc_results")
    p.add_argument("--save-samples", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    summary_file = os.path.join(args.outdir, "summary.csv")
    write_header = not os.path.exists(summary_file)

    with open(summary_file, "a", newline="") as fsum:
        writer = csv.writer(fsum)

        if write_header:
            writer.writerow([
                "N", "epsilon", "temperature", "steps", "burn", "sample_every",
                "acceptance", "rg2_mean", "rg2_var", "rg4_mean", "nsamples"
            ])
            fsum.flush()

        for N in args.Ns:
            for ie, eps in enumerate(args.epsilons):
                seed = args.seed + 1000 * N + ie
                print(
                    "Running MC: N={}, epsilon={:.3f}, seed={}".format(N, eps, seed),
                    flush=True
                )

                results, rg2_samples, energy_samples = run_mc(
                    N=N,
                    epsilon=eps,
                    steps=args.steps,
                    burn=args.burn,
                    sample_every=args.sample_every,
                    max_disp=args.max_disp,
                    temperature=args.temperature,
                    init_mode=args.init,
                    seed=seed,
                    pivot_prob=args.pivot_prob,
                    save_samples=args.save_samples
                )

                writer.writerow([
                    results["N"], results["epsilon"], results["temperature"],
                    results["steps"], results["burn"], results["sample_every"],
                    results["acceptance"], results["rg2_mean"], results["rg2_var"],
                    results["rg4_mean"], results["nsamples"]
                ])
                fsum.flush()

                if args.save_samples:
                    sample_file = os.path.join(
                        args.outdir,
                        "samples_N{}_eps{:.3f}.csv".format(N, eps)
                    )
                    with open(sample_file, "w", newline="") as fs:
                        w = csv.writer(fs)
                        w.writerow(["sample_index", "rg2", "energy"])
                        for k in range(len(rg2_samples)):
                            w.writerow([k, rg2_samples[k], energy_samples[k]])
                        fs.flush()

                print(
                    "Finished: N={}, epsilon={:.3f}, <Rg^2>={:.5f}, acc={:.3f}".format(
                        N, eps, results["rg2_mean"], results["acceptance"]
                    ),
                    flush=True
                )

    print("Done.", flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()