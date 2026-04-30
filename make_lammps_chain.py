#!/usr/bin/env python3
# make_lammps_chain.py

import numpy as np
import argparse

def too_close(candidate, existing, min_dist):
    for r in existing:
        if np.linalg.norm(candidate - r) < min_dist:
            return True
    return False

def make_chain(N, box=200.0, bond_length=1.0, mode="coil", seed=1234, min_dist=0.90):
    rng = np.random.default_rng(seed)
    pos = np.zeros((N, 3), dtype=float)

    for i in range(1, N):
        accepted = False
        for _ in range(5000):
            step = rng.normal(size=3)
            step /= np.linalg.norm(step)
            candidate = pos[i - 1] + bond_length * step

            if mode == "compact":
                candidate *= 0.985

            # avoid overlaps with earlier nonbonded beads
            if not too_close(candidate, pos[:max(i - 2, 0)], min_dist):
                pos[i] = candidate
                accepted = True
                break

        if not accepted:
            pos[i] = pos[i - 1] + np.array([bond_length, 0.0, 0.0])

    pos -= np.mean(pos, axis=0)
    pos += box / 2.0
    return pos

def write_lammps_data(filename, pos, box=200.0):
    N = len(pos)
    nbonds = N - 1

    with open(filename, "w") as f:
        f.write("Polymer chain data file\n\n")
        f.write(f"{N} atoms\n")
        f.write(f"{nbonds} bonds\n\n")
        f.write("1 atom types\n")
        f.write("1 bond types\n\n")
        f.write(f"0.0 {box:.6f} xlo xhi\n")
        f.write(f"0.0 {box:.6f} ylo yhi\n")
        f.write(f"0.0 {box:.6f} zlo zhi\n\n")

        f.write("Masses\n\n")
        f.write("1 1.0\n\n")

        f.write("Atoms\n\n")
        for i, r in enumerate(pos, start=1):
            f.write(f"{i} 1 1 {r[0]:.8f} {r[1]:.8f} {r[2]:.8f}\n")

        f.write("\nBonds\n\n")
        for i in range(1, N):
            f.write(f"{i} 1 {i} {i+1}\n")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--box", type=float, default=200.0)
    p.add_argument("--bond-length", type=float, default=1.0)
    p.add_argument("--mode", choices=["coil", "compact"], default="coil")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--min-dist", type=float, default=0.90)
    p.add_argument("--out", default="polymer.data")
    args = p.parse_args()

    pos = make_chain(
        args.N,
        box=args.box,
        bond_length=args.bond_length,
        mode=args.mode,
        seed=args.seed,
        min_dist=args.min_dist
    )
    write_lammps_data(args.out, pos, box=args.box)
    print(f"Wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()