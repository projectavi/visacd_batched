#!/usr/bin/env python3
"""Benchmark batched VisACD on the meshes in data/samples.

For each seed, all meshes are decomposed by one ``process_batch`` call. Mesh
loading is done before timing begins, and the final report uses population
standard deviation across complete-batch timings.
"""

import argparse
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import trimesh

import visacd


SAMPLE_MESHES = (
    "Bottle.obj",
    "Kettle.obj",
    "KitchenPot.obj",
    "Octocat-v2.obj",
    "SnowFlake.obj",
    "fandisk.obj",
    "rocker-arm.obj",
    "teapot.obj",
    "alligator.obj",
    "armadillo.obj",
    "cow.obj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Time batched VisACD decomposition of the sample meshes across "
            "multiple random seeds."
        )
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "samples",
        help="directory containing the sample OBJ files",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="number of consecutive seeds to benchmark (default: 10)",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="first seed value (default: 0)",
    )
    parser.add_argument(
        "--concavity",
        type=float,
        default=0.04,
        help="VisACD concavity threshold (default: 0.04)",
    )
    parser.add_argument(
        "--num-parts",
        type=int,
        default=8,
        help="maximum number of output parts (default: 8)",
    )
    parser.add_argument(
        "--show-visacd-output",
        action="store_true",
        help="show verbose native VisACD output during each run",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="only print the final aggregate report",
    )
    args = parser.parse_args()

    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if args.num_parts < 1:
        parser.error("--num-parts must be at least 1")
    if args.concavity < 0:
        parser.error("--concavity must be non-negative")

    return args


def load_mesh(path: Path) -> visacd.Mesh:
    loaded = trimesh.load(str(path), force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError("{} did not load as a triangle mesh".format(path))
    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise ValueError("{} is empty".format(path))

    mesh = visacd.Mesh()
    mesh.vertices = visacd.VecArray3d(loaded.vertices.tolist())
    mesh.triangles = visacd.make_vecarray3i(
        np.asarray(loaded.faces, dtype=np.int32)
    )
    return mesh


def load_samples(mesh_dir: Path) -> List[Tuple[str, visacd.Mesh]]:
    missing = [name for name in SAMPLE_MESHES if not (mesh_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "missing sample mesh(es) in {}: {}".format(
                mesh_dir, ", ".join(missing)
            )
        )

    samples = []
    for name in SAMPLE_MESHES:
        path = mesh_dir / name
        print("Loading {}...".format(path), flush=True)
        samples.append((name, load_mesh(path)))
    return samples


@contextmanager
def suppress_native_stdout(enabled: bool) -> Iterator[None]:
    """Temporarily suppress C/C++ writes to stdout, restoring it on errors."""
    if not enabled:
        yield
        return

    sys.stdout.flush()
    stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            yield
    finally:
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)


def population_std(values: List[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def print_summary(totals: List[float], seed_count: int) -> None:
    print()
    print(
        "Summary over {} seeds (seconds; population standard deviation)".format(
            seed_count
        )
    )
    print(
        "{:<22} {:>8} {:>12} {:>12}".format(
            "Batch", "Runs", "Mean", "Std dev"
        )
    )
    print("-" * 58)
    print(
        "{:<22} {:>8d} {:>12.6f} {:>12.6f}".format(
            "TOTAL ({} meshes)".format(len(SAMPLE_MESHES)),
            len(totals),
            statistics.mean(totals),
            population_std(totals),
        )
    )


def main() -> None:
    args = parse_args()
    samples = load_samples(args.mesh_dir)
    meshes = [mesh for _, mesh in samples]
    totals = []  # type: List[float]

    for seed_index in range(args.seeds):
        seed = args.seed_start + seed_index
        visacd.set_seed(seed)

        with suppress_native_stdout(not args.show_visacd_output):
            start = time.perf_counter()
            results = visacd.process_batch(
                meshes,
                concavity=args.concavity,
                num_parts=args.num_parts,
            )
            elapsed = time.perf_counter() - start

        if len(results) != len(samples):
            raise RuntimeError(
                "process_batch returned {} results for {} meshes".format(
                    len(results), len(samples)
                )
            )

        totals.append(elapsed)
        if not args.quiet_progress:
            for (name, _), result in zip(samples, results):
                print(
                    "seed={:<6d} mesh={:<18} parts={}".format(
                        seed, name, result.num_parts
                    ),
                    flush=True,
                )
            print(
                "seed={:<6d} batch={:>10.6f}s".format(seed, elapsed),
                flush=True,
            )

    print_summary(totals, args.seeds)


if __name__ == "__main__":
    main()
