#!/usr/bin/env python3
"""Benchmark VisACD on a deterministic heterogeneous Thingi10K sample."""

import argparse
import hashlib
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import thingi10k

import visacd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one heterogeneous, vertex-stratified Thingi10K sample. "
            "Mesh loading and Python-to-VisACD conversion are outside timing."
        )
    )
    parser.add_argument("-n", type=int, default=200, help="sample size (default: 200)")
    parser.add_argument(
        "--mode",
        choices=("batch", "sequential"),
        default="batch",
        help="one process_batch call or repeated process calls",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--process-seed", type=int, default=42)
    parser.add_argument("--strata", type=int, default=4)
    parser.add_argument("--min-vertices", type=int, default=100)
    parser.add_argument("--max-vertices", type=int, default=50_000)
    parser.add_argument("--concavity", type=float, default=0.04)
    parser.add_argument("--num-parts", type=int, default=2)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("downloads/thingi10k"),
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="print the selected Thingi10K file IDs",
    )
    parser.add_argument(
        "--show-visacd-output",
        action="store_true",
        help="do not suppress native VisACD logging",
    )
    args = parser.parse_args()
    if args.n < 1:
        parser.error("-n must be at least 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")
    if args.strata < 1:
        parser.error("--strata must be at least 1")
    if args.min_vertices < 3:
        parser.error("--min-vertices must be at least 3")
    if args.max_vertices < args.min_vertices:
        parser.error("--max-vertices must be >= --min-vertices")
    return args


@contextmanager
def suppress_native_stdout(enabled: bool):
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


def choose_rows(args: argparse.Namespace):
    thingi10k.init(variant="npz", cache_dir=str(args.cache_dir))
    population = thingi10k.dataset(
        num_vertices=(args.min_vertices, args.max_vertices),
        num_components=1,
        self_intersecting=False,
        solid=True,
    )
    if args.n > len(population):
        raise ValueError(
            f"requested {args.n} meshes from a population of {len(population)}"
        )

    vertex_counts = np.asarray(population["num_vertices"], dtype=np.int64)
    file_ids = np.asarray(population["file_id"], dtype=np.int64)
    order = np.lexsort((file_ids, vertex_counts))
    bins = np.array_split(order, args.strata)
    quotas = np.full(args.strata, args.n // args.strata, dtype=np.int64)
    quotas[: args.n % args.strata] += 1
    if any(quota > len(bin_indices) for quota, bin_indices in zip(quotas, bins)):
        raise ValueError("a vertex-count stratum is too small for the requested sample")

    rng = np.random.default_rng(args.sample_seed)
    chosen = np.concatenate(
        [
            rng.choice(bin_indices, size=int(quota), replace=False)
            for quota, bin_indices in zip(quotas, bins)
            if quota
        ]
    )
    rng.shuffle(chosen)
    return population.select(chosen.tolist()), len(population)


def make_mesh(path: str):
    with np.load(path) as arrays:
        vertices = np.asarray(arrays["vertices"], dtype=np.float64)
        facets = np.asarray(arrays["facets"], dtype=np.int32)
    try:
        return visacd.Mesh(
            np.ascontiguousarray(vertices),
            np.ascontiguousarray(facets),
        )
    except TypeError:
        pass
    mesh = visacd.Mesh()
    mesh.vertices = visacd.VecArray3d(vertices.tolist())
    mesh.triangles = visacd.make_vecarray3i(facets)
    return mesh


def result_digest(results) -> str:
    digest = hashlib.sha256()
    for result in results:
        digest.update(int(result.num_parts).to_bytes(4, "little", signed=False))
        for part in result.parts:
            vertices = np.asarray(list(part.vertices), dtype=np.float64)
            triangles = np.asarray(list(part.triangles), dtype=np.int32)
            digest.update(vertices.shape[0].to_bytes(8, "little", signed=False))
            digest.update(vertices.tobytes())
            digest.update(triangles.shape[0].to_bytes(8, "little", signed=False))
            digest.update(triangles.tobytes())
    return digest.hexdigest()[:16]


def run_once(args: argparse.Namespace, meshes):
    visacd.set_seed(args.process_seed)
    start = time.perf_counter()
    if args.mode == "batch":
        results = visacd.process_batch(
            meshes,
            concavity=args.concavity,
            num_parts=args.num_parts,
        )
    else:
        results = [
            visacd.process(
                mesh,
                concavity=args.concavity,
                num_parts=args.num_parts,
            )
            for mesh in meshes
        ]
    elapsed = time.perf_counter() - start
    return elapsed, results


def main() -> None:
    args = parse_args()
    rows, population_size = choose_rows(args)
    ids = [int(value) for value in rows["file_id"]]
    counts = np.asarray(rows["num_vertices"], dtype=np.int64)
    categories = sorted({value for value in rows["category"] if value})
    manifest = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]

    print(
        "Dataset: Thingi10K npz; solid, non-self-intersecting, "
        "single-component meshes"
    )
    print(
        f"Population: {population_size}; sample: {len(rows)}; "
        f"sample seed: {args.sample_seed}; strata: {args.strata}"
    )
    print(
        "Vertices (min/q1/median/q3/max): "
        + "/".join(
            str(int(value))
            for value in np.quantile(counts, [0.0, 0.25, 0.5, 0.75, 1.0])
        )
    )
    print(f"Categories: {len(categories)}; manifest digest: {manifest}")
    if args.show_ids:
        print("File IDs: " + " ".join(map(str, ids)))

    print("Converting meshes outside the timed region...", flush=True)
    meshes = [make_mesh(path) for path in rows["file_path"]]

    with suppress_native_stdout(not args.show_visacd_output):
        for _ in range(args.warmups):
            run_once(args, meshes)

    timings = []
    digests = []
    for repeat in range(args.repeats):
        with suppress_native_stdout(not args.show_visacd_output):
            elapsed, results = run_once(args, meshes)
        digest = result_digest(results)
        timings.append(elapsed)
        digests.append(digest)
        print(
            f"repeat {repeat + 1}: {elapsed:.4f} s, "
            f"{len(meshes) / elapsed:.2f} meshes/s, digest {digest}",
            flush=True,
        )

    median = statistics.median(timings)
    print(
        f"Median: {median:.4f} s, {len(meshes) / median:.2f} meshes/s; "
        f"repeatable: {len(set(digests)) == 1}"
    )


if __name__ == "__main__":
    main()
