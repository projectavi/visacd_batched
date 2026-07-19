#!/usr/bin/env python3
"""Verify seeded batch equivalence across CPU threads and GPU wave sizes."""

import argparse
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np
import trimesh

import visacd


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "meshes",
        type=Path,
        nargs="*",
        help="meshes to verify; defaults to every data/samples OBJ",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        action="append",
        help="thread count to test; may be repeated (defaults: 0, 1, 4, 32)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        action="append",
        help="wave cap to test; may be repeated (defaults: 0, 1, 4, 32, 200)",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--concavity", type=float, default=0.04)
    parser.add_argument("--num-parts", type=int, default=2)
    parser.add_argument(
        "--show-visacd-output",
        action="store_true",
        help="show native per-mesh progress output",
    )
    args = parser.parse_args()
    args.cpu_threads = args.cpu_threads or [0, 1, 4, 32]
    args.max_batch_size = args.max_batch_size or [0, 1, 4, 32, 200]
    if any(value < 0 for value in args.cpu_threads):
        parser.error("--cpu-threads values must be non-negative")
    if any(value < 0 for value in args.max_batch_size):
        parser.error("--max-batch-size values must be non-negative")
    if not 0.0 < args.memory_fraction <= 1.0:
        parser.error("--memory-fraction must be in (0, 1]")
    if args.num_parts < 1:
        parser.error("--num-parts must be at least 1")
    if args.concavity < 0.0:
        parser.error("--concavity must be non-negative")
    return args


def load_mesh(path):
    source = trimesh.load(path, force="mesh")
    if not isinstance(source, trimesh.Trimesh):
        raise TypeError("{} did not load as a triangle mesh".format(path))
    return visacd.Mesh(
        np.ascontiguousarray(source.vertices, dtype=np.float64),
        np.ascontiguousarray(source.faces, dtype=np.int32),
    )


def result_digest(results):
    digest = hashlib.sha256()
    for result in results:
        digest.update(struct.pack("<di", result.concavity, result.num_parts))
        for part in result.parts:
            vertices = np.asarray(part.vertices, dtype="<f8")
            triangles = np.asarray(part.triangles, dtype="<i4")
            digest.update(struct.pack("<QQ", len(vertices), len(triangles)))
            digest.update(vertices.tobytes())
            digest.update(triangles.tobytes())
    return digest.hexdigest()


@contextmanager
def suppress_native_stdout(enabled):
    if not enabled:
        yield
        return
    sys.stdout.flush()
    saved_stdout = os.dup(sys.stdout.fileno())
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            yield
    finally:
        os.dup2(saved_stdout, sys.stdout.fileno())
        os.close(saved_stdout)


def main():
    args = parse_args()
    paths = args.meshes or sorted((ROOT / "data" / "samples").glob("*.obj"))
    meshes = [load_mesh(path) for path in paths]
    visacd.config.batch_memory_fraction = args.memory_fraction
    reference_digest = None
    failures = 0
    cases = 0
    for cpu_threads in args.cpu_threads:
        for max_batch_size in args.max_batch_size:
            visacd.config.batch_cpu_threads = cpu_threads
            visacd.config.max_batch_size = max_batch_size
            visacd.set_seed(args.seed)
            start = time.perf_counter()
            with suppress_native_stdout(not args.show_visacd_output):
                results = visacd.process_batch(
                    meshes,
                    concavity=args.concavity,
                    num_parts=args.num_parts,
                )
            elapsed = time.perf_counter() - start
            digest = result_digest(results)
            if reference_digest is None:
                reference_digest = digest
            matches = digest == reference_digest
            failures += 0 if matches else 1
            cases += 1
            print(
                "case={} meshes={} cpu_threads={} max_batch_size={} "
                "seconds={:.6f} meshes_per_second={:.3f} "
                "matches_reference={} digest={}".format(
                    cases,
                    len(meshes),
                    cpu_threads,
                    max_batch_size,
                    elapsed,
                    len(meshes) / elapsed,
                    matches,
                    digest,
                ),
                flush=True,
            )
    print(
        "summary cases={} meshes={} failures={} reference_digest={}".format(
            cases, len(meshes), failures, reference_digest
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
