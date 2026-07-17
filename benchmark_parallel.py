#!/usr/bin/env python3
"""Benchmark one VisACD process_batch call with N parallel meshes."""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import statistics
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
        "-n",
        "--num-meshes",
        type=int,
        required=True,
        help="number of meshes passed to one process_batch call",
    )
    parser.add_argument(
        "--mesh",
        type=Path,
        default=ROOT / "data" / "samples" / "cow.obj",
        help="mesh to replicate (default: data/samples/cow.obj)",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--concavity", type=float, default=0.04)
    parser.add_argument("--num-parts", type=int, default=2)
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="batch CPU threads; 0 selects the automatic heuristic",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=0,
        help="maximum GPU wave size; 0 selects memory-aware sizing",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument(
        "--optix-build-preference",
        choices=("trace", "build", "none"),
        default="trace",
        help="OptiX acceleration-structure build preference",
    )
    parser.add_argument(
        "--optix-max-concurrency",
        type=int,
        default=0,
        help="maximum in-flight OptiX jobs; 0 selects automatic sizing",
    )
    parser.add_argument(
        "--show-visacd-output",
        action="store_true",
        help="show native per-mesh progress output",
    )
    args = parser.parse_args()

    if args.num_meshes < 1:
        parser.error("-n/--num-meshes must be at least 1")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.num_parts < 1:
        parser.error("--num-parts must be at least 1")
    if args.concavity < 0.0:
        parser.error("--concavity must be non-negative")
    if args.cpu_threads < 0:
        parser.error("--cpu-threads must be non-negative")
    if args.max_batch_size < 0:
        parser.error("--max-batch-size must be non-negative")
    if not 0.0 < args.memory_fraction <= 1.0:
        parser.error("--memory-fraction must be in (0, 1]")
    if args.optix_max_concurrency < 0:
        parser.error("--optix-max-concurrency must be non-negative")
    return args


def load_batch(path, count):
    loaded = trimesh.load(path, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError("{} did not load as a triangle mesh".format(path))
    vertices = np.asarray(loaded.vertices, dtype=np.float64).tolist()
    triangles = np.asarray(loaded.faces, dtype=np.int32)

    meshes = []
    for _ in range(count):
        mesh = visacd.Mesh()
        mesh.vertices = visacd.VecArray3d(vertices)
        mesh.triangles = visacd.make_vecarray3i(triangles)
        meshes.append(mesh)
    return meshes


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


def result_digest(results):
    digest = hashlib.sha256()
    for result in results:
        digest.update(struct.pack("<di", result.concavity, result.num_parts))
        for part in result.parts:
            vertices = np.asarray(list(part.vertices), dtype="<f8")
            triangles = np.asarray(list(part.triangles), dtype="<i4")
            digest.update(struct.pack("<QQ", len(vertices), len(triangles)))
            digest.update(vertices.tobytes())
            digest.update(triangles.tobytes())
    return digest.hexdigest()


def main():
    args = parse_args()
    meshes = load_batch(args.mesh, args.num_meshes)

    visacd.config.batch_cpu_threads = args.cpu_threads
    visacd.config.max_batch_size = args.max_batch_size
    visacd.config.batch_memory_fraction = args.memory_fraction
    os.environ["VISACD_OPTIX_BUILD_PREFERENCE"] = (
        args.optix_build_preference
    )
    if args.optix_max_concurrency:
        os.environ["VISACD_OPTIX_MAX_CONCURRENCY"] = str(
            args.optix_max_concurrency
        )
    else:
        os.environ.pop("VISACD_OPTIX_MAX_CONCURRENCY", None)

    print(
        "mesh={} n={} repeats={} cpu_threads={} max_batch_size={} "
        "memory_fraction={} optix_build_preference={} "
        "optix_max_concurrency={}".format(
            args.mesh,
            args.num_meshes,
            args.repeats,
            args.cpu_threads,
            args.max_batch_size,
            args.memory_fraction,
            args.optix_build_preference,
            args.optix_max_concurrency,
        )
    )

    timings = []
    digests = []
    for repeat in range(args.repeats):
        visacd.set_seed(args.seed)
        with suppress_native_stdout(not args.show_visacd_output):
            start = time.perf_counter()
            results = visacd.process_batch(
                meshes,
                concavity=args.concavity,
                num_parts=args.num_parts,
            )
            elapsed = time.perf_counter() - start

        if len(results) != args.num_meshes:
            raise RuntimeError(
                "process_batch returned {} results, expected {}".format(
                    len(results), args.num_meshes
                )
            )

        digest = result_digest(results)
        timings.append(elapsed)
        digests.append(digest)
        part_counts = dict(sorted(Counter(
            result.num_parts for result in results
        ).items()))
        print(
            "run={} seconds={:.6f} meshes_per_second={:.3f} "
            "part_counts={} digest={}".format(
                repeat + 1,
                elapsed,
                args.num_meshes / elapsed,
                part_counts,
                digest,
            ),
            flush=True,
        )

    mean = statistics.mean(timings)
    stddev = statistics.pstdev(timings) if len(timings) > 1 else 0.0
    print(
        "summary mean_seconds={:.6f} stddev_seconds={:.6f} "
        "mean_meshes_per_second={:.3f} repeatable={}".format(
            mean,
            stddev,
            args.num_meshes / mean,
            len(set(digests)) == 1,
        )
    )


if __name__ == "__main__":
    main()
