#!/usr/bin/env python3
"""Benchmark VisACD on deterministic smooth variants of one source mesh."""

import argparse
import hashlib
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import trimesh

import visacd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one batch of meshes created by deterministic smooth "
            "vertex perturbations. Input construction is outside timing."
        )
    )
    parser.add_argument("-n", type=int, default=200, help="number of variants")
    parser.add_argument(
        "--mesh",
        type=Path,
        default=Path("data/samples/cow.obj"),
        help="source triangle mesh",
    )
    parser.add_argument(
        "--mode",
        choices=("batch", "sequential"),
        default="batch",
        help="one process_batch call or repeated process calls",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--perturb-seed", type=int, default=42)
    parser.add_argument("--process-seed", type=int, default=42)
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.02,
        help=(
            "smooth normal-displacement amplitude as a fraction of the source "
            "bounding-box diagonal; anisotropic scale varies by twice this"
        ),
    )
    parser.add_argument("--concavity", type=float, default=0.04)
    parser.add_argument("--num-parts", type=int, default=2)
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
    if not np.isfinite(args.amplitude) or args.amplitude < 0:
        parser.error("--amplitude must be finite and non-negative")
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


def smooth_variant(
    vertices: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    diagonal: float,
    amplitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if amplitude == 0:
        return vertices.copy()

    normalized = (vertices - center) / diagonal
    directions = rng.normal(size=(5, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    frequencies = rng.uniform(7.0, 16.0, size=5)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=5)
    weights = rng.normal(size=5)
    weights /= np.sum(np.abs(weights))
    field = np.sin(
        normalized @ directions.T * frequencies[None, :] + phases[None, :]
    ) @ weights

    scale = 1.0 + rng.uniform(-2.0 * amplitude, 2.0 * amplitude, size=3)
    warped = center + (vertices - center) * scale
    warped += normals * (amplitude * diagonal * field[:, None])
    return np.ascontiguousarray(warped, dtype=np.float64)


def make_visacd_mesh(vertices: np.ndarray, faces: np.ndarray):
    try:
        return visacd.Mesh(
            np.ascontiguousarray(vertices),
            np.ascontiguousarray(faces),
        )
    except TypeError:
        pass
    mesh = visacd.Mesh()
    mesh.vertices = visacd.VecArray3d(vertices.tolist())
    mesh.triangles = visacd.make_vecarray3i(faces)
    return mesh


def make_variants(args: argparse.Namespace):
    source = trimesh.load(args.mesh, force="mesh")
    if not isinstance(source, trimesh.Trimesh):
        raise TypeError(f"{args.mesh} did not load as a triangle mesh")
    vertices = np.asarray(source.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(source.faces, dtype=np.int32)
    normals = np.asarray(source.vertex_normals, dtype=np.float64)
    bounds = np.asarray(source.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("source mesh has a degenerate bounding box")

    variants = []
    hashes = set()
    rms_displacements = []
    max_displacements = []
    for index in range(args.n):
        rng = np.random.default_rng(
            np.random.SeedSequence([args.perturb_seed, index])
        )
        warped = smooth_variant(
            vertices,
            normals,
            center,
            diagonal,
            args.amplitude,
            rng,
        )
        displacement = np.linalg.norm(warped - vertices, axis=1) / diagonal
        rms_displacements.append(float(np.sqrt(np.mean(displacement**2))))
        max_displacements.append(float(np.max(displacement)))
        hashes.add(hashlib.sha256(warped.tobytes()).digest())
        variants.append(make_visacd_mesh(warped, faces))
    return variants, len(hashes), rms_displacements, max_displacements


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
    print("Generating variants outside the timed region...", flush=True)
    meshes, unique_count, rms_displacements, max_displacements = make_variants(args)
    print(
        f"Source: {args.mesh}; variants: {len(meshes)}; "
        f"unique geometry hashes: {unique_count}"
    )
    print(
        "Displacement / bbox diagonal: "
        f"median RMS {statistics.median(rms_displacements) * 100:.2f}%, "
        f"maximum {max(max_displacements) * 100:.2f}%"
    )

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
