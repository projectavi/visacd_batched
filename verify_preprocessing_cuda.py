#!/usr/bin/env python3
"""Compare CUDA surface voxelization with the exact OpenVDB 8.2 reference."""

import argparse
from pathlib import Path

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
        "--scale",
        type=float,
        action="append",
        dest="scales",
        help="preprocessing scale; may be repeated (defaults: 20, 30, 40)",
    )
    parser.add_argument("--memory-fraction", type=float, default=0.7)
    parser.add_argument(
        "--full-output",
        action="store_true",
        help=(
            "also compare complete OpenVDB remeshing at every production "
            "scale/level-set configuration"
        ),
    )
    parser.add_argument(
        "--packed",
        action="store_true",
        help="also verify all meshes through packed CUDA waves",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=0,
        help="forced packed wave size; 0 uses memory-aware sizing",
    )
    return parser.parse_args()


def load_mesh(path):
    source = trimesh.load(path, force="mesh")
    if not isinstance(source, trimesh.Trimesh):
        raise TypeError("{} did not load as a triangle mesh".format(path))
    return visacd.Mesh(
        np.ascontiguousarray(source.vertices, dtype=np.float64),
        np.ascontiguousarray(source.faces, dtype=np.int32),
    )


def main():
    args = parse_args()
    paths = args.meshes or sorted((ROOT / "data" / "samples").glob("*.obj"))
    scales = args.scales or [20.0, 30.0, 40.0]
    failures = 0
    exact = 0
    fallbacks = 0
    for path in paths:
        mesh = load_mesh(path)
        for scale in scales:
            result = visacd._verify_preprocess_voxelization(
                mesh, scale, args.memory_fraction
            )
            print(
                "mesh={} scale={} supported={} exact={} "
                "reference_voxels={} candidate_voxels={} "
                "coordinate_mismatches={} distance_mismatches={} "
                "triangle_mismatches={} reference_ms={:.3f} "
                "cuda_ms={:.3f} fallback_reason={}".format(
                    path,
                    scale,
                    result["supported"],
                    result["exact"],
                    result["reference_voxels"],
                    result["candidate_voxels"],
                    result["coordinate_mismatches"],
                    result["distance_mismatches"],
                    result["triangle_mismatches"],
                    result["reference_ms"],
                    result["cuda_ms"],
                    result["fallback_reason"],
                ),
                flush=True,
            )
            if result["exact"]:
                exact += 1
            elif not result["supported"]:
                fallbacks += 1
            else:
                failures += 1
    print(
        "summary cases={} exact={} fallbacks={} failures={}".format(
            exact + fallbacks + failures, exact, fallbacks, failures
        )
    )

    if args.packed:
        packed_exact = 0
        packed_fallbacks = 0
        packed_failures = 0
        meshes = [load_mesh(path) for path in paths]
        for scale in scales:
            comparison = visacd._verify_preprocess_voxelization_batch(
                meshes,
                scale,
                args.max_batch_size,
                args.memory_fraction,
            )
            for path, result in zip(paths, comparison["cases"]):
                print(
                    "packed_mesh={} scale={} supported={} exact={} "
                    "reference_voxels={} candidate_voxels={} "
                    "coordinate_mismatches={} distance_mismatches={} "
                    "triangle_mismatches={} wave_ms={:.3f} "
                    "fallback_reason={}".format(
                        path,
                        scale,
                        result["supported"],
                        result["exact"],
                        result["reference_voxels"],
                        result["candidate_voxels"],
                        result["coordinate_mismatches"],
                        result["distance_mismatches"],
                        result["triangle_mismatches"],
                        result["cuda_wave_ms"],
                        result["fallback_reason"],
                    ),
                    flush=True,
                )
                if result["exact"]:
                    packed_exact += 1
                elif not result["supported"]:
                    packed_fallbacks += 1
                else:
                    packed_failures += 1
        print(
            "packed_summary cases={} exact={} fallbacks={} failures={}".format(
                packed_exact + packed_fallbacks + packed_failures,
                packed_exact,
                packed_fallbacks,
                packed_failures,
            )
        )
        failures += packed_failures

    if args.full_output:
        configurations = [
            (20.0, 0.55 / 20.0),
            (30.0, 0.55 / 30.0),
            (40.0, 0.03),
            (40.0, 0.02),
        ]
        full_exact = 0
        full_fallbacks = 0
        full_failures = 0
        for path in paths:
            mesh = load_mesh(path)
            for scale, level_set in configurations:
                result = visacd._verify_manifold_preprocessing(
                    mesh, scale, level_set
                )
                print(
                    "full_mesh={} scale={} level_set={} supported={} "
                    "exact={} reference_vertices={} candidate_vertices={} "
                    "reference_triangles={} candidate_triangles={} "
                    "vertex_mismatches={} triangle_mismatches={} "
                    "reference_ms={:.3f} candidate_ms={:.3f} "
                    "fallback_reason={}".format(
                        path,
                        scale,
                        level_set,
                        result["supported"],
                        result["exact"],
                        result["reference_vertices"],
                        result["candidate_vertices"],
                        result["reference_triangles"],
                        result["candidate_triangles"],
                        result["vertex_mismatches"],
                        result["triangle_mismatches"],
                        result["reference_ms"],
                        result["candidate_ms"],
                        result["fallback_reason"],
                    ),
                    flush=True,
                )
                if result["exact"]:
                    full_exact += 1
                elif not result["supported"]:
                    full_fallbacks += 1
                else:
                    full_failures += 1
        print(
            "full_summary cases={} exact={} fallbacks={} failures={}".format(
                full_exact + full_fallbacks + full_failures,
                full_exact,
                full_fallbacks,
                full_failures,
            )
        )
        failures += full_failures
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
