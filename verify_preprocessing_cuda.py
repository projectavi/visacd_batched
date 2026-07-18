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
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
