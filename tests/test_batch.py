import hashlib
import os
from pathlib import Path
import struct
import unittest

import numpy as np
import trimesh

import visacd


ROOT = Path(__file__).resolve().parents[1]
RUN_GPU_TESTS = os.environ.get("VISACD_RUN_GPU_TESTS") == "1"


def load_sample(name, x_offset=0.0):
    tm = trimesh.load(ROOT / "data" / "samples" / name, force="mesh")
    vertices = np.asarray(tm.vertices, dtype=np.float64).copy()
    vertices[:, 0] += x_offset

    mesh = visacd.Mesh()
    mesh.vertices = visacd.VecArray3d(vertices.tolist())
    mesh.triangles = visacd.make_vecarray3i(
        np.asarray(tm.faces, dtype=np.int32)
    )
    return mesh


def load_cow(x_offset=0.0):
    return load_sample("cow.obj", x_offset)


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


def result_x_bounds(result):
    vertices = [
        np.asarray(list(part.vertices), dtype=np.float64)
        for part in result.parts
    ]
    x_coordinates = np.concatenate([part[:, 0] for part in vertices])
    return x_coordinates.min(), x_coordinates.max()


class BatchValidationTests(unittest.TestCase):
    def setUp(self):
        self.max_batch_size = visacd.config.max_batch_size
        self.batch_memory_fraction = visacd.config.batch_memory_fraction
        self.batch_cpu_threads = visacd.config.batch_cpu_threads

    def tearDown(self):
        visacd.config.max_batch_size = self.max_batch_size
        visacd.config.batch_memory_fraction = self.batch_memory_fraction
        visacd.config.batch_cpu_threads = self.batch_cpu_threads

    def test_empty_batch(self):
        self.assertEqual(visacd.process_batch([], 0.04, 2), [])

    def test_invalid_common_parameters(self):
        with self.assertRaisesRegex(ValueError, "num_parts"):
            visacd.process_batch([], 0.04, 0)
        with self.assertRaisesRegex(ValueError, "concavity"):
            visacd.process_batch([], float("nan"), 2)

    def test_empty_mesh_is_rejected_before_gpu_initialization(self):
        with self.assertRaisesRegex(ValueError, "triangles"):
            visacd.process_batch([visacd.Mesh()], 0.04, 2)

    def test_invalid_batch_controls(self):
        visacd.config.batch_cpu_threads = -1
        with self.assertRaisesRegex(ValueError, "batch_cpu_threads"):
            visacd.process_batch([], 0.04, 2)

        visacd.config.batch_cpu_threads = 0
        visacd.config.max_batch_size = -1
        with self.assertRaisesRegex(ValueError, "max_batch_size"):
            visacd.process_batch([], 0.04, 2)

        visacd.config.max_batch_size = 0
        for value in (0.0, -0.1, 1.1):
            with self.subTest(batch_memory_fraction=value):
                visacd.config.batch_memory_fraction = value
                with self.assertRaisesRegex(ValueError, "batch_memory_fraction"):
                    visacd.process_batch([], 0.04, 2)


@unittest.skipUnless(
    RUN_GPU_TESTS,
    "set VISACD_RUN_GPU_TESTS=1 to run CUDA/OptiX integration tests",
)
class BatchGpuTests(unittest.TestCase):
    def setUp(self):
        self.saved_config = {
            "return_parts": visacd.config.return_parts,
            "score_mode": visacd.config.score_mode,
            "use_flat_surfaces": visacd.config.use_flat_surfaces,
            "use_merging": visacd.config.use_merging,
            "max_batch_size": visacd.config.max_batch_size,
            "batch_memory_fraction": visacd.config.batch_memory_fraction,
            "batch_cpu_threads": visacd.config.batch_cpu_threads,
        }
        visacd.config.return_parts = False
        visacd.config.score_mode = "concavity"
        visacd.config.use_flat_surfaces = False
        visacd.config.use_merging = False
        visacd.config.batch_memory_fraction = 0.7
        visacd.config.batch_cpu_threads = 0

    def tearDown(self):
        for name, value in self.saved_config.items():
            setattr(visacd.config, name, value)

    def run_batch(
        self,
        max_batch_size,
        batch_cpu_threads=0,
        batch_memory_fraction=0.7,
    ):
        visacd.config.max_batch_size = max_batch_size
        visacd.config.batch_cpu_threads = batch_cpu_threads
        visacd.config.batch_memory_fraction = batch_memory_fraction
        visacd.set_seed(1234)
        return visacd.process_batch(
            [load_cow(-100.0), load_cow(100.0)],
            concavity=0.04,
            num_parts=2,
        )

    def test_order_repeatability_and_forced_waves(self):
        automatic = self.run_batch(max_batch_size=0)
        repeated = self.run_batch(max_batch_size=0)
        one_request_waves = self.run_batch(max_batch_size=1)
        one_cpu_thread = self.run_batch(
            max_batch_size=0, batch_cpu_threads=1
        )
        host_packed_fallback = self.run_batch(
            max_batch_size=0,
            batch_memory_fraction=1e-9,
        )

        self.assertEqual(len(automatic), 2)
        self.assertEqual(result_digest(automatic), result_digest(repeated))
        self.assertEqual(
            result_digest(automatic),
            result_digest(one_request_waves),
        )
        self.assertEqual(
            result_digest(automatic),
            result_digest(one_cpu_thread),
        )
        self.assertEqual(
            result_digest(automatic),
            result_digest(host_packed_fallback),
        )

        first_bounds = result_x_bounds(automatic[0])
        second_bounds = result_x_bounds(automatic[1])
        self.assertLess(first_bounds[1], 0.0)
        self.assertGreater(second_bounds[0], 0.0)

    def test_selection_hull_is_closed_and_convex(self):
        visacd.config.max_batch_size = 0
        visacd.set_seed(9876)
        result = visacd.process_batch(
            [load_cow()],
            concavity=10.0,
            num_parts=2,
        )[0]

        self.assertEqual(result.num_parts, 1)
        hull = result.parts[0]
        hull_mesh = trimesh.Trimesh(
            vertices=np.asarray(list(hull.vertices), dtype=np.float64),
            faces=np.asarray(list(hull.triangles), dtype=np.int32),
            process=False,
        )
        self.assertTrue(hull_mesh.is_watertight)
        self.assertTrue(hull_mesh.is_winding_consistent)
        vertices = np.asarray(hull_mesh.vertices, dtype=np.float64)
        for triangle in np.asarray(hull_mesh.faces, dtype=np.int32):
            first, second, third = vertices[triangle]
            normal = np.cross(second - first, third - first)
            tolerance = 1e-8 * np.linalg.norm(normal)
            signed_distances = (vertices - first) @ normal
            self.assertLessEqual(signed_distances.max(), tolerance)

    def test_flat_surface_pipeline_is_repeatable(self):
        def run(
            max_batch_size,
            batch_cpu_threads=0,
            batch_memory_fraction=0.7,
        ):
            visacd.config.use_flat_surfaces = True
            visacd.config.max_batch_size = max_batch_size
            visacd.config.batch_cpu_threads = batch_cpu_threads
            visacd.config.batch_memory_fraction = batch_memory_fraction
            visacd.set_seed(2468)
            return visacd.process_batch(
                [
                    load_sample("fandisk.obj"),
                    load_sample("cow.obj"),
                ],
                concavity=0.04,
                num_parts=2,
            )

        automatic = run(max_batch_size=0)
        repeated = run(max_batch_size=0)
        one_request_waves = run(max_batch_size=1)
        one_cpu_thread = run(
            max_batch_size=0,
            batch_cpu_threads=1,
        )
        low_memory = run(
            max_batch_size=0,
            batch_memory_fraction=1e-9,
        )

        digest = result_digest(automatic)
        self.assertEqual(digest, result_digest(repeated))
        self.assertEqual(digest, result_digest(one_request_waves))
        self.assertEqual(digest, result_digest(one_cpu_thread))
        self.assertEqual(digest, result_digest(low_memory))

    def test_mixed_mesh_pipeline_is_repeatable(self):
        def run(
            max_batch_size,
            batch_cpu_threads,
            batch_memory_fraction=0.7,
        ):
            visacd.config.max_batch_size = max_batch_size
            visacd.config.batch_cpu_threads = batch_cpu_threads
            visacd.config.batch_memory_fraction = batch_memory_fraction
            visacd.set_seed(4321)
            return visacd.process_batch(
                [
                    load_sample("Bottle.obj"),
                    load_sample("teapot.obj"),
                    load_sample("cow.obj"),
                ],
                concavity=0.04,
                num_parts=4,
            )

        automatic = run(max_batch_size=0, batch_cpu_threads=0)
        repeated = run(max_batch_size=0, batch_cpu_threads=0)
        one_request_waves = run(max_batch_size=1, batch_cpu_threads=0)
        one_cpu_thread = run(max_batch_size=0, batch_cpu_threads=1)
        host_packed_fallback = run(
            max_batch_size=0,
            batch_cpu_threads=0,
            batch_memory_fraction=1e-9,
        )

        expected_digest = result_digest(automatic)
        self.assertEqual(expected_digest, result_digest(repeated))
        self.assertEqual(expected_digest, result_digest(one_request_waves))
        self.assertEqual(expected_digest, result_digest(one_cpu_thread))
        self.assertEqual(expected_digest, result_digest(host_packed_fallback))


if __name__ == "__main__":
    unittest.main()
