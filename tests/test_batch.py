import ctypes
import hashlib
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np
import trimesh

import visacd


ROOT = Path(__file__).resolve().parents[1]
RUN_GPU_TESTS = os.environ.get("VISACD_RUN_GPU_TESTS") == "1"
CUDA_PREPROCESS_ENV = (
    "VISACD_ENABLE_CUDA_PREPROCESS",
    "VISACD_DISABLE_CUDA_PREPROCESS",
    "VISACD_VERIFY_CUDA_PREPROCESS",
    "VISACD_ENABLE_CUDA_PREPROCESS_SIGN",
    "VISACD_ENABLE_CUDA_PREPROCESS_EXPAND",
    "VISACD_ENABLE_CUDA_PREPROCESS_EXPAND_DENSE",
    "VISACD_ENABLE_CUDA_PREPROCESS_RENORMALIZE",
    "VISACD_ENABLE_CUDA_PREPROCESS_MESH",
    "VISACD_ENABLE_CUDA_PREPROCESS_RESIDENT",
)


def free_cuda_memory():
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    driver.cuDevicePrimaryCtxRetain.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
    ]
    driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
    driver.cuMemGetInfo_v2.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    if driver.cuInit(0) != 0:
        raise RuntimeError("cuInit failed")
    context = ctypes.c_void_p()
    if driver.cuCtxGetCurrent(ctypes.byref(context)) != 0:
        raise RuntimeError("cuCtxGetCurrent failed")
    if not context.value:
        if driver.cuDevicePrimaryCtxRetain(ctypes.byref(context), 0) != 0:
            raise RuntimeError("cuDevicePrimaryCtxRetain failed")
        if driver.cuCtxSetCurrent(context) != 0:
            raise RuntimeError("cuCtxSetCurrent failed")
    if driver.cuCtxSynchronize() != 0:
        raise RuntimeError("cuCtxSynchronize failed")
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    if driver.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)) != 0:
        raise RuntimeError("cuMemGetInfo_v2 failed")
    return free.value


def capture_native_output(callback):
    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile() as stdout_file:
        with tempfile.TemporaryFile() as stderr_file:
            saved_stdout = os.dup(sys.stdout.fileno())
            saved_stderr = os.dup(sys.stderr.fileno())
            try:
                os.dup2(stdout_file.fileno(), sys.stdout.fileno())
                os.dup2(stderr_file.fileno(), sys.stderr.fileno())
                result = callback()
                sys.stdout.flush()
                sys.stderr.flush()
                libc = ctypes.CDLL(None)
                libc.fflush.argtypes = [ctypes.c_void_p]
                libc.fflush.restype = ctypes.c_int
                libc.fflush(None)
            finally:
                os.dup2(saved_stdout, sys.stdout.fileno())
                os.dup2(saved_stderr, sys.stderr.fileno())
                os.close(saved_stdout)
                os.close(saved_stderr)
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read()
            stderr = stderr_file.read()
    return result, stdout, stderr


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
        self.part_limit_policy = visacd.config.part_limit_policy
        self.batch_logging = visacd.config.batch_logging

    def tearDown(self):
        visacd.config.max_batch_size = self.max_batch_size
        visacd.config.batch_memory_fraction = self.batch_memory_fraction
        visacd.config.batch_cpu_threads = self.batch_cpu_threads
        visacd.config.part_limit_policy = self.part_limit_policy
        visacd.config.batch_logging = self.batch_logging

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
        visacd.config.part_limit_policy = "invalid"
        with self.assertRaisesRegex(ValueError, "part_limit_policy"):
            visacd.process_batch([], 0.04, 2)

        visacd.config.part_limit_policy = "split_budget"
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

    def test_numpy_mesh_constructor_and_buffer_views(self):
        vertices = np.arange(12, dtype=np.float64).reshape(4, 3)
        triangles = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        mesh = visacd.Mesh(vertices, triangles)

        vertex_view = np.asarray(mesh.vertices)
        triangle_view = np.asarray(mesh.triangles)
        self.assertEqual(vertex_view.shape, (4, 3))
        self.assertEqual(triangle_view.shape, (2, 3))
        self.assertFalse(vertex_view.flags.owndata)
        self.assertFalse(triangle_view.flags.owndata)
        vertex_view[0, 0] = 99.0
        self.assertEqual(np.asarray(mesh.vertices)[0, 0], 99.0)

        vertices[1, 1] = -1.0
        self.assertNotEqual(np.asarray(mesh.vertices)[1, 1], -1.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            visacd.Mesh(np.zeros((3, 2)), triangles)


@unittest.skipUnless(
    RUN_GPU_TESTS,
    "set VISACD_RUN_GPU_TESTS=1 to run CUDA/OptiX integration tests",
)
class BatchGpuTests(unittest.TestCase):
    def setUp(self):
        self.saved_cuda_preprocess_env = {
            name: os.environ.get(name) for name in CUDA_PREPROCESS_ENV
        }
        self.saved_config = {
            "return_parts": visacd.config.return_parts,
            "score_mode": visacd.config.score_mode,
            "use_flat_surfaces": visacd.config.use_flat_surfaces,
            "use_merging": visacd.config.use_merging,
            "part_limit_policy": visacd.config.part_limit_policy,
            "max_batch_size": visacd.config.max_batch_size,
            "batch_memory_fraction": visacd.config.batch_memory_fraction,
            "batch_logging": visacd.config.batch_logging,
            "batch_cpu_threads": visacd.config.batch_cpu_threads,
            "retain_gpu_resources": visacd.config.retain_gpu_resources,
        }
        visacd.config.return_parts = False
        visacd.config.score_mode = "concavity"
        visacd.config.use_flat_surfaces = False
        visacd.config.use_merging = False
        visacd.config.part_limit_policy = "split_budget"
        visacd.config.batch_logging = True
        visacd.config.batch_memory_fraction = 0.7
        visacd.config.batch_cpu_threads = 0
        visacd.config.retain_gpu_resources = True

    def tearDown(self):
        for name, value in self.saved_cuda_preprocess_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        for name, value in self.saved_config.items():
            setattr(visacd.config, name, value)

    def set_cuda_preprocess_flags(self, *enabled):
        enabled = set(enabled)
        for name in CUDA_PREPROCESS_ENV:
            if name in enabled:
                os.environ[name] = "1"
            else:
                os.environ.pop(name, None)

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

    def test_gpu_resource_release_preserves_results(self):
        visacd.config.retain_gpu_resources = True
        retained = self.run_batch(max_batch_size=0)
        retained_digest = result_digest(retained)

        visacd.release_gpu_resources()
        visacd.release_gpu_resources()

        visacd.config.retain_gpu_resources = False
        released = self.run_batch(max_batch_size=0)
        self.assertEqual(retained_digest, result_digest(released))

        # Automatic cleanup also runs when validation rejects a batch, and an
        # explicit release remains safe afterward.
        with self.assertRaisesRegex(ValueError, "num_parts"):
            visacd.process_batch([], 0.04, 0)
        visacd.release_gpu_resources()

        visacd.config.retain_gpu_resources = True

    def test_gpu_resource_release_recovers_vram(self):
        visacd.release_gpu_resources()
        baseline = free_cuda_memory()

        visacd.config.retain_gpu_resources = True
        retained_results = self.run_batch(max_batch_size=0)
        retained = free_cuda_memory()
        self.assertEqual(len(retained_results), 2)

        visacd.release_gpu_resources()
        released = free_cuda_memory()

        mib = 1024 * 1024
        self.assertGreater(
            released - retained,
            64 * mib,
            "explicit release did not recover meaningful device memory",
        )
        self.assertLess(
            abs(baseline - released),
            256 * mib,
            "released VRAM did not return near the initialized baseline",
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

    def test_cuda_surface_voxelization_matches_openvdb(self):
        for sample in ("cow.obj", "KitchenPot.obj", "armadillo.obj"):
            mesh = load_sample(sample)
            for scale in (20.0, 30.0, 40.0):
                with self.subTest(sample=sample, scale=scale):
                    comparison = visacd._verify_preprocess_voxelization(
                        mesh, scale
                    )
                    self.assertTrue(comparison["supported"])
                    self.assertTrue(comparison["exact"])
                    self.assertEqual(
                        comparison["coordinate_mismatches"], 0
                    )
                    self.assertEqual(comparison["distance_mismatches"], 0)
                    self.assertEqual(comparison["triangle_mismatches"], 0)

        fallback = visacd._verify_preprocess_voxelization(
            load_cow(), 30.0, 1e-12
        )
        self.assertFalse(fallback["supported"])
        self.assertIn("memory budget", fallback["fallback_reason"])

    def test_cuda_batched_surface_voxelization_matches_openvdb(self):
        meshes = [
            load_sample(name)
            for name in ("cow.obj", "KitchenPot.obj", "armadillo.obj")
        ]
        for max_batch_size in (0, 1, 3, 200):
            for scale in (20.0, 30.0, 40.0):
                with self.subTest(
                    max_batch_size=max_batch_size, scale=scale
                ):
                    comparison = (
                        visacd._verify_preprocess_voxelization_batch(
                            meshes, scale, max_batch_size
                        )
                    )
                    self.assertEqual(len(comparison["cases"]), len(meshes))
                    for case in comparison["cases"]:
                        self.assertTrue(case["supported"])
                        self.assertTrue(case["exact"])
                        self.assertEqual(case["coordinate_mismatches"], 0)
                        self.assertEqual(case["distance_mismatches"], 0)
                        self.assertEqual(case["triangle_mismatches"], 0)

        low_memory = visacd._verify_preprocess_voxelization_batch(
            meshes, 30.0, 0, 1e-12
        )
        for case in low_memory["cases"]:
            self.assertFalse(case["supported"])
            self.assertIn("memory budget", case["fallback_reason"])

    def test_cuda_manifold_preprocessing_matches_openvdb(self):
        configurations = (
            (20.0, 0.55 / 20.0),
            (30.0, 0.55 / 30.0),
            (40.0, 0.03),
            (40.0, 0.02),
        )
        for sample in ("cow.obj", "KitchenPot.obj", "armadillo.obj"):
            mesh = load_sample(sample)
            for scale, level_set in configurations:
                with self.subTest(
                    sample=sample, scale=scale, level_set=level_set
                ):
                    comparison = visacd._verify_manifold_preprocessing(
                        mesh, scale, level_set
                    )
                    self.assertTrue(comparison["supported"])
                    self.assertTrue(comparison["exact"])
                    self.assertEqual(comparison["vertex_mismatches"], 0)
                    self.assertEqual(comparison["triangle_mismatches"], 0)

    def test_cuda_manifold_preprocessing_stage_variants_match_openvdb(self):
        variants = {
            "signed_flood": (
                "VISACD_ENABLE_CUDA_PREPROCESS_SIGN",
            ),
            "sparse_expand": (
                "VISACD_ENABLE_CUDA_PREPROCESS_EXPAND",
            ),
            "sparse_renormalize": (
                "VISACD_ENABLE_CUDA_PREPROCESS_RENORMALIZE",
            ),
            "volume_meshing": (
                "VISACD_ENABLE_CUDA_PREPROCESS_MESH",
            ),
            "resident_dense_chain": (
                "VISACD_ENABLE_CUDA_PREPROCESS_EXPAND_DENSE",
                "VISACD_ENABLE_CUDA_PREPROCESS_RENORMALIZE",
                "VISACD_ENABLE_CUDA_PREPROCESS_MESH",
                "VISACD_ENABLE_CUDA_PREPROCESS_RESIDENT",
            ),
            "complete_cuda_chain": (
                "VISACD_ENABLE_CUDA_PREPROCESS_SIGN",
                "VISACD_ENABLE_CUDA_PREPROCESS_EXPAND_DENSE",
                "VISACD_ENABLE_CUDA_PREPROCESS_RENORMALIZE",
                "VISACD_ENABLE_CUDA_PREPROCESS_MESH",
                "VISACD_ENABLE_CUDA_PREPROCESS_RESIDENT",
            ),
        }
        mesh = load_cow()
        for variant, flags in variants.items():
            with self.subTest(variant=variant):
                self.set_cuda_preprocess_flags(*flags)
                comparison = visacd._verify_manifold_preprocessing(
                    mesh, 40.0, 0.02
                )
                self.assertTrue(comparison["supported"])
                self.assertTrue(comparison["exact"])
                self.assertEqual(comparison["vertex_mismatches"], 0)
                self.assertEqual(comparison["triangle_mismatches"], 0)

    def test_cuda_preprocessing_preserves_decomposition_output(self):
        self.set_cuda_preprocess_flags("VISACD_DISABLE_CUDA_PREPROCESS")
        visacd.config.max_batch_size = 0
        visacd.config.batch_cpu_threads = 0
        visacd.set_seed(1234)
        reference = visacd.process_batch(
            [load_cow(-100.0), load_cow(100.0)], 0.04, 2
        )

        self.set_cuda_preprocess_flags("VISACD_ENABLE_CUDA_PREPROCESS")
        visacd.set_seed(1234)
        candidate = visacd.process_batch(
            [load_cow(-100.0), load_cow(100.0)], 0.04, 2
        )
        visacd.config.max_batch_size = 1
        visacd.config.batch_cpu_threads = 1
        visacd.set_seed(1234)
        forced_waves = visacd.process_batch(
            [load_cow(-100.0), load_cow(100.0)], 0.04, 2
        )
        self.set_cuda_preprocess_flags(
            "VISACD_ENABLE_CUDA_PREPROCESS",
            "VISACD_ENABLE_CUDA_PREPROCESS_SIGN",
            "VISACD_ENABLE_CUDA_PREPROCESS_EXPAND_DENSE",
            "VISACD_ENABLE_CUDA_PREPROCESS_RENORMALIZE",
            "VISACD_ENABLE_CUDA_PREPROCESS_MESH",
            "VISACD_ENABLE_CUDA_PREPROCESS_RESIDENT",
        )
        visacd.config.max_batch_size = 0
        visacd.config.batch_cpu_threads = 0
        visacd.set_seed(1234)
        complete_cuda_chain = visacd.process_batch(
            [load_cow(-100.0), load_cow(100.0)], 0.04, 2
        )
        self.assertEqual(result_digest(reference), result_digest(candidate))
        self.assertEqual(
            result_digest(reference), result_digest(forced_waves)
        )
        self.assertEqual(
            result_digest(reference), result_digest(complete_cuda_chain)
        )

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

    def test_batch_logging_can_silence_all_native_output(self):
        diagnostic_names = (
            "VISACD_STAGE_TIMING",
            "VISACD_PREPROCESS_TRACE",
            "VISACD_CLIP_COMPACTION_DIAGNOSTICS",
            "VISACD_HULL_TOPOLOGY_DIAGNOSTICS",
        )
        saved_diagnostics = {
            name: os.environ.get(name) for name in diagnostic_names
        }
        for name in diagnostic_names:
            os.environ[name] = "1"

        mesh = load_cow()
        visacd.config.part_limit_policy = "adjacent_merge"
        visacd.config.score_mode = "edge"
        try:
            visacd.config.batch_logging = False
            visacd.set_seed(31415)
            quiet, quiet_stdout, quiet_stderr = capture_native_output(
                lambda: visacd.process_batch([mesh], 0.2, 4)
            )

            visacd.config.batch_logging = True
            visacd.set_seed(31415)
            verbose, verbose_stdout, verbose_stderr = capture_native_output(
                lambda: visacd.process_batch([mesh], 0.2, 4)
            )
        finally:
            for name, value in saved_diagnostics.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

        self.assertEqual(quiet_stdout, b"")
        self.assertEqual(quiet_stderr, b"")
        self.assertIn(b"[visacd batch 0]", verbose_stdout)
        self.assertIn(b"[visacd stages]", verbose_stderr)
        self.assertEqual(result_digest(quiet), result_digest(verbose))

    def run_adjacent_limit(
        self,
        max_batch_size=0,
        batch_cpu_threads=0,
        batch_memory_fraction=0.7,
        score_mode="edge",
        return_parts=False,
        use_merging=False,
    ):
        visacd.config.part_limit_policy = "adjacent_merge"
        visacd.config.score_mode = score_mode
        visacd.config.return_parts = return_parts
        visacd.config.use_merging = use_merging
        visacd.config.max_batch_size = max_batch_size
        visacd.config.batch_cpu_threads = batch_cpu_threads
        visacd.config.batch_memory_fraction = batch_memory_fraction
        visacd.set_seed(31415)
        return visacd.process_batch(
            [load_cow(-100.0), load_cow(100.0)],
            concavity=0.2 if score_mode == "edge" else 0.04,
            num_parts=4,
        )

    def test_adjacent_part_limit_repeatability_and_fallbacks(self):
        automatic = self.run_adjacent_limit()
        repeated = self.run_adjacent_limit()
        one_request_waves = self.run_adjacent_limit(max_batch_size=1)
        one_cpu_thread = self.run_adjacent_limit(batch_cpu_threads=1)
        host_packed_fallback = self.run_adjacent_limit(
            batch_memory_fraction=1e-9
        )

        expected_digest = result_digest(automatic)
        self.assertEqual([result.num_parts for result in automatic], [4, 4])
        self.assertEqual(expected_digest, result_digest(repeated))
        self.assertEqual(expected_digest, result_digest(one_request_waves))
        self.assertEqual(expected_digest, result_digest(one_cpu_thread))
        self.assertEqual(
            expected_digest, result_digest(host_packed_fallback)
        )
        self.assertLess(result_x_bounds(automatic[0])[1], 0.0)
        self.assertGreater(result_x_bounds(automatic[1])[0], 0.0)

    def test_adjacent_part_limit_all_modes_are_seeded_and_bounded(self):
        for score_mode in ("edge", "concavity"):
            for return_parts in (False, True):
                for use_merging in (False, True):
                    with self.subTest(
                        score_mode=score_mode,
                        return_parts=return_parts,
                        use_merging=use_merging,
                    ):
                        first = self.run_adjacent_limit(
                            score_mode=score_mode,
                            return_parts=return_parts,
                            use_merging=use_merging,
                        )
                        second = self.run_adjacent_limit(
                            score_mode=score_mode,
                            return_parts=return_parts,
                            use_merging=use_merging,
                        )
                        self.assertTrue(
                            all(result.num_parts <= 4 for result in first)
                        )
                        self.assertEqual(
                            result_digest(first), result_digest(second)
                        )

    def test_adjacent_part_limit_preserves_source_geometry(self):
        visacd.config.score_mode = "edge"
        visacd.config.return_parts = True
        visacd.config.use_merging = False
        visacd.config.part_limit_policy = "split_budget"
        visacd.set_seed(31415)
        legacy = visacd.process_batch([load_cow()], 0.2, 4)[0]

        visacd.config.part_limit_policy = "adjacent_merge"
        visacd.set_seed(31415)
        limited = visacd.process_batch([load_cow()], 0.2, 4)[0]

        self.assertGreater(legacy.num_parts, 4)
        self.assertEqual(limited.num_parts, 4)
        self.assertEqual(
            sum(len(part.triangles) for part in legacy.parts),
            sum(len(part.triangles) for part in limited.parts),
        )
        for part in limited.parts:
            vertices = np.asarray(part.vertices, dtype=np.float64)
            triangles = np.asarray(part.triangles, dtype=np.int64)
            self.assertTrue(np.isfinite(vertices).all())
            self.assertGreaterEqual(triangles.min(), 0)
            self.assertLess(triangles.max(), len(vertices))

    def test_adjacent_part_limit_handles_infinite_cost_fallback(self):
        visacd.config.part_limit_policy = "adjacent_merge"
        visacd.config.score_mode = "edge"
        visacd.config.use_flat_surfaces = True
        visacd.set_seed(2026)
        result = visacd.process_batch(
            [load_sample("Octocat-v2.obj")], 0.2, 4
        )[0]
        self.assertEqual(result.num_parts, 4)
        self.assertTrue(np.isfinite(result.concavity))

    def test_adjacent_part_limit_rejects_initially_disconnected_mesh(self):
        first = trimesh.creation.icosphere(subdivisions=1)
        second = first.copy()
        second.apply_translation([4.0, 0.0, 0.0])
        disconnected = trimesh.util.concatenate([first, second])
        mesh = visacd.Mesh(
            np.asarray(disconnected.vertices, dtype=np.float64),
            np.asarray(disconnected.faces, dtype=np.int32),
        )
        visacd.config.part_limit_policy = "adjacent_merge"
        with self.assertRaisesRegex(
            ValueError,
            "initial disconnected component count 2 exceeds num_parts 1",
        ):
            visacd.process_batch([mesh], 0.04, 1)

    def test_merge_pipeline_is_repeatable(self):
        def run(
            max_batch_size,
            batch_cpu_threads=0,
            batch_memory_fraction=0.7,
        ):
            visacd.config.score_mode = "edge"
            visacd.config.use_merging = True
            visacd.config.max_batch_size = max_batch_size
            visacd.config.batch_cpu_threads = batch_cpu_threads
            visacd.config.batch_memory_fraction = batch_memory_fraction
            visacd.set_seed(31415)
            return visacd.process_batch(
                [load_cow(-100.0), load_cow(100.0)],
                concavity=0.2,
                num_parts=4,
            )

        automatic = run(max_batch_size=0)
        repeated = run(max_batch_size=0)
        one_request_waves = run(max_batch_size=1)
        one_cpu_thread = run(
            max_batch_size=0,
            batch_cpu_threads=1,
        )
        host_packed_fallback = run(
            max_batch_size=0,
            batch_memory_fraction=1e-9,
        )

        expected_digest = result_digest(automatic)
        self.assertEqual([result.num_parts for result in automatic], [5, 5])
        self.assertEqual(expected_digest, result_digest(repeated))
        self.assertEqual(expected_digest, result_digest(one_request_waves))
        self.assertEqual(expected_digest, result_digest(one_cpu_thread))
        self.assertEqual(
            expected_digest,
            result_digest(host_packed_fallback),
        )
        self.assertLess(result_x_bounds(automatic[0])[1], 0.0)
        self.assertGreater(result_x_bounds(automatic[1])[0], 0.0)


if __name__ == "__main__":
    unittest.main()
