#include <config.hpp>
#include <core.hpp>
#include <preprocess.hpp>
#include <preprocess_cuda.hpp>
#include <chrono>
#include <cstring>
#include <iostream>
#include <process.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 3>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::array<int, 3>>);

namespace {

template <typename T>
std::vector<std::array<T, 3>> copy_triplets(
    py::array_t<T, py::array::c_style | py::array::forcecast> input,
    const char *name) {
    static_assert(sizeof(std::array<T, 3>) == 3 * sizeof(T));
    const py::buffer_info buffer = input.request();
    if (buffer.ndim != 2 || buffer.shape[1] != 3) {
        throw py::value_error(std::string(name) +
                              " must have shape (N, 3)");
    }
    std::vector<std::array<T, 3>> result(
        static_cast<size_t>(buffer.shape[0]));
    if (!result.empty()) {
        std::memcpy(result.data(), buffer.ptr,
                    result.size() * sizeof(std::array<T, 3>));
    }
    return result;
}

} // namespace

PYBIND11_MODULE(visacd, m)
{
    auto vec_array_3d =
        py::bind_vector<std::vector<std::array<double, 3>>>(
            m, "VecArray3d", py::buffer_protocol());
    vec_array_3d.def_buffer(
        [](std::vector<std::array<double, 3>> &values) {
            return py::buffer_info(
                values.data(), sizeof(double),
                py::format_descriptor<double>::format(), 2,
                {values.size(), size_t{3}},
                {sizeof(std::array<double, 3>), sizeof(double)});
        });
    auto vec_array_3i = py::bind_vector<std::vector<std::array<int, 3>>>(
        m, "VecArray3i", py::buffer_protocol());
    vec_array_3i.def_buffer(
        [](std::vector<std::array<int, 3>> &values) {
            return py::buffer_info(
                values.data(), sizeof(int),
                py::format_descriptor<int>::format(), 2,
                {values.size(), size_t{3}},
                {sizeof(std::array<int, 3>), sizeof(int)});
        });

    py::class_<neural_acd::Mesh>(m, "Mesh")
        .def_readwrite("vertices", &neural_acd::Mesh::vertices)
        .def_readwrite("triangles", &neural_acd::Mesh::triangles)
        .def(py::init<>())
        .def(py::init([](
                          py::array_t<
                              double, py::array::c_style |
                                          py::array::forcecast> vertices,
                          py::array_t<
                              int, py::array::c_style |
                                       py::array::forcecast> triangles) {
            neural_acd::Mesh mesh;
            mesh.vertices = copy_triplets<double>(vertices, "vertices");
            mesh.triangles = copy_triplets<int>(triangles, "triangles");
            return mesh;
        }), py::arg("vertices"), py::arg("triangles"));

    py::bind_vector<neural_acd::MeshList>(m, "MeshList");

    py::class_<neural_acd::Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("return_parts", &neural_acd::Config::return_parts)
        .def_readwrite("score_mode", &neural_acd::Config::score_mode)
        .def_readwrite("flat_surface_min_area",
                       &neural_acd::Config::flat_surface_min_area)
        .def_readwrite("use_flat_surfaces",
                       &neural_acd::Config::use_flat_surfaces)
        .def_readwrite("flat_surface_k",
                       &neural_acd::Config::flat_surface_k)
        .def_readwrite("use_merging", &neural_acd::Config::use_merging)
        .def_readwrite("max_batch_size",
                       &neural_acd::Config::max_batch_size)
        .def_readwrite("batch_memory_fraction",
                       &neural_acd::Config::batch_memory_fraction)
        .def_readwrite("batch_cpu_threads",
                       &neural_acd::Config::batch_cpu_threads)
        .def_readwrite("retain_gpu_resources",
                       &neural_acd::Config::retain_gpu_resources);

    m.def("make_vecarray3d", [](py::array_t<
                                     double, py::array::c_style |
                                                 py::array::forcecast> input) {
        return copy_triplets<double>(input, "input");
    });
    m.def("make_vecarray3i", [](py::array_t<
                                     int, py::array::c_style |
                                              py::array::forcecast> input) {
        return copy_triplets<int>(input, "input");
    });

    m.def("set_seed", &neural_acd::set_seed, py::arg("seed"));

    m.def("_verify_preprocess_voxelization",
          [](neural_acd::Mesh mesh, double scale,
             double memory_fraction) {
              mesh.normalize();
              const auto reference_start =
                  std::chrono::steady_clock::now();
              const auto reference =
                  neural_acd::reference_surface_voxelization(mesh, scale);
              const double reference_ms =
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - reference_start)
                      .count();
              static thread_local neural_acd::ManifoldCudaRuntime runtime;
              auto candidate = runtime.voxelize_surface(
                  mesh, scale, memory_fraction);

              size_t coordinate_mismatches = 0;
              size_t distance_mismatches = 0;
              size_t triangle_mismatches = 0;
              unsigned long long maximum_ulp_difference = 0;
              double maximum_absolute_difference = 0.0;
              const size_t common =
                  std::min(reference.size(), candidate.records.size());
              for (size_t index = 0; index < common; ++index) {
                  const auto &expected = reference[index];
                  const auto &actual = candidate.records[index];
                  if (expected.x != actual.x || expected.y != actual.y ||
                      expected.z != actual.z) {
                      ++coordinate_mismatches;
                      continue;
                  }
                  if (std::memcmp(&expected.squared_distance,
                                  &actual.squared_distance,
                                  sizeof(double)) != 0) {
                      ++distance_mismatches;
                      unsigned long long expected_bits = 0;
                      unsigned long long actual_bits = 0;
                      std::memcpy(&expected_bits,
                                  &expected.squared_distance,
                                  sizeof(expected_bits));
                      std::memcpy(&actual_bits, &actual.squared_distance,
                                  sizeof(actual_bits));
                      const unsigned long long ulp_difference =
                          expected_bits > actual_bits
                              ? expected_bits - actual_bits
                              : actual_bits - expected_bits;
                      maximum_ulp_difference = std::max(
                          maximum_ulp_difference, ulp_difference);
                      maximum_absolute_difference = std::max(
                          maximum_absolute_difference,
                          std::abs(expected.squared_distance -
                                   actual.squared_distance));
                  }
                  if (expected.triangle_index != actual.triangle_index)
                      ++triangle_mismatches;
              }
              coordinate_mismatches +=
                  reference.size() > common
                      ? reference.size() - common
                      : candidate.records.size() - common;

              py::dict result;
              result["supported"] = candidate.supported;
              result["fallback_reason"] = candidate.fallback_reason;
              result["reference_voxels"] = reference.size();
              result["candidate_voxels"] = candidate.records.size();
              result["candidate_evaluations"] =
                  candidate.candidate_voxels;
              result["coordinate_mismatches"] = coordinate_mismatches;
              result["distance_mismatches"] = distance_mismatches;
              result["triangle_mismatches"] = triangle_mismatches;
              result["maximum_ulp_difference"] =
                  maximum_ulp_difference;
              result["maximum_absolute_difference"] =
                  maximum_absolute_difference;
              result["exact"] =
                  candidate.supported && coordinate_mismatches == 0 &&
                  distance_mismatches == 0 && triangle_mismatches == 0;
              result["reference_ms"] = reference_ms;
              result["cuda_ms"] = candidate.elapsed_ms;
              return result;
          },
          py::arg("mesh"), py::arg("scale"),
          py::arg("memory_fraction") = 0.7);

    m.def("_verify_preprocess_voxelization_batch",
          [](neural_acd::MeshList meshes, double scale,
             size_t max_batch_size, double memory_fraction) {
              std::vector<std::vector<neural_acd::SurfaceVoxelRecord>>
                  references(meshes.size());
              std::vector<neural_acd::SurfaceVoxelizationResult>
                  candidates(meshes.size());
              std::vector<neural_acd::SurfaceVoxelizationInput> inputs;
              inputs.reserve(meshes.size());
              for (size_t index = 0; index < meshes.size(); ++index) {
                  meshes[index].normalize();
                  references[index] =
                      neural_acd::reference_surface_voxelization(
                          meshes[index], scale);
                  inputs.push_back(
                      {&meshes[index], scale, &candidates[index]});
              }
              static thread_local
                  neural_acd::ManifoldCudaBatchRuntime runtime;
              const auto started = std::chrono::steady_clock::now();
              neural_acd::voxelize_surfaces_batch(
                  inputs, runtime, max_batch_size, memory_fraction);
              const double elapsed_ms =
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - started)
                      .count();

              py::list cases;
              for (size_t input_index = 0;
                   input_index < meshes.size(); ++input_index) {
                  const auto &reference = references[input_index];
                  const auto &candidate = candidates[input_index];
                  size_t coordinate_mismatches = 0;
                  size_t distance_mismatches = 0;
                  size_t triangle_mismatches = 0;
                  const size_t common = std::min(
                      reference.size(), candidate.records.size());
                  for (size_t index = 0; index < common; ++index) {
                      const auto &expected = reference[index];
                      const auto &actual = candidate.records[index];
                      if (expected.x != actual.x ||
                          expected.y != actual.y ||
                          expected.z != actual.z) {
                          ++coordinate_mismatches;
                          continue;
                      }
                      if (std::memcmp(&expected.squared_distance,
                                      &actual.squared_distance,
                                      sizeof(double)) != 0) {
                          ++distance_mismatches;
                      }
                      if (expected.triangle_index !=
                          actual.triangle_index) {
                          ++triangle_mismatches;
                      }
                  }
                  coordinate_mismatches +=
                      reference.size() > common
                          ? reference.size() - common
                          : candidate.records.size() - common;
                  py::dict item;
                  item["supported"] = candidate.supported;
                  item["fallback_reason"] = candidate.fallback_reason;
                  item["reference_voxels"] = reference.size();
                  item["candidate_voxels"] =
                      candidate.records.size();
                  item["coordinate_mismatches"] =
                      coordinate_mismatches;
                  item["distance_mismatches"] = distance_mismatches;
                  item["triangle_mismatches"] = triangle_mismatches;
                  item["exact"] =
                      candidate.supported &&
                      coordinate_mismatches == 0 &&
                      distance_mismatches == 0 &&
                      triangle_mismatches == 0;
                  item["cuda_wave_ms"] = candidate.elapsed_ms;
                  cases.append(std::move(item));
              }
              py::dict result;
              result["cases"] = std::move(cases);
              result["elapsed_ms"] = elapsed_ms;
              return result;
          },
          py::arg("meshes"), py::arg("scale"),
          py::arg("max_batch_size") = 0,
          py::arg("memory_fraction") = 0.7);

    m.def("_verify_manifold_preprocessing",
          [](neural_acd::Mesh mesh, double scale, double level_set) {
              mesh.normalize();
              neural_acd::Mesh reference = mesh;
              neural_acd::Mesh candidate = mesh;
              neural_acd::ManifoldPreprocessMetrics reference_metrics;
              neural_acd::ManifoldPreprocessMetrics candidate_metrics;

              const auto reference_start =
                  std::chrono::steady_clock::now();
              neural_acd::manifold_preprocess_cpu_reference(
                  reference, scale, level_set, &reference_metrics);
              const double reference_ms =
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - reference_start)
                      .count();

              std::string fallback_reason;
              const auto candidate_start =
                  std::chrono::steady_clock::now();
              const bool supported =
                  neural_acd::manifold_preprocess_cuda_candidate(
                      candidate, scale, level_set, &fallback_reason,
                      &candidate_metrics);
              const double candidate_ms =
                  std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - candidate_start)
                      .count();

              size_t vertex_mismatches = 0;
              size_t triangle_mismatches = 0;
              if (supported) {
                  const size_t common_vertices = std::min(
                      reference.vertices.size(), candidate.vertices.size());
                  for (size_t index = 0; index < common_vertices; ++index) {
                      bool mismatch = false;
                      for (int axis = 0; axis < 3; ++axis) {
                          mismatch = mismatch ||
                              std::memcmp(
                                  &reference.vertices[index][axis],
                                  &candidate.vertices[index][axis],
                                  sizeof(double)) != 0;
                      }
                      vertex_mismatches += mismatch ? 1 : 0;
                  }
                  vertex_mismatches +=
                      reference.vertices.size() > common_vertices
                          ? reference.vertices.size() - common_vertices
                          : candidate.vertices.size() - common_vertices;
                  const size_t common_triangles = std::min(
                      reference.triangles.size(),
                      candidate.triangles.size());
                  for (size_t index = 0; index < common_triangles; ++index) {
                      triangle_mismatches +=
                          reference.triangles[index] !=
                                  candidate.triangles[index]
                              ? 1
                              : 0;
                  }
                  triangle_mismatches +=
                      reference.triangles.size() > common_triangles
                          ? reference.triangles.size() - common_triangles
                          : candidate.triangles.size() - common_triangles;
              }

              py::dict result;
              result["supported"] = supported;
              result["fallback_reason"] = fallback_reason;
              result["exact"] =
                  supported && vertex_mismatches == 0 &&
                  triangle_mismatches == 0;
              result["reference_vertices"] = reference.vertices.size();
              result["candidate_vertices"] = candidate.vertices.size();
              result["reference_triangles"] =
                  reference.triangles.size();
              result["candidate_triangles"] =
                  candidate.triangles.size();
              result["vertex_mismatches"] = vertex_mismatches;
              result["triangle_mismatches"] = triangle_mismatches;
              result["reference_ms"] = reference_ms;
              result["candidate_ms"] = candidate_ms;
              result["reference_sdf_ms"] =
                  reference_metrics.mesh_to_sdf_ns / 1e6;
              result["candidate_sdf_ms"] =
                  candidate_metrics.mesh_to_sdf_ns / 1e6;
              return result;
          },
          py::arg("mesh"), py::arg("scale"), py::arg("level_set"));

    m.attr("config") =
        py::cast(&neural_acd::config, py::return_value_policy::reference);

    py::class_<neural_acd::ProcessResult>(m, "ProcessResult")
        .def_readwrite("parts", &neural_acd::ProcessResult::parts)
        .def_readwrite("concavity", &neural_acd::ProcessResult::concavity)
        .def_readwrite("num_parts", &neural_acd::ProcessResult::num_parts);

    m.def("process", &neural_acd::process, py::arg("mesh"),
          py::arg("concavity"), py::arg("num_parts"),
          py::call_guard<py::gil_scoped_release>());
    m.def("process_batch", &neural_acd::process_batch, py::arg("meshes"),
          py::arg("concavity"), py::arg("num_parts"),
          py::call_guard<py::gil_scoped_release>());
    m.def("release_gpu_resources", &neural_acd::release_gpu_resources,
          py::call_guard<py::gil_scoped_release>());
}
