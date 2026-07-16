#include <cmath>
#include <core.hpp>
#include <cost.hpp>
#include <process.hpp>
#include <random>
#include <utility>
#include <vector>

using namespace std;

namespace neural_acd {


  vector<Plane>
  get_candidate_planes(vector<Vec3D> &vertices,
                       vector<pair<unsigned int, unsigned int>> &edges,
                       int num_planes)
  {
      vector<Plane> planes;

      if (edges.empty())
          return planes;

      uniform_int_distribution<> dis(0, edges.size() - 1);
      const double normal_eps = cos(5.0 * M_PI / 180.0); // ~5° angular tolerance
      const double dist_eps = 1e-3;                      // distance tolerance

      for (int i = 0; i < num_planes * 5 && (int)planes.size() < num_planes; ++i) {
          int idx = dis(random_engine);
          Vec3D p1 = vertices[edges[idx].first];
          Vec3D p2 = vertices[edges[idx].second];
          Vec3D n = p2 - p1; // normal vector

          if (vector_length(n) < 1e-6)
              continue;

          n = normalize_vector(n);
          Vec3D m = (p1 + p2) * 0.5; // midpoint
          double d = -(n[0] * m[0] + n[1] * m[1] + n[2] * m[2]);

          Plane candidate(n[0], n[1], n[2], d);

          // check if similar to an existing plane
          bool too_similar = false;
          for (const auto &p : planes) {
              double dot = fabs(n[0]*p.a + n[1]*p.b + n[2]*p.c); // |n·p.n|
              if (dot > normal_eps && fabs(d - p.d) < dist_eps) {
                  too_similar = true;
                  break;
              }
          }

          if (!too_similar)
              planes.push_back(candidate);
      }

      return planes;
  }


double compute_part_score(Mesh &part) {
  double score = 0.0;
  for (auto &e : part.intersecting_edges) {
    Vec3D v1 = part.vertices[e.first];
    Vec3D v2 = part.vertices[e.second];

    double len = vector_length(v2 - v1);
    score += len;
  }
  return score;
}

int get_part_with_highest_score(MeshList &parts) {
  double max_score = -1.0;
  int best_idx = -1;
  for (int i = 0; i < parts.size(); i++) {
    double score = compute_part_score(parts[i]);
    if (score > max_score) {
      max_score = score;
      best_idx = i;
    }
  }
  return best_idx;
}

int get_part_with_highest_concavity(MeshList &parts, double &max_concavity) {

  MeshList cvxs;
  for (auto &part : parts) {
    Mesh cvx;
    part.compute_ch(cvx, true);
    cvxs.push_back(cvx);
  }

  int best_idx = -1;
  for (int i = 0; i < parts.size(); i++) {
    double concavity = compute_h(parts[i], cvxs[i], 0.3, 3000, 42);
    if (concavity > max_concavity) {
      max_concavity = concavity;
      best_idx = i;
    }
  }
  return best_idx;
}

double compute_final_concavity(MeshList &parts, MeshList &cvxs) {
  double h = 0;
  for (int i = 0; i < parts.size(); i++) {
    double cur_h = compute_hb(parts[i], cvxs[i], 10000, 42);
    if (cur_h > h)
      h = cur_h;
  }
  return h;
}

ProcessResult process(Mesh mesh, double concavity, int num_parts) {
  MeshList meshes;
  meshes.push_back(move(mesh));
  vector<ProcessResult> results =
      process_batch(move(meshes), concavity, num_parts);
  return move(results.front());
}

} // namespace neural_acd
