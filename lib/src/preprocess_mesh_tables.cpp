#include <cstring>
#include <openvdb/tools/VolumeToMesh.h>
#include <preprocess_mesh_tables.hpp>

namespace neural_acd {

void copy_openvdb_volume_mesh_tables(unsigned char *edge_groups,
                                     unsigned char *ambiguous_faces) {
  using namespace openvdb::tools::volume_to_mesh_internal;
  std::memcpy(edge_groups, sEdgeGroupTable, 256 * 13);
  std::memcpy(ambiguous_faces, sAmbiguousFace, 256);
}

} // namespace neural_acd
