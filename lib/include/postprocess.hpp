#pragma once

#include <core.hpp>
#include <cost.hpp>
#include <iostream>
#include <vector>

namespace neural_acd {
    void multimerge_ch(MeshList &meshs, MeshList &cvxs, double current_concavity, double threshold);
    void separate_disjoint(MeshList &parts);
    void separate_disjoint_prepared(
        MeshList &parts, const std::vector<std::vector<int>> &labels);
}
