//
// Created by 86189 on 26-6-9.
//

#ifndef BVH_H
#define BVH_H

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

#include "primitive.h"

struct BVHNode {
    aabb bbox;
    int left = -1;
    int right = -1;
    int prim_index = -1;
    int is_leaf = 0;
};

class bvh {
public:
    std::vector<int> prim_indices;
    std::vector<BVHNode> nodes;
    int root_index = -1;

    void build(
        const std::vector<primitive>& prims,
        const sphereData* spheres,
        const triangleData* triangles,
        const quadData* quads
    ) {
        prims_ = &prims;
        spheres_ = spheres;
        triangles_ = triangles;
        quads_ = quads;

        nodes.clear();
        prim_indices.resize(prims.size());
        std::iota(prim_indices.begin(), prim_indices.end(), 0);

        if (prim_indices.empty()) {
            root_index = -1;
            return;
        }

        root_index = build_range(0, prim_indices.size());
    }

private:
    const std::vector<primitive>* prims_ = nullptr;
    const sphereData* spheres_ = nullptr;
    const triangleData* triangles_ = nullptr;
    const quadData* quads_ = nullptr;

    int build_range(std::size_t start, std::size_t end) {
        const int node_index = static_cast<int>(nodes.size());
        nodes.emplace_back();

        BVHNode node;
        node.bbox = range_bbox(start, end);

        const std::size_t object_span = end - start;
        if (object_span == 1) {
            node.is_leaf = 1;
            node.prim_index = prim_indices[start];
            nodes[node_index] = node;
            return node_index;
        }

        const int axis = node.bbox.longest_axis();
        std::sort(
            prim_indices.begin() + start,
            prim_indices.begin() + end,
            [this, axis](int a, int b) {
                return box_compare(a, b, axis);
            }
        );

        const std::size_t mid = start + object_span / 2;
        node.left = build_range(start, mid);
        node.right = build_range(mid, end);
        node.is_leaf = 0;
        node.prim_index = -1;
        node.bbox = aabb(nodes[node.left].bbox, nodes[node.right].bbox);

        nodes[node_index] = node;
        return node_index;
    }

    aabb range_bbox(std::size_t start, std::size_t end) const {
        aabb bbox = aabb::empty();

        for (std::size_t i = start; i < end; i++) {
            const int prim_index = prim_indices[i];
            bbox = aabb(bbox, get_bbox(prim_index));
        }

        return bbox;
    }

    aabb get_bbox(int prim_index) const {
        return primitive_bbox((*prims_)[prim_index], spheres_, triangles_, quads_);
    }

    bool box_compare(int a, int b, int axis) const {
        const aabb box_a = get_bbox(a);
        const aabb box_b = get_bbox(b);
        const interval& axis_a = box_a.axis_interval(axis);
        const interval& axis_b = box_b.axis_interval(axis);

        const float center_a = 0.5f * (axis_a.min + axis_a.max);
        const float center_b = 0.5f * (axis_b.min + axis_b.max);
        if (center_a == center_b) {
            return axis_a.min < axis_b.min;
        }

        return center_a < center_b;
    }
};

#endif //BVH_H
