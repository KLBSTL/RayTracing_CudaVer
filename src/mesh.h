//
// Created by 86189 on 26-5-29.
//

#ifndef MESH_H
#define MESH_H

#include <utility>
#include <vector>

#include "aabb.h"
#include "material.h"
#include "primitive.h"

class mesh {
public:
    mesh() = default;

    mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices)
        : vertices(std::move(vertices)), indices(std::move(indices)) {
        build_bbox();
    }

    const aabb& bounding_box() const {
        return bbox;
    }

    void append_to_scene(
        std::vector<primitive>& prims,
        std::vector<triangleData>& triangles,
        std::vector<Material>& mats,
        const Material& mat
    ) const {
        if (indices.size() < 3 || vertices.empty()) {
            return;
        }

        const int material_id = static_cast<int>(mats.size());
        mats.push_back(mat);

        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            const unsigned int i0 = indices[i];
            const unsigned int i1 = indices[i + 1];
            const unsigned int i2 = indices[i + 2];

            if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size()) {
                continue;
            }

            const Vertex& v0 = vertices[i0];
            const Vertex& v1 = vertices[i1];
            const Vertex& v2 = vertices[i2];

            const int triangle_id = static_cast<int>(triangles.size());
            triangles.push_back(make_triangle(
                v0.Position,
                v1.Position,
                v2.Position,
                v0.TexCoords,
                v1.TexCoords,
                v2.TexCoords
            ));
            prims.push_back({PRIM_triangle, triangle_id, material_id});
        }
    }

private:
    void build_bbox() {
        bbox = aabb::empty();
        for (const Vertex& vertex : vertices) {
            bbox = aabb(bbox, aabb(vertex.Position, vertex.Position));
        }
    }

    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    aabb bbox = aabb::empty();
};

#endif //MESH_H
