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
#include <algorithm>
#include <cmath>

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
        const Material& mat,
        vec3 center,
        float scale,
        float theta,
        const vec3& target
    ) const {
        if (indices.size() < 3 || vertices.empty()) {
            return;
        }

        const int material_id = static_cast<int>(mats.size());
        mats.push_back(mat);

        float cos_t = std::cos(theta);
        float sin_t = std::sin(theta);

        for (size_t i = 0; i + 2 < indices.size(); i += 3) {
            const unsigned int i0 = indices[i];
            const unsigned int i1 = indices[i + 1];
            const unsigned int i2 = indices[i + 2];

            if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size()) {
                continue;
            }

            auto p0_world = trans_pos(vertices[i0],center,cos_t,sin_t,target,scale);
            auto p1_world = trans_pos(vertices[i1],center,cos_t,sin_t,target,scale);
            auto p2_world = trans_pos(vertices[i2],center,cos_t,sin_t,target,scale);

            const int triangle_id = static_cast<int>(triangles.size());
            triangles.push_back(make_triangle(
                p0_world,
                p1_world,
                p2_world,
                vertices[i0].TexCoords,
                vertices[i1].TexCoords,
                vertices[i2].TexCoords
            ));
            prims.push_back({PRIM_triangle, triangle_id, material_id});
        }
    }

    vec3 trans_pos(const Vertex& v,vec3 center,float cos_t,float sin_t ,vec3 target,float scale) const {
        vec3 q = scale * (v.Position - center);

        vec3 rotated(
            cos_t * q.x() + sin_t * q.z(),
            q.y(),
            -sin_t * q.x() + cos_t * q.z()
        );

        return rotated + target;
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
