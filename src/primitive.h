//
// Created by 86189 on 26-6-8.
//

#ifndef PRIMITIVE_H
#define PRIMITIVE_H
#include "aabb.h"
#include "vec3.h"

enum PrimitiveType {
    PRIM_sphere,
    PRIM_quad,
    PRIM_triangle
};

struct sphereData {
    float radius;
    vec3 center;
    aabb bbox;
};

struct quadData {
    vec3 Q,u,v;
    vec3 normal;
    vec3 w;
    float D;
    aabb bbox;
};

struct triangleData {
    vec3 a,b,c;
    vec3 e1,e2;
    vec3 normal;
    vec2 uv0,uv1,uv2;
    float area;
    aabb bbox;

    __host__ __device__ triangleData() {
        uv0 = vec2(0,0);
        uv1 = vec2(1,0);
        uv2 = vec2(0,1);
    }
};

struct primitive {
    PrimitiveType type;
    int data_index;
    int material_id;
};

__host__ __device__ inline vec3 safe_unit_vector(const vec3& v) {
    float len = v.length();
    return len > 0.0f ? v / len : vec3(0.0f, 1.0f, 0.0f);
}

__host__ __device__ inline sphereData make_sphere(const vec3& cen, const float radius) {
    sphereData sphere;
    sphere.center = cen;
    sphere.radius = radius;

    auto rvec = vec3(radius,radius,radius);
    sphere.bbox = aabb(cen - rvec,cen + rvec);

    return sphere;
}

__host__ __device__ inline quadData make_quad(const vec3& Q, const vec3& u, const vec3& v) {
    quadData quad;
    quad.Q = Q;
    quad.u = u;
    quad.v = v;

    vec3 n = cross(u, v);
    float n_len2 = dot(n, n);
    quad.normal = safe_unit_vector(n);
    quad.w = n_len2 > 0.0f ? n / n_len2 : vec3(0.0f, 0.0f, 0.0f);
    quad.D = dot(quad.normal, Q);

    auto b1 = aabb(quad.Q, quad.Q + quad.u + quad.v);
    auto b2 = aabb(quad.Q + quad.u ,quad.Q + quad.v);

    quad.bbox = aabb(b1, b2);

    return quad;
}

__host__ __device__ inline triangleData make_triangle(
    const vec3& a,
    const vec3& b,
    const vec3& c,
    const vec2& uv0 = vec2(0.0f, 0.0f),
    const vec2& uv1 = vec2(1.0f, 0.0f),
    const vec2& uv2 = vec2(0.0f, 1.0f)
) {
    triangleData triangle;
    triangle.a = a;
    triangle.b = b;
    triangle.c = c;
    triangle.e1 = b - a;
    triangle.e2 = c - a;
    vec3 n = cross(triangle.e1, triangle.e2);
    triangle.area = 0.5f * n.length();
    triangle.normal = safe_unit_vector(n);
    triangle.uv0 = uv0;
    triangle.uv1 = uv1;
    triangle.uv2 = uv2;

    auto b1 = aabb(triangle.a, triangle.b);
    auto b2 = aabb(triangle.c, triangle.c);

    triangle.bbox = aabb(b1, b2);

    return triangle;
}

__host__ __device__ inline aabb primitive_bbox(
    const primitive& p,
    const sphereData* spheres,
    const triangleData* triangles,
    const quadData* quads
) {
    switch (p.type) {
        case PRIM_sphere:
            return spheres[p.data_index].bbox;
        case PRIM_triangle:
            return triangles[p.data_index].bbox;
        case PRIM_quad:
            return quads[p.data_index].bbox;
    }

    return aabb::empty();
}


#endif //PRIMITIVE_H
