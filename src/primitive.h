//
// Created by 86189 on 26-6-8.
//

#ifndef PRIMITIVE_H
#define PRIMITIVE_H
#include "vec3.h"

enum PrimitiveType {
    PRIM_sphere,
    PRIM_quad,
    PRIM_triangle
};

struct sphereData {
    float radius;
    vec3 center;
};

struct quadData {
    vec3 Q,u,v;
    vec3 normal;
    vec3 w;
    float D;
};

struct triangleData {
    vec3 a,b,c;
    vec3 e1,e2;
    vec3 normal;
    vec2 uv0,uv1,uv2;
    float area;

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

    return triangle;
}

#endif //PRIMITIVE_H
