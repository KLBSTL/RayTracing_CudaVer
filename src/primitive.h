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
    float D;
};

struct triangleData {
    vec3 a,b,c;
    vec3 normal;
};

struct primitive {
    PrimitiveType type;
    int data_index;
    int material_id;
};

#endif //PRIMITIVE_H
