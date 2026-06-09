//
// Created by 86189 on 26-4-27.
//

#ifndef AABB_H
#define AABB_H
#include "interval.h"
#include "ray.h"
#include "vec3.h"


class aabb {
public:
    interval x = interval::empty(),y=interval::empty(),z=interval::empty();

    // 构造一个空包围盒。
    __host__ __device__ aabb(){}

    // 使用三个轴向区间构造包围盒。
    __host__ __device__ aabb(const interval& x, const interval& y, const interval& z):x(x),y(y),z(z) {
        pad_to_minimums();
    }

    // 使用两个对角点构造包围盒。
    __host__ __device__ aabb(const vec3& a,const vec3& b) {
        x = (a[0]<=b[0]) ? interval(a[0],b[0]) : interval(b[0],a[0]);
        y = (a[1]<=b[1]) ? interval(a[1],b[1]) : interval(b[1],a[1]);
        z = (a[2]<=b[2]) ? interval(a[2],b[2]) : interval(b[2],a[2]);

        pad_to_minimums();
    }

    // 构造同时包含两个包围盒的新包围盒。
    __host__ __device__ aabb(const aabb& a,const aabb& b) {
        x = interval(a.x, b.x);
        y = interval(a.y, b.y);
        z = interval(a.z, b.z);
    }

    // 返回指定轴的区间，0/1/2 分别表示 x/y/z。
    __host__ __device__ const interval& axis_interval(int n)const {
        if (n == 1)return y;
        if (n == 2)return z;
        return x;
    }

    // 使用 slab 方法判断射线是否穿过包围盒。
    __host__ __device__ bool hit_bbox(const ray& r,interval ray_t)const {
        const vec3& ray_orig = r.origin();
        const vec3& ray_dir = r.direction();

        for (int axis = 0;axis < 3;axis++) {
            const interval& ax = axis_interval(axis);
            const float adinv = 1.0f / ray_dir[axis];

            float t0 = (ax.min - ray_orig[axis]) * adinv;
            float t1 = (ax.max - ray_orig[axis]) * adinv;

            if (t0 < t1) {
                if (t0 > ray_t.min) ray_t.min = t0;
                if (t1 < ray_t.max) ray_t.max = t1;
            }
            else {
                if (t0 < ray_t.max) ray_t.max = t0;
                if (t1 > ray_t.min) ray_t.min = t1;
            }

            if (ray_t.max <= ray_t.min) return false;

        }
        return true;
    }

    // 返回包围盒跨度最大的轴，用于 BVH 分割。
    __host__ __device__ int longest_axis() const {
        if (x.size() > y.size()) {
            return x.size() > z.size() ? 0 : 2;
        }
        else {
            return y.size() > z.size() ? 1 : 2;
        }
    }


    __host__ __device__ static aabb empty() {
        return aabb(interval::empty(), interval::empty(),interval::empty());
    }

    __host__ __device__ static aabb universe() {
        return aabb(interval::universe(), interval::universe(),interval::universe());
    }

    private:
    // 给过薄的包围盒补最小厚度，避免浮点误差导致漏交。
    __host__ __device__ void pad_to_minimums() {
        float delta = 0.0001f;
        if (x.size() < delta) x = x.expand(delta);
        if (y.size() < delta) y = y.expand(delta);
        if (z.size() < delta) z = z.expand(delta);
    }
};

// 将包围盒整体平移指定偏移量。
__host__ __device__ inline aabb operator+(const aabb& bbox, const vec3& offset) {
    return aabb(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
}

// 支持偏移量在左侧的包围盒平移写法。
__host__ __device__ inline aabb operator+(const vec3& offset, const aabb& bbox) {
    return bbox + offset;
}



#endif //AABB_H
