//
// Created by 86189 on 26-4-27.
//

#ifndef INTERVAL_H
#define INTERVAL_H
static constexpr float infinity = 1.0e30f;

class interval {
public:
    float min,max;
    // 构造一个空区间。
    __host__ __device__ interval():min(+infinity),max(-infinity) {}

    // 使用上下界构造区间。
    __host__ __device__ interval(float _min, float _max):min(_min),max(_max) {}

    // 构造能同时包含两个输入区间的区间。
    __host__ __device__ interval(const interval& a,const interval& b) {
        min = a.min <= b.min ? a.min : b.min;
        max = a.max >= b.max ? a.max : b.max;
    }

    // 将 x 限制在当前区间范围内。
    __host__ __device__ float clamp(float x) {
        if (x < min) return min;
        if (x > max) return max;
        return x;
    }

    // 返回区间长度。
    __host__ __device__ float size() const {
        return max - min;
    }

    // 判断 x 是否位于闭区间内。
    __host__ __device__ bool contains(float x) const {
        return min <= x && x <= max;
    }

    // 判断 x 是否位于开区间内。
    __host__ __device__ bool surrounds(float x) const {
        return min < x && x < max;
    }

    // 向两端扩展区间，常用于避免零厚度包围盒。
    __host__ __device__ interval expand(float delta)const {
        float padding = delta / 2.0f;
        return interval(min -
            padding, max + padding);
    }

    __host__ __device__ static interval empty() {
        return interval(+infinity, -infinity);
    }

    __host__ __device__ static interval universe() {
        return interval(-infinity, +infinity);
    }
};

// 将区间整体平移指定距离。
__host__ __device__ inline interval operator+(const interval & ival,float displacement) {
    return interval(ival.min+displacement,ival.max+displacement);
}

// 支持标量在左侧的区间平移写法。
__host__ __device__ inline interval operator+(float displacement,const interval & ival) {
    return ival+displacement;
}



#endif //INTERVAL_H
