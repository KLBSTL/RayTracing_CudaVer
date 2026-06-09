//
// Created by 86189 on 26-6-8.
//

#ifndef SCENE_H
#define SCENE_H


#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector>

#include "camera.h"
#include "hitable.h"
#include "material.h"
#include "ray.h"
#include "vec3.h"
#include "primitive.h"
#include "bvh.h"
#include "model.h"
#include "stb_image_write.h"

#ifndef RT_ENABLE_ASSIMP_MODEL
#define RT_ENABLE_ASSIMP_MODEL 0
#endif


#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ bool hit_sphere(const sphereData &sphere,const ray& r, float t_min, float t_max, hit_record& rec,int id) {
    vec3 oc = r.origin() - sphere.center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - sphere.radius*sphere.radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - sphere.center) / sphere.radius;
            rec.material_id = id;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.at(rec.t);
            rec.normal = (rec.p - sphere.center) / sphere.radius;
            rec.material_id = id;
            return true;
        }
    }
    return false;
}

__device__ bool contains(float tmin,float tmax,float t) {
    return tmin <= t && t <= tmax;
}

__device__ bool hit_quad(const quadData &quad,const ray& r, float t_min, float t_max, hit_record& rec,int id) {

    auto denom = dot(quad.normal, r.direction());

    // No hit if the ray is parallel to the plane.
    if (std::fabs(denom) < 1e-8)
        return false;

    // Return false if the hit point parameter t is outside the ray interval.
    auto t = (quad.D - dot(quad.normal, r.origin())) / denom;
    if (!contains(t_min,t_max,t))
        return false;

    // Determine if the hit point lies within the planar shape using its plane coordinates.
    auto intersection = r.at(t);
    vec3 planar_hitpt_vector = intersection - quad.Q;
    auto alpha = dot(quad.w, cross(planar_hitpt_vector, quad.v));
    auto beta = dot(quad.w, cross(quad.u, planar_hitpt_vector));

    if (!contains(0,1,alpha) || !contains(0,1,beta)) {
        return false;
    }

    rec.u = alpha;
    rec.v = beta;

    // Ray hits the 2D shape; set the rest of the hit record and return true.
    rec.t = t;
    rec.p = intersection;
    rec.material_id = id;
    rec.set_face_normal(r, quad.normal);

    return true;
}

__device__ bool hit_triangle(const triangleData &triangle,const ray &r, float t_min, float t_max,hit_record& rec,int id) {
    vec3 s = r.origin() - triangle.a;
    vec3 s1 = cross(r.direction(), triangle.e2);
    vec3 s2 = cross(s, triangle.e1);

    float det = dot(s1,triangle.e1);
    if (std::fabs(det) < 1e-8){return false;}

    float s1e1 = 1.0f / det;

    float tnear = s1e1 * dot(triangle.e2,s2);

    float u11 = s1e1 * dot(s1,s);
    float v11 = s1e1 * dot(s2,r.direction());


    if (!contains(t_min,t_max,tnear))
        return false;


    if (u11 < 0 || v11 < 0 || u11 + v11 > 1){return false;}

    auto w = 1.0 - u11 - v11;

    rec.t = tnear;
    rec.u = w * triangle.uv0.x + u11 * triangle.uv1.x + v11 * triangle.uv2.x;
    rec.v = w * triangle.uv0.y + u11 * triangle.uv1.y + v11 * triangle.uv2.y;
    rec.p = r.at(tnear);
    rec.material_id = id;
    rec.set_face_normal(r, triangle.normal);

    return true;
}



__device__ bool hit_primitive(
    const primitive* prim,
    const int prim_counts,
    const sphereData* spheres,
    const triangleData* triangles,
    const quadData* quads,
    const ray& r,
    float t_min,
    float t_max,
    hit_record& rec
) {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < prim_counts; i++) {
        switch (prim[i].type) {
            case PRIM_sphere:
                if (hit_sphere(spheres[prim[i].data_index], r, t_min, closest_so_far, temp_rec, prim[i].material_id)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
                break;
            case PRIM_triangle:
                if (hit_triangle(triangles[prim[i].data_index], r, t_min, closest_so_far, temp_rec, prim[i].material_id)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
                break;
            case PRIM_quad:
                if (hit_quad(quads[prim[i].data_index], r, t_min, closest_so_far, temp_rec, prim[i].material_id)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
                break;
        }
    }

    return hit_anything;
}

__device__ bool scatter_lambertian(const Material &mat,const ray& r_in,
    const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state)
{
    vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
    scattered = ray(rec.p, target-rec.p);
    attenuation = mat.albedo;
    return true;
}

__device__ bool scatter_metal(const Material &mat,const ray& r_in,
    const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state)
{
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + mat.fuzz*random_in_unit_sphere(local_rand_state));
    attenuation = mat.albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
}

__device__ bool scatter_dielectirc(const Material &mat,const ray& r_in,
    const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state)
{
    vec3 outward_normal;
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = vec3(1.0, 1.0, 1.0);
    vec3 refracted;
    float reflect_prob;
    float cosine;
    if (dot(r_in.direction(), rec.normal) > 0.0f) {
        outward_normal = -rec.normal;
        ni_over_nt = mat.ref_idx;
        cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
        cosine = sqrt(1.0f - mat.ref_idx*mat.ref_idx*(1-cosine*cosine));
    }
    else {
        outward_normal = rec.normal;
        ni_over_nt = 1.0f / mat.ref_idx;
        cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
        reflect_prob = schlick(cosine, mat.ref_idx);
    else
        reflect_prob = 1.0f;
    if (curand_uniform(local_rand_state) < reflect_prob)
        scattered = ray(rec.p, reflected);
    else
        scattered = ray(rec.p, refracted);
    return true;
}

__device__ bool scatter_material(const Material &mat,const ray& r_in, const hit_record& rec,
    vec3& attenuation, ray& scattered, curandState *local_rand_state)
{

    switch (mat.type) {
        case Lambertian:
            return scatter_lambertian(mat, r_in, rec, attenuation, scattered, local_rand_state);
        case Dielectric:
            return scatter_dielectirc(mat, r_in, rec, attenuation, scattered, local_rand_state);
        case Metal:
            return scatter_metal(mat, r_in, rec, attenuation, scattered, local_rand_state);
    }

    return false;
}

__device__ bool hit_one_primitive(
    const primitive& p,
    const sphereData* spheres,
    const triangleData* triangles,
    const quadData* quads,
    const ray& r,
    float t_min,
    float t_max,
    hit_record& rec
) {
    switch (p.type) {
        case PRIM_sphere:
            return hit_sphere(spheres[p.data_index], r, t_min, t_max, rec, p.material_id);
        case PRIM_triangle:
            return hit_triangle(triangles[p.data_index], r, t_min, t_max, rec, p.material_id);
        case PRIM_quad:
            return hit_quad(quads[p.data_index], r, t_min, t_max, rec, p.material_id);
    }
    return false;
}

__device__ bool hit_bvh(
    const BVHNode* nodes,
    int root_index,
    const primitive* prims,
    const sphereData* spheres,
    const triangleData* triangles,
    const quadData* quads,
    const ray& r,
    float t_min,
    float t_max,
    hit_record& rec
) {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    int stack[64];
    int stack_size = 0;
    stack[stack_size++] = root_index;

    while (stack_size > 0) {
        int node_index = stack[--stack_size];
        BVHNode node = nodes[node_index];

        if (!node.bbox.hit_bbox(r, interval(t_min, closest_so_far)))
            continue;

        if (node.is_leaf) {
            const primitive& p = prims[node.prim_index];

            if (hit_one_primitive(
                p, spheres, triangles, quads,
                r, t_min, closest_so_far, temp_rec
            )) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        } else {
            if (stack_size + 2 <= 64) {
                stack[stack_size++] = node.left;
                stack[stack_size++] = node.right;
            }
        }
    }

    return hit_anything;
}

__device__ vec3 color(
    const ray& r,
    const primitive* prims,
    int prim_count,
    const sphereData* spheres,
    const triangleData* triangles,
    const quadData* quads,
    const Material* mats,
    const BVHNode* bvh_nodes,
    int bvh_root,
    curandState* rng
) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if (hit_bvh(bvh_nodes,bvh_root,prims,spheres,triangles,quads,
            cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if(scatter_material(mats[rec.material_id],
                cur_ray, rec, attenuation, scattered, rng)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0,0.0,0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f*(unit_direction.y() + 1.0f);
            vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0,0.0,0.0);
}

__global__ void rand_init(curandState *rand_state,camera **d_camera,int nx,int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0;
        float aperture = 0.1;
        *d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 30.0, float(nx)/float(ny), aperture, dist_to_focus);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam,
    const primitive* prims,
    int prim_count,
    const sphereData* spheres,
    const triangleData* triangles,
    const quadData* quads,
    const Material* mats,
    const BVHNode* bvh_nodes,
    int bvh_root,
    curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0,0,0);
    for(int s=0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r,prims,prim_count,spheres,triangles,quads,mats,bvh_nodes,bvh_root, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    fb[pixel_index] = col;
}


inline Material make_lambertian_material(const vec3& albedo) {
    Material mat{};
    mat.type = Lambertian;
    mat.albedo = albedo;
    mat.ref_idx = 1.0f;
    return mat;
}

inline Material make_metal_material(const vec3& albedo, float fuzz) {
    Material mat{};
    mat.type = Metal;
    mat.albedo = albedo;
    mat.fuzz = fuzz;
    mat.ref_idx = 1.0f;
    return mat;
}

inline Material make_dielectric_material(float ref_idx) {
    Material mat{};
    mat.type = Dielectric;
    mat.ref_idx = ref_idx;
    return mat;
}

inline void add_sphere_primitive(
    std::vector<primitive>& prims,
    std::vector<sphereData>& spheres,
    std::vector<Material>& mats,
    const vec3& center,
    float radius,
    const Material& mat
) {
    const int material_id = static_cast<int>(mats.size());
    mats.push_back(mat);

    const int sphere_id = static_cast<int>(spheres.size());
    spheres.push_back(make_sphere(center, radius));
    prims.push_back({PRIM_sphere, sphere_id, material_id});
}

inline void add_quad_primitive(
    std::vector<primitive>& prims,
    std::vector<quadData>& quads,
    std::vector<Material>& mats,
    const vec3& Q,
    const vec3& u,
    const vec3& v,
    const Material& mat
) {
    const int material_id = static_cast<int>(mats.size());
    mats.push_back(mat);

    const int quad_id = static_cast<int>(quads.size());
    quads.push_back(make_quad(Q, u, v));
    prims.push_back({PRIM_quad, quad_id, material_id});
}

inline void add_triangle_primitive(
    std::vector<primitive>& prims,
    std::vector<triangleData>& triangles,
    std::vector<Material>& mats,
    const vec3& a,
    const vec3& b,
    const vec3& c,
    const Material& mat
) {
    const int material_id = static_cast<int>(mats.size());
    mats.push_back(mat);

    const int triangle_id = static_cast<int>(triangles.size());
    triangles.push_back(make_triangle(a, b, c));
    prims.push_back({PRIM_triangle, triangle_id, material_id});
}

void create_world(
    std::vector<primitive> &prim,
    std::vector<sphereData> &spheres,
    std::vector<quadData> &quads,
    std::vector<triangleData> &triangles,
    std::vector<Material> &mats,
    // camera **d_camera,
    int nx, int ny)
{
        prim.clear();
        spheres.clear();
        quads.clear();
        triangles.clear();
        mats.clear();

        prim.reserve(22 * 22 + 3);
        spheres.reserve(22 * 22 + 1);
        quads.reserve(1);
        triangles.reserve(1);
        mats.reserve(22 * 22 + 3);

        add_sphere_primitive(
            prim,
            spheres,
            mats,
            vec3(0, -1000.0f, -1),
            1000.0f,
            make_lambertian_material(vec3(0.5f, 0.5f, 0.5f))
        );

        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = random();
                vec3 center(a+random(),0.2f + random() * 0.4f,b+random());
                const float radius = 0.2f + random() * 0.1f;
                if(choose_mat < 0.8f) {
                    add_sphere_primitive(
                        prim,
                        spheres,
                        mats,
                        center,
                        radius,
                        make_lambertian_material(vec3(random()*random(), random()*random(), random()*random()))
                    );
                }
                else if(choose_mat < 0.95f) {
                    add_sphere_primitive(
                        prim,
                        spheres,
                        mats,
                        center,
                        radius,
                        make_metal_material(
                            vec3(0.5f*(1.0f+random()), 0.5f*(1.0f+random()), 0.5f*(1.0f+random())),
                            0.5f*(1.0f+random())
                        )
                    );
                }
                else {
                    add_sphere_primitive(
                        prim,
                        spheres,
                        mats,
                        center,
                        radius,
                        make_dielectric_material(1.4f + random() * 0.2f)
                    );
                }
            }
        }

        // add_quad_primitive(
        //     prim,
        //     quads,
        //     mats,
        //     vec3(-1.7f, 0.8f, -1.2f),
        //     vec3(3.4f, 0.0f, 0.0f),
        //     vec3(0.0f, 2.1f, 0.0f),
        //     make_lambertian_material(vec3(0.95f, 0.18f, 0.12f))
        // );
        //
        // add_triangle_primitive(
        //     prim,
        //     triangles,
        //     mats,
        //     vec3(2.0f, 0.8f, 0.6f),
        //     vec3(4.2f, 0.8f, -0.2f),
        //     vec3(3.0f, 3.0f, 0.2f),
        //     make_lambertian_material(vec3(0.10f, 0.45f, 0.95f))
        // );

#if RT_ENABLE_ASSIMP_MODEL
        const std::string model_path = "D:/ClionFlies/rayTracingRenderer/resources/backpack/backpack.obj";
        Model backpack(model_path);
        if (backpack.loaded()) {
            const size_t old_prim_count = prim.size();
            backpack.append_to_scene(
                prim,
                triangles,
                mats,
                make_lambertian_material(vec3(0.72f, 0.72f, 0.72f))
            );
            std::cerr << "Loaded model primitives: " << prim.size() - old_prim_count << "\n";
        }
#endif


        // vec3 lookfrom(13,2,3);
        // vec3 lookat(0,0,0);
        // float dist_to_focus = 10.0;
        // float aperture = 0.1;
        // *d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 30.0, float(nx)/float(ny), aperture, dist_to_focus);

}

__global__ void clear_world(camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        delete *d_camera;
        *d_camera = nullptr;
    }
}


class scene {
    public:
    scene() = default;

    void set_KernelSize() {
        blocks = dim3((nx + tx - 1) / tx, (ny + ty - 1) / ty);
        threads = dim3(tx,ty);
    }

    void init() {
        num_pixels = nx*ny;
        fb_size = num_pixels*sizeof(vec3);

        image.assign(nx * ny * 3,0);

        set_KernelSize();

        std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
        std::cerr << "in " << tx << "x" << ty << " blocks.\n";

        checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

        checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
        checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

        checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

        rand_init<<<1,1>>>(d_rand_state2,d_camera,nx,ny);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());


        create_world(h_prims,h_spheres,h_quads,h_triangles,h_mats,nx,ny);
        counts = static_cast<int>(h_prims.size());
        const int sphere_count = static_cast<int>(h_spheres.size());
        const int quad_count = static_cast<int>(h_quads.size());
        const int triangle_count = static_cast<int>(h_triangles.size());
        std::cerr << "Scene primitives: " << counts
                  << " (spheres " << sphere_count
                  << ", quads " << quad_count
                  << ", triangles " << triangle_count << ")\n";

        h_bvh.build(h_prims, h_spheres.data(), h_triangles.data(), h_quads.data());


        bvh_root = h_bvh.root_index;
        bvh_node_count = static_cast<int>(h_bvh.nodes.size());

        checkCudaErrors(cudaMalloc(&bvh_nodes, bvh_node_count * sizeof(BVHNode)));
        checkCudaErrors(cudaMemcpy(bvh_nodes, h_bvh.nodes.data(),
                   bvh_node_count * sizeof(BVHNode),
                   cudaMemcpyHostToDevice));


        checkCudaErrors(cudaMalloc(&prims, counts*sizeof(primitive)));
        checkCudaErrors(cudaMalloc(&spheres, sphere_count * sizeof(sphereData)));
        checkCudaErrors(cudaMalloc(&quads, quad_count * sizeof(quadData)));
        checkCudaErrors(cudaMalloc(&triangles, triangle_count * sizeof(triangleData)));
        checkCudaErrors(cudaMalloc(&mats, counts * sizeof(Material)));


        checkCudaErrors(cudaMemcpy(prims,h_prims.data(),sizeof(primitive) * h_prims.size(),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(spheres,h_spheres.data(),sizeof(sphereData) * h_spheres.size(),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(quads,h_quads.data(),sizeof(quadData) * h_quads.size(),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(triangles,h_triangles.data(),sizeof(triangleData) * h_triangles.size(),cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(mats,h_mats.data(),sizeof(Material) * h_mats.size(),cudaMemcpyHostToDevice));



        // create_world<<<1,1>>>(prims,spheres,quads,triangles,mats, d_camera, nx, ny, d_rand_state2);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());


    }

    void output() {
        double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
        std::cerr << "took " << timer_seconds << " seconds.\n";

        for (int row = 0; row < ny; row++) {
            int j = ny - 1 - row;
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j*nx + i;
                write_color_buffer(image, i, row, nx, fb[pixel_index], ns);
            }
        }

        const std::string filename = output_path + image_name + ".png";
        if (!stbi_write_png(filename.c_str(), nx, ny, 3, image.data(), nx * 3)) {
            std::cerr << "ERROR: Could not write image file '" << filename << "'.\n";
            return;
        }

        std::cerr << "Wrote " << filename << "\n";
    }


    void write_color_buffer(std::vector<unsigned char> &image, int x, int y,
        int image_width, const vec3 &pixel_color, int samples_per_pixel)
    {
        float r = pixel_color.x();
        float g = pixel_color.y();
        float b = pixel_color.z();

        float scale = 1.0f / samples_per_pixel;

        r = std::sqrt(scale * r);
        g = std::sqrt(scale * g);
        b = std::sqrt(scale * b);

        if (r != r){r = 0.0;}
        if (g != g){g = 0.0;}
        if (b != b){b = 0.0;}

        int index = 3 * (y * image_width + x);

        image[index + 0] = static_cast<unsigned char>(256.0f * std::clamp(r, 0.0f, 0.999f));
        image[index + 1] = static_cast<unsigned char>(256.0f * std::clamp(g, 0.0f, 0.999f));
        image[index + 2] = static_cast<unsigned char>(256.0f * std::clamp(b, 0.0f, 0.999f));
    }


    void run() {
        init();


        start = clock();


        render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, prims,counts,spheres,triangles,
            quads,mats,bvh_nodes,bvh_root,d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());


        stop = clock();


        output();

        clear();
    }


    void clear() {
        // clean up
        checkCudaErrors(cudaDeviceSynchronize());
        clear_world<<<1,1>>>(d_camera);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaFree(prims));
        checkCudaErrors(cudaFree(spheres));
        checkCudaErrors(cudaFree(quads));
        checkCudaErrors(cudaFree(triangles));
        checkCudaErrors(cudaFree(mats));
        checkCudaErrors(cudaFree(bvh_nodes));
        checkCudaErrors(cudaFree(d_camera));
        checkCudaErrors(cudaFree(d_rand_state));
        checkCudaErrors(cudaFree(d_rand_state2));
        checkCudaErrors(cudaFree(fb));

        cudaDeviceReset();
    }

private:
    int nx = 1200;
    int ny = 800;
    int ns = 100;
    int tx = 16;
    int ty = 16;

    primitive *prims = nullptr;
    Material *mats = nullptr;
    sphereData * spheres = nullptr;
    triangleData * triangles = nullptr;
    quadData * quads = nullptr;
    camera **d_camera = nullptr;

    std::vector<primitive> h_prims;
    std::vector<sphereData> h_spheres;
    std::vector<quadData> h_quads;
    std::vector<triangleData> h_triangles;
    std::vector<Material> h_mats;
    bvh h_bvh;

    int counts;


    vec3 *fb = nullptr;
    int num_pixels;
    size_t fb_size;

    clock_t start, stop;

    dim3 blocks;
    dim3 threads;

    curandState *d_rand_state = nullptr;
    curandState *d_rand_state2 = nullptr;


    BVHNode* bvh_nodes = nullptr;
    int bvh_root = -1;
    int bvh_node_count = 0;


    std::vector<unsigned char> image;
    std::string image_name = "cuda_render";
    std::string output_path = "../image/";

};
#endif //SCENE_H
