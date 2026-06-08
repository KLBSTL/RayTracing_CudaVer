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
#include "stb_image_write.h"


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
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - sphere.center) / sphere.radius;
            rec.material_id = id;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - sphere.center) / sphere.radius;
            rec.material_id = id;
            return true;
        }
    }
    return false;
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
                break;
            case PRIM_quad:
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

__device__ vec3 color(
    const ray& r,
    const primitive* prims,
    int prim_count,
    const sphereData* spheres,
    const triangleData* triangles,
    const quadData* quads,
    const Material* mats,
    curandState* rng
) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0,1.0,1.0);
    for(int i = 0; i < 50; i++) {
        hit_record rec;
        if (hit_primitive(prims,prim_count,spheres,triangles,quads,
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

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
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
        col += color(r,prims,prim_count,spheres,triangles,quads,mats, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(primitive *prim,sphereData *spheres,Material *mats, camera **d_camera, int nx, int ny, curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandState local_rand_state = *rand_state;

        prim[0] = {PRIM_sphere, 0, 0};
        spheres[0] = {1000,vec3(0,-1000.0,-1)};

        mats[0] = {Lambertian,vec3(0.5, 0.5, 0.5)};
        mats[1] = {Lambertian,vec3(RND*RND, RND*RND, RND*RND)};
        mats[2] = {Metal,vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)),0.5f*(1.0f+RND)};
        mats[3] = {Dielectric,vec3(0,0,0),0,1.5};

        int p_id = 1;
        int s_id = 1;
        for(int a = -11; a < 11; a++) {
            for(int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a+RND,0.2,b+RND);
                if(choose_mat < 0.8f) {
                    prim[p_id] = {PRIM_sphere, s_id, 1};
                    spheres[s_id] = {0.2,center};
                    p_id++;
                    s_id++;
                }
                else if(choose_mat < 0.95f) {
                    prim[p_id] = {PRIM_sphere, s_id, 2};
                    spheres[s_id] = {0.2,center};
                    p_id++;
                    s_id++;
                }
                else {
                    prim[p_id] = {PRIM_sphere, s_id, 3};
                    spheres[s_id] = {0.2,center};
                    p_id++;
                    s_id++;
                }
            }
        }

        *rand_state = local_rand_state;

        vec3 lookfrom(13,2,3);
        vec3 lookat(0,0,0);
        float dist_to_focus = 10.0;
        float aperture = 0.1;
        *d_camera = new camera(lookfrom, lookat, vec3(0,1,0), 30.0, float(nx)/float(ny), aperture, dist_to_focus);
    }
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
        counts = 22*22 + 1;
        num_pixels = nx*ny;
        fb_size = num_pixels*sizeof(vec3);

        image.assign(nx * ny * 3,0);

        set_KernelSize();

        std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
        std::cerr << "in " << tx << "x" << ty << " blocks.\n";

        checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

        checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
        checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));

        rand_init<<<1,1>>>(d_rand_state2);

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());


        checkCudaErrors(cudaMalloc(&prims, counts*sizeof(primitive)));
        checkCudaErrors(cudaMalloc(&spheres, counts * sizeof(sphereData)));
        checkCudaErrors(cudaMalloc(&mats, 5 * sizeof(Material)));
        checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));


        create_world<<<1,1>>>(prims,spheres,mats, d_camera, nx, ny, d_rand_state2);
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
        double r = pixel_color.x();
        double g = pixel_color.y();
        double b = pixel_color.z();

        double scale = 1.0 / samples_per_pixel;

        r = std::sqrt(scale * r);
        g = std::sqrt(scale * g);
        b = std::sqrt(scale * b);

        if (r != r){r = 0.0;}
        if (g != g){g = 0.0;}
        if (b != b){b = 0.0;}

        int index = 3 * (y * image_width + x);

        image[index + 0] = static_cast<unsigned char>(256 * std::clamp(r, 0.0, 0.999));
        image[index + 1] = static_cast<unsigned char>(256 * std::clamp(g, 0.0, 0.999));
        image[index + 2] = static_cast<unsigned char>(256 * std::clamp(b, 0.0, 0.999));
    }


    void run() {
        init();


        start = clock();


        render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        render<<<blocks, threads>>>(fb, nx, ny,  ns, d_camera, prims,counts,spheres,triangles,
            quads,mats,d_rand_state);
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
        checkCudaErrors(cudaFree(mats));
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

    Material *mats = nullptr;
    sphereData * spheres = nullptr;
    triangleData * triangles = nullptr;
    quadData * quads = nullptr;
    camera **d_camera = nullptr;

    int counts;

    primitive *prims;
    vec3 *fb;
    int num_pixels;
    size_t fb_size;

    clock_t start, stop;

    dim3 blocks;
    dim3 threads;

    curandState *d_rand_state;
    curandState *d_rand_state2;



    std::vector<unsigned char> image;
    std::string image_name = "cuda_render";
    std::string output_path = "../image/";

};
#endif //SCENE_H
