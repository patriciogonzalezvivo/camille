
#include <string>
#include <time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#define BLOCK_SIZE 8

#include "hilma/ops/fs.h"

#include "hilma/io/obj.h"
#include "hilma/io/ply.h"
#include "hilma/io/stl.h"
#include "hilma/io/gltf.h"
#include "hilma/io/png.h"

#include "hilma/types/image.h"

#include "hilma/ops/generate.h"
#include "hilma/ops/transform.h"
#include "hilma/ops/raytrace.h"
#include "hilma/ops/intersection.h"
#include "hilma/ops/compute.h"
#include "hilma/ops/image.h"

#include "lygia/math/make.cuh"
#include "lygia/math/cross.cuh"
#include "lygia/math/clamp.cuh"
#include "lygia/math/floor.cuh"
#include "lygia/math/length.cuh"
#include "lygia/math/normalize.cuh"
#include "lygia/math/operations.cuh"
#include "lygia/geometry/aabb.cuh"
#include "lygia/geometry/triangle.cuh"

__global__ void kernel( Triangle* _tris, float3* _trisNormals, int _Ntris, 
                        float *_pixels, 
                        AABB _aabb, float3 _bdiagonal, float _max_dist, 
                        float _voxel_size, int _voxel_resolution, int _layersTotal)  {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= _voxel_resolution || y >= _voxel_resolution || z >= _voxel_resolution)
        return;

    float3 p = make_float3((float)x, (float)y, (float)z) * _voxel_size;
    p = _aabb.min + p * _bdiagonal;

    float min_dist = 99999.9f;
    for (int i = 0; i < _Ntris; i++ ) {
        float dist = signedDistance(_tris[i], _trisNormals[i], p);
        if (abs(dist) < abs(min_dist) )
            min_dist = dist;
    }
    min_dist = min_dist/_max_dist;
    min_dist = clamp(min_dist, -1.0f, 1.0f) * 0.5 + 0.5;

    int layerX = (z % _layersTotal) * _voxel_resolution; 
    int layerY = (z / _layersTotal) * _voxel_resolution;

    int width = _voxel_resolution * _layersTotal;
    int index = (layerX + x) + (layerY + y) * width;
    _pixels[index] = min_dist;
}

int main(int argc, char **argv) {

    std::string filename = std::string(argv[1]);
    std::string ext = hilma::getExt(filename);

    hilma::Mesh mesh;
    if ( ext == "ply" || ext == "PLY" )
        hilma::loadPly( filename, mesh );

    else if ( ext == "obj" || ext == "OBJ" )
        hilma::loadObj( filename, mesh );

    else if ( ext == "stl" || ext == "STL" )
        hilma::loadStl( filename, mesh );

    else if (   ext == "gltf" || ext == "GLTF" ||
                ext == "glb" || ext == "GLB" )
        hilma::loadGltf( filename, mesh );

    std::cout << "Mesh loaded" << std::endl;

    clock_t start, end;
    start = clock();

    hilma::center(mesh);
    hilma::BoundingBox bbox = getBoundingBox(mesh);
    bbox.square();
    std::vector<hilma::Triangle> mesh_triangles = mesh.getTriangles();

    // Create array triangles and axis aligned bounding box 
    Triangle*   cpuTris = new Triangle[mesh_triangles.size()];
    float3*     cpuTrisNormals = new float3[mesh_triangles.size()];

    AABB aabb;
    aabb.min = make_float3(99999.9f);
    aabb.max = make_float3(-99999.9f);

    for (size_t i = 0; i < mesh_triangles.size(); i++) {
        cpuTris[i].a = make_float3(mesh_triangles[i][0]);
        cpuTris[i].b = make_float3(mesh_triangles[i][1]);
        cpuTris[i].c = make_float3(mesh_triangles[i][2]);
        cpuTrisNormals[i] = normal(cpuTris[i]);
        expand( aabb, cpuTris[i] );
    }

    // square the aabb on the longest side
    square(aabb);

    Triangle* gpuTris;
    cudaMalloc(&gpuTris, sizeof(Triangle) * mesh_triangles.size());
    cudaMemcpy(gpuTris, cpuTris, sizeof(Triangle) * mesh_triangles.size(), cudaMemcpyHostToDevice);

    float3* gpuTrisNormals;
    cudaMalloc(&gpuTrisNormals, sizeof(Triangle) * mesh_triangles.size());
    cudaMemcpy(gpuTrisNormals, cpuTrisNormals, sizeof(float3) * mesh_triangles.size(), cudaMemcpyHostToDevice);

    // expand it just a bit
    float        paddingPct = 0.001f;
    float3        bdiagonal = diagonal(aabb);
    float          max_dist = length(bdiagonal);
    expand( aabb, (max_dist*max_dist) * paddingPct );
    max_dist *= 0.5f;

    // Calculate the voxel matrix resolution
    int          resolution = 8;
    int    voxel_resolution = std::pow(2, resolution);
    float        voxel_size = 1.0/float(voxel_resolution);
    int         layersTotal = std::sqrt(voxel_resolution);

    // Calculate the image resolution and allocate pixels
    int    image_resolution = voxel_resolution * layersTotal;
    int      image_channels = 1;
    float *image_pixels;
    cudaMalloc(&image_pixels, image_channels * image_resolution * image_resolution * sizeof(float));

    // define Kernel
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(voxel_resolution / threads.x + 1, 
                voxel_resolution / threads.y + 1, 
                voxel_resolution / threads.z + 1);

    // run the kernel
    kernel<<<blocks, threads>>>(gpuTris, gpuTrisNormals, (int)mesh_triangles.size(), image_pixels, aabb, bdiagonal, max_dist, voxel_size, voxel_resolution, layersTotal);

    // free(cpuTris);
    cudaFree(gpuTris);
    hilma::Image sdf = hilma::Image(image_resolution, image_resolution, image_channels);
    cudaMemcpy(&sdf[0], image_pixels, image_channels * image_resolution * image_resolution * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(image_pixels);
    
    end = clock();
    double duration_sec = double(end-start)/CLOCKS_PER_SEC;

    std::cout << "Took " << duration_sec << "secs" << std::endl;

    // Save Image
    filename.erase(filename.length() - ext.length());
    filename += "png";

    hilma::flip(sdf);
    hilma::savePng(filename, sdf);

    return 1;
}
