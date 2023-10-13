#include "similarity_check.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;
using namespace simcheck;


__host__ void CHECK_LAST_CUDA_ERROR(const char * kernel_name){
    cudaError_t cudaerr {cudaGetLastError()};

    printf("----- Kernel \"%s\" ----- \n", kernel_name);
    if (cudaerr != cudaSuccess){
        printf("    CUDA Runtime Error at \"%s\".\n", cudaGetErrorString(cudaerr));
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
    else
        printf("    No CUDA Runtime Error (No Synchronous Error)\n\n\n");

}

__host__ void CHECK_CUDA_ASYNC_ERROR(const char * kernel_name){
    cudaError_t cudaerr {cudaDeviceSynchronize()};
    //printf("Kernel \"%s\": \n", kernel_name);
    printf("----- Kernel \"%s\" ----- \n", kernel_name);
    if (cudaerr != cudaSuccess){
        printf("    kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
    }
    else
        printf("    Successfully launch the kernel (No Asynchronous Error)\n\n\n");

}

__host__ void similarity_check::execute(const char * image_1_path,
                                        const char * image_2_path){
   
    Mat img = imread(image_1_path, IMREAD_COLOR);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_1_path << std::endl;
        return;
    }
    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }
    return;
}

__global__ void compute_psnr(double *out,
                             unsigned int *image_1,
                             unsigned int *image_2,
                             unsigned int height,
                             unsigned int width){

    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;
}
