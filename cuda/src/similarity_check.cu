#include "similarity_check.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;
using namespace simcheck;


__host__ void CHECK_LAST_CUDA_ERROR(const char * kernel_name)
{
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

__host__ void CHECK_CUDA_ASYNC_ERROR(const char * kernel_name)
{
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
                                        const char * image_2_path)
{   
    Mat img_1 = imread(image_1_path, IMREAD_COLOR);
    Mat img_2 = imread(image_2_path, IMREAD_COLOR);

    int img_1_w = img_1.size().width;
    int img_1_h = img_1.size().height;

    int img_2_w = img_2.size().width;
    int img_2_h = img_2.size().height;

    printf("Image: %s\n", image_1_path);
    printf("    width: %d\n", img_1_w);
    printf("    height: %d\n", img_1_h);
    printf("    type: %d\n", img_1.type());
 
    printf(" debug: %d\n", *img_1.data);

    printf("Image: %s\n", image_2_path);
    printf("    width: %d\n", img_2_w);
    printf("    height: %d\n", img_2_h);
    printf("    type: %d\n", img_2.type());



    if(img_1.empty())
    {
        cout << "Could not read the image: " << image_1_path << endl;
        return;
    }

    if(img_2.empty())
    {
        cout << "Could not read the image: " << image_2_path << endl;
        return;
    }


    if(img_1_w != img_2_w || img_1_h != img_2_h)
    {
        cout << "Two images are not in the same size." << endl;
    }



    //imshow("Display window", img_2);


    //int k = waitKey(0); // Wait for a keystroke in the window
    //if(k == 's')
    //{
    //    imwrite("starry_night.png", img_1);
    //}


    unsigned long long N = img_1_h * img_1_w * 3;
    uint8_t *d_img_1, *d_img_2, *d_diff;

    cudaMalloc((void**)&d_img_1, sizeof(uint8_t) * N);
    cudaMalloc((void**)&d_img_2, sizeof(uint8_t) * N);
    cudaMalloc((void**)&d_diff,  sizeof(uint8_t) * N);
    uint8_t * res_diff = (uint8_t *)malloc(sizeof(uint8_t)*N);

    cudaMemcpy(d_img_1, img_1.data, sizeof(uint8_t) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_2, img_2.data, sizeof(uint8_t) * N, cudaMemcpyHostToDevice);
    cout << "Transfer image 1 and image 2 to GPU Memory" << endl;


    const dim3 numBlocks(100, 100);
    const dim3 threadsPerBlock(10, 10);

    compute_psnr_cuda<<<numBlocks,threadsPerBlock>>>(d_diff,
                                                     d_img_1,
                                                     d_img_2,
                                                     img_1_h,
                                                     img_1_w);

    cudaMemcpy(res_diff, d_diff, sizeof(uint8_t) * N, cudaMemcpyDeviceToHost);


    printf("test res: %d\n", res_diff[0]);
    cudaFree(d_img_1);
    cudaFree(d_img_2);
    cudaFree(d_diff);
    return;
}

__global__ void compute_psnr_cuda(uint8_t *d_diff,
                                  uint8_t *d_img_1,
                                  uint8_t *d_img_2,
                                  unsigned int height,
                                  unsigned int width)
{

    const int x = threadIdx.x+blockIdx.x*blockDim.x;
    const int y = threadIdx.y+blockIdx.y*blockDim.y;

    d_diff[x+y] = abs(d_img_1[x+y]-d_img_2[x+y]);
    d_diff[0] = 199;
}
