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


    img_height = img_1_h;
    img_width  = img_1_w;


    //imshow("Display window", img_2);


    //int k = waitKey(0); // Wait for a keystroke in the window
    //if(k == 's')
    //{
    //    imwrite("starry_night.png", img_1);
    //}


    unsigned long long N = img_height * img_width * channel;
    uint8_t *d_img_1, *d_img_2, *d_diff;

    cudaMalloc((void**)&d_img_1, sizeof(uint8_t) * N);
    cudaMalloc((void**)&d_img_2, sizeof(uint8_t) * N);
    cudaMalloc((void**)&d_diff,  sizeof(uint8_t) * N);
    uint8_t * res_diff = (uint8_t *)malloc(sizeof(uint8_t)*N);

    cudaMemcpy(d_img_1, img_1.data, sizeof(uint8_t) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img_2, img_2.data, sizeof(uint8_t) * N, cudaMemcpyHostToDevice);
    cout << "Transfer image 1 and image 2 to GPU Memory" << endl;


    //const dim3 numBlocks(99, 45);
    //const dim3 threadsPerBlock(18, 18);a

    constexpr int block_size {32};
    printf("DIM: %ld %ld\n",(block_size+img_width-1)/block_size, (block_size+img_height-1)/block_size);
    const dim3 numBlocks((block_size+img_width-1)/block_size, (block_size+img_height-1)/block_size);
    const dim3 threadsPerBlock(block_size, block_size);


    compute_psnr_cuda<<<numBlocks,threadsPerBlock>>>(d_diff,
                                                     d_img_1,
                                                     d_img_2,
                                                     img_1_h,
                                                     img_1_w);

    CHECK_LAST_CUDA_ERROR("PSRN Calculation");
    CHECK_CUDA_ASYNC_ERROR("Calculation");


    cudaMemcpy(res_diff, d_diff, sizeof(uint8_t) * N, cudaMemcpyDeviceToHost);


    //printf("test res: %d\n", res_diff[0]);

    //uint8_t * tmp_img_1 = img_1.data;
    //uint8_t * tmp_img_2 = img_2.data;
    //for(int i=0;i<img_1_h;++i){
    //    for(int j=0;j<3*img_1_w;j+=3){
    //        if(res_diff[i*3*img_1_w+j+0] !=0 || res_diff[i*3*img_1_w+j+1] !=0 || res_diff[i*3*img_1_w+j+2] !=0){
    //            printf("Pixel[%d][%d]:\n",i,j);
    //            printf("    R: %d\n",res_diff[i*3*img_1_w+j+0]);
    //            printf("    G: %d\n",res_diff[i*3*img_1_w+j+1]);
    //            printf("    B: %d\n",res_diff[i*3*img_1_w+j+2]);
    //            printf("CPU R: %d\n",abs(tmp_img_1[i*3*img_1_w+j+0] - tmp_img_2[i*3*img_1_w+j+0]));
    //            printf("CPU G: %d\n",abs(tmp_img_1[i*3*img_1_w+j+1] - tmp_img_2[i*3*img_1_w+j+1]));
    //            printf("CPU B: %d\n",abs(tmp_img_1[i*3*img_1_w+j+2] - tmp_img_2[i*3*img_1_w+j+2]));

    //        }
    //    }
    //}

    //uint8_t * tmp_img_1 = img_1.data;
    //uint8_t * tmp_img_2 = img_2.data;
    //for(int i=0;i<img_1_h;++i){
    //    for(int j=0;j<3*img_1_w;j+=3){
    //        if(tmp_img_1[i*3*img_1_w+j+0] != tmp_img_2[i*3*img_1_w+j+0]){
    //            printf("Pixel[%d][%d] is different\n", i,j/3);
    //            printf("    img_1: %d\n",tmp_img_1[i*3*img_1_w+j+0]);
    //            printf("    img_2: %d\n",tmp_img_2[i*3*img_1_w+j+0]);
    //            printf("    diff: %d\n",res_diff[i*3*img_1_w+j+0]);

    //        }
    //    }
    //}


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

    if(x>=width || y>=height) return;

    d_diff[x+y*width*3+0] = abs(d_img_1[x+y*width*3+0]-d_img_2[x+y*width*3+0]);
    d_diff[x+y*width*3+1] = abs(d_img_1[x+y*width*3+1]-d_img_2[x+y*width*3+1]);
    d_diff[x+y*width*3+2] = abs(d_img_1[x+y*width*3+2]-d_img_2[x+y*width*3+2]);
}
