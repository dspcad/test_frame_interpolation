#include<bits/stdc++.h>
namespace simcheck{

    class similarity_check{
        public:
            unsigned int img_height;
            unsigned int img_width;
            unsigned int channel;      

            similarity_check(): image_1{nullptr}, image_2{nullptr}, img_height{1}, img_width{1}, channel{3}{}
            ~similarity_check()=default;

            
            __host__ double execute(const char * image_1_path,
                                    const char * image_2_path);

        private:
            unsigned int *image_1;
            unsigned int *image_2;

    };
}// end of namespace  simcheck



__global__ void compute_rgb_diff_cuda(int *d_diff,
                                      uint8_t *d_img_1,
                                      uint8_t *d_img_2,
                                      unsigned int height,
                                      unsigned int width);

__global__ void compute_psnr_cuda(unsigned long long *d_diff_sum,
                                  uint8_t *d_img_1,
                                  uint8_t *d_img_2,
                                  unsigned int height,
                                  unsigned int width);

