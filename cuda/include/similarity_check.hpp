#include<bits/stdc++.h>
namespace simcheck{

    class similarity_check{
        public:
            similarity_check(): image_1{nullptr}, image_2{nullptr}, img_height{1}, img_width(1) {}
            ~similarity_check()=default;

            
            __host__ void execute(const char * image_1_path,
                                 const char * image_2_path);

        private:
            unsigned int *image_1;
            unsigned int *image_2;
            unsigned int img_height;
            unsigned int img_width;

    };
}// end of namespace  simcheck



__global__ void compute_psnr_cuda(uint8_t *d_diff,
                                  uint8_t *d_img_1,
                                  uint8_t *d_img_2,
                                  unsigned int height,
                                  unsigned int width);
