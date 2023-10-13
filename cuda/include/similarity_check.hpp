#include<bits/stdc++.h>

namespace simcheck{

    class similarity_check{
        unsigned int *image_1;
        unsigned int *image_2;
        unsigned int img_height;
        unsigned int img_width;
        public:
            similarity_check(): image_1{nullptr}, image_2{nullptr}, img_height{1}, img_width(1) {}
            ~similarity_check()=default;

            __host__ void execute(const char * image_1_path,
                                 const char * image_2_path);
    };
}// end of namespace  simcheck



__global__ void compute_psnr(double *out,
                             unsigned int *image_1,
                             unsigned int *image_2,
                             unsigned int height,
                             unsigned int width);
