#include"similarity_check.hpp"


int main(int argc, char *argv[]){
    if(argc != 3){
        printf("Usage: %s image_1.png image_2.png\n", argv[0]);
        return 1;
    }

    simcheck::similarity_check engine;
    engine.execute("/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/0.png",
                   "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/2.png");

    return 0;
}
