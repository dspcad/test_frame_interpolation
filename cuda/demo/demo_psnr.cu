#include"similarity_check.hpp"

using namespace std;

int main(int argc, char *argv[]){
    if(argc != 3){
        printf("Usage: %s image_1.png image_2.png\n", argv[0]);
        return 1;
    }

    const char * img_1 = argv[1];
    const char * img_2 = argv[2];


    simcheck::similarity_check engine;
    engine.execute(img_1,
                   img_2);

    return 0;
}
