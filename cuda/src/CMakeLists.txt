find_package(OpenCV REQUIRED)
set_source_files_properties(similarity_check.cu PROPERTIES LANGUAGE CUDA)

add_library(similarity_check
            STATIC similarity_check.cu
           )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_compile_features(similarity_check
                        PUBLIC cxx_std_11)
set_target_properties( similarity_check
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "75;80")

