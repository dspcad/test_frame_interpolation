find_package(OpenCV REQUIRED)
set_source_files_properties(demo_psnr.cu PROPERTIES LANGUAGE CUDA)
add_executable( demo_psnr demo_psnr.cu )
target_link_libraries( demo_psnr similarity_check ${OpenCV_LIBS})

set_target_properties( demo_psnr
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES "75;80")
