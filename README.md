Some utility codes for testing frame interpolation


# Install the required packages
* ```pip3 install -r requirements.txt```

# Demo codes for how to use innocompute_examples_frame_interp.exe on windows
* Make sure you have binary built on windows
* ```python demo_*.py -h``` to know the usage
* To read two images and produce the interpolated one, please use below for example:
``` python3 demo_two_frames_interpolation_res.py --input_1 ..\..\000000.png --input_2 ..\..\000001.png --bin_file ..\..\innocompute_examples_frame_interp.exe --out res.png```

# FFmpeg cheat sheet

* Generate the video from the images
``` cat Genshin_1600x900_fsr2on_uioff_60fps_map_wild_2023-10-04_11-30-30_view_report/*.png | ffmpeg -framerate 60 -f image2pipe -i - -c:v libx264 -r 60 -pix_fmt yuv420p blend_output.mp4 ```


# CUDA version of PSNR
In case PSNR computation is too slow but now python PSNR looks OK for me.
