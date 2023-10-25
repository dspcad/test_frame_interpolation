Some utility codes for testing frame interpolation


# Install the required packages
* ```pip3 install -r requirements.txt```

# Demo codes for how to use innocompute_examples_frame_interp.exe on windows
* Make sure you have binary built on windows
* ```python demo_*.py -h``` to know the usage
* To read two images and produce the interpolated one, please use below for example:
``` python3 demo_two_frames_interpolation_res.py --input_1 ..\..\000000.png --input_2 ..\..\000001.png --bin_file ..\..\innocompute_examples_frame_interp.exe --out res.png```


# CUDA version of PSNR
In case PSNR computation is too slow but now python PSNR looks OK for me.
