import numpy as np
np.random.seed(237)
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize
import cv2, os, argparse
from math import log10, sqrt 
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

# Instantiate the parser
parser = argparse.ArgumentParser(description='bayesian optimization for frame interpolation in terms of PSNR')

# Required positional argument
parser.add_argument('--dataset_dir', type=str,
                    help='A required input of images')



args = parser.parse_args()
print(args.dataset_dir)
if(not os.path.isdir(args.dataset_dir)):
    print(f"{args.dataset_dir} cannot be found ... make sure input directory is correct")
    sys.exit(1)


params={}

params['SCALE']= 0.5
params['LOD']= 2.5
params['KERNEL']= 7
params['STRIDE']= 1




x = [0.5,5.0,7,1]
bin_file = "/home/hhwu/project/leying/innocompute/build/demos/frame_interp/innocompute_examples_frame_interp"
interpolated_frame = "res_frame_interp.png"

# Root directory of this script
root_dir = os.path.dirname(os.path.realpath(__file__))
print(f"root dir: {root_dir}")


def ssim(x, img_1_path, img_2_path, ground_truth_path):
    #PARAMS
    SCALE  = x[0]
    LOD    = x[1]
    KERNEL = int(x[2]+0.5)
    STRIDE = int(x[3]+0.5)


    print(f"SCALE: {SCALE}   LOD: {LOD}  KERNEL: {KERNEL}  STRIDE: {STRIDE}")
    os.system("{} --input1 {} --input2 {} --scaleFactor {} --lod {} --kernel {} --stride {} --out {}".format(
                bin_file,
                img_1_path,
                img_2_path,
                SCALE,
                LOD,
                KERNEL,
                STRIDE,
                interpolated_frame,
                )
    )

    interpolated_img = cv2.imread(interpolated_frame)
    ground_truth_img = cv2.imread(ground_truth_path)


    #print(interpolated_img.shape)
    #print(ground_truth_img.shape)

    #print(type(interpolated_img))

    return sk_ssim(ground_truth_img, interpolated_img, data_range=ground_truth_img.max() - ground_truth_img.min(), channel_axis=2)

def psnr(x, img_1_path, img_2_path, ground_truth_path):
    #PARAMS
    SCALE  = x[0]
    LOD    = x[1]
    KERNEL = int(x[2]+0.5)
    STRIDE = int(x[3]+0.5)


    print(f"SCALE: {SCALE}   LOD: {LOD}  KERNEL: {KERNEL}  STRIDE: {STRIDE}")
    os.system("{} --input1 {} --input2 {} --scaleFactor {} --lod {} --kernel {} --stride {} --out {}".format(
                bin_file,
                img_1_path,
                img_2_path,
                SCALE,
                LOD,
                KERNEL,
                STRIDE,
                interpolated_frame,
                )
    )

    interpolated_img = cv2.imread(interpolated_frame)
    ground_truth_img = cv2.imread(ground_truth_path)


    #print(interpolated_img.shape)
    #print(ground_truth_img.shape)

    #print(type(interpolated_img))


    return sk_psnr(ground_truth_img, interpolated_img)



#def f(x, noise_level=noise_level):
#    return np.sin(5 * x[0]) * (1 - np.tanh(x[0] ** 2)) + np.random.randn() * noise_level

def f(x):
    img_1_path        = "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/0.png"
    img_2_path        = "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/2.png"
    ground_truth_path = "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/1.png"

    return -psnr(x, img_1_path, img_2_path, ground_truth_path)
    #return -ssim(x, img_1_path, img_2_path, ground_truth_path)


res = gp_minimize(f,                  # the function to minimize
                  [(0, 1.0), (0,5.0), (3,9), (1,9)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=100,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234)   # the random seed

print(res)
print("Best params:")
print(f"    SCALE: {res.x[0]}")
print(f"      LOD: {res.x[1]}")
print(f"   KERNEL: {res.x[2]}")
print(f"   STRIDE: {res.x[3]}")
print(f"PSNR: {-res.fun}")
#print(f"SSIM: {-res.fun}")

print(psnr(res.x, "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/0.png", "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/2.png", "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/1.png"))
#print(ssim(res.x, "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/0.png", "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/2.png", "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/1.png"))
