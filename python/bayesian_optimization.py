import numpy as np
np.random.seed(237)
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize
import skopt, sys
import cv2, os, argparse
from math import log10, sqrt 
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from natsort import natsorted
import random

# Instantiate the parser
parser = argparse.ArgumentParser(description='bayesian optimization for frame interpolation in terms of PSNR')

# Required positional argument
parser.add_argument('--dataset_dir', type=str,
                    help='A required input of images')

parser.add_argument('--bin_file', type=str, 
                    help='A required bin file for frame interpolation')


parser.add_argument('--obj_fun', type=str, default="PSNR",
                    help='A opitonal objective function (either PSNR or SSIM), default is PSNR')

funs = ["PSNR","SSIM"]


args = parser.parse_args()


if(not os.path.isdir(args.dataset_dir)):
    print(f"{args.dataset_dir} cannot be found ... make sure input directory is correct")
    sys.exit(1)

if args.obj_fun not in funs:
    print(f"objective function: {args.obj_fun}")
    print(f"Supported objective functions: {funs}")
    sys.exit(1)



x = [0.5,5.0,7,1]
#bin_file = "/home/hhwu/project/leying/innocompute/build/demos/frame_interp/innocompute_examples_frame_interp"
bin_file = args.bin_file
interpolated_frame = "res_frame_interp.png"

# Root directory of this script
root_dir = os.path.dirname(os.path.realpath(__file__))


print("====================================")
print(f"= root dir: {root_dir}")
print(f"= bin file: {bin_file}")
print(f"= objective function: {args.obj_fun}")
print(f"= output file: {interpolated_frame}")
print("====================================")

files = os.listdir(args.dataset_dir)
files = [f for f in files if f.endswith(".png")]
natsort_file_names = natsorted(files)
print(natsort_file_names)
print(natsort_file_names[:-1])





def psnr_or_ssim(x, img_1_path, img_2_path, ground_truth_path):
    #PARAMS
    SCALE  = x[0]
    LOD    = x[1]
    THRESHOLD = 0.0
    KERNEL = x[2]
    STRIDE = x[3]
    NGRID  = x[4]


    print(f"SCALE: {SCALE}   LOD: {LOD}  THRESHOLD: {THRESHOLD}  KERNEL: {KERNEL}  STRIDE: {STRIDE}   NGRID: {NGRID}")


    os.system("{} --input1 {} --input2 {} --scaleFactor {} --lod {} --threshold {} --kernel {} --stride {} --ngrid {} --out {}".format(
                    bin_file,
                    img_1_path,
                    img_2_path,
                    SCALE,
                    LOD,
                    THRESHOLD,
                    KERNEL,
                    STRIDE,
                    NGRID,
                    interpolated_frame,
                )
            )


    interpolated_img = cv2.imread(interpolated_frame)
    ground_truth_img = cv2.imread(ground_truth_path)


    #print(interpolated_img.shape)
    #print(ground_truth_img.shape)

    #print(type(interpolated_img))


    if args.obj_fun == "PSNR":
        return sk_psnr(ground_truth_img, interpolated_img)
    elif args.obj_fun == "SSIM":
        return sk_ssim(ground_truth_img, interpolated_img, data_range=ground_truth_img.max() - ground_truth_img.min(), channel_axis=2)




def f(x):

    batch = random.choices(natsort_file_names[1:-1], k=5)

    tot=[]
    for f in batch:
        ground_truth_path = os.path.join(args.dataset_dir, f)
        img_1_path        = os.path.join(args.dataset_dir, natsort_file_names[natsort_file_names.index(f)-1])
        img_2_path        = os.path.join(args.dataset_dir, natsort_file_names[natsort_file_names.index(f)+1])
        val               = -psnr_or_ssim(x, img_1_path, img_2_path, ground_truth_path)

        tot.append(val)

    print(f"Batch average loss: {np.mean(tot)}")
    return np.mean(tot)



#############################
# Parameter Searching Space #
#############################
SPACE=[
    skopt.space.Real(0.0, 1.0, name='SCALE', prior='uniform'),
    skopt.space.Real(0.0, 5.0, name='LOD'),
    skopt.space.Integer(3, 9, name='KERNEL'),
    skopt.space.Integer(1, 9, name='STRIDE'),
    skopt.space.space.Categorical([4,8,16,32], name='NGRID')]


res = gp_minimize(f,                  # the function to minimize
                  SPACE,              # the bounds on each dimension of x
                  acq_func="PI",      # the acquisition function
                  n_calls=50,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234)  # the random seed


print(res)
print("Best params:")
print(f"    SCALE: {res.x[0]}")
print(f"      LOD: {res.x[1]}")
print(f"   KERNEL: {res.x[2]}")
print(f"   STRIDE: {res.x[3]}")
print(f"    NGRID: {res.x[4]}")
print(f"{args.obj_fun}: {-res.fun}")
#print(f"SSIM: {-res.fun}")



#print(psnr_or_ssim(res.x, "/home/hhwu/project/leying/test_frame_interpolation/python/wzry/000002.png", "/home/hhwu/project/leying/test_frame_interpolation/python/wzry/000004.png", "/home/hhwu/project/leying/test_frame_interpolation/python/wzry/000003.png"))
#print(ssim(res.x, "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/0.png", "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/2.png", "/home/hhwu/project/leying/test_frame_interpolation/python/test_imgs/1.png"))
