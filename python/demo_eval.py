from frame_interpolation_test import FrameInterpolationTest
import os, sys
import argparse

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



demo = FrameInterpolationTest(args.bin_file)
demo.setDataset(args.dataset_dir)
demo.eval()

