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




args = parser.parse_args()


if(not os.path.isdir(args.dataset_dir)):
    print(f"{args.dataset_dir} cannot be found ... make sure input directory is correct")
    sys.exit(1)




demo = FrameInterpolationTest(args.bin_file)
demo.setDataset(args.dataset_dir)

demo.SCALE     = 0.01
demo.LOD       = 5.0
demo.THRESHOLD = 0.0
demo.KERNEL    = 5
demo.STRIDE    = 2
demo.NGRID     = 8

demo.eval()

