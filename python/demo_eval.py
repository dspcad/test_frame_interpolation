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


demo.SCALE     = 0.5
demo.SPREAD    = 2.92
demo.LOD       = 2
demo.THRESHOLD = 0.0
demo.KERNEL    =7
demo.STRIDE    =1
demo.NGRID     =8

params=[2.92, 2, 7, 1, 8]

#demo.SCALE     = 0.903
#demo.LOD       = 3.0
#demo.THRESHOLD = 0.0
#demo.KERNEL    = 9
#demo.STRIDE    = 1
#demo.NGRID     = 4



demo.eval(params, args.dataset_dir)

