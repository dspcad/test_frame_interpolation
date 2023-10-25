from frame_interpolation_test import FrameInterpolationTest
import os, sys
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='frame interpolation arguments')

# Required positional argument
parser.add_argument('--input_1', type=str,
                    help='A required input of image 1')

parser.add_argument('--input_2', type=str,
                    help='A required input of image 2')

parser.add_argument('--bin_file', type=str,
                    help='A required bin file for frame interpolation')

parser.add_argument('--out', type=str,
                    help='A required output file name')



args = parser.parse_args()
if(not os.path.isfile(args.input_1)):
    print(f"{args.input_1} cannot be found ... make sure the path to input image 1 is correct")
    sys.exit(1)

if(not os.path.isfile(args.input_2)):
    print(f"{args.input_2} cannot be found ... make sure the path to input image 2 is correct")
    sys.exit(1)

if(not os.path.isfile(args.bin_file)):
    print(f"{args.bin_file} cannot be found ... make sure the path to bin file is correct")
    sys.exit(1)



demo = FrameInterpolationTest(args.bin_file)
demo.interpolate_frame(args.input_1, args.input_2, args.out)
