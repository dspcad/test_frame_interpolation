from frame_interpolation_test import FrameInterpolationTest
import os, sys
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='frame interpolation arguments')

# Required positional argument
parser.add_argument('--dataset_dir', type=str,
                    help='A required input of images')

parser.add_argument('--bin_file', type=str,
                    help='A required bin file for frame interpolation')

parser.add_argument('--fps', type=int, default=15,
                    help='An optional integer fps argument and default is 30')



args = parser.parse_args()
print(args.dataset_dir)
if(not os.path.isdir(args.dataset_dir)):
    print(f"{args.dataset_dir} cannot be found ... make sure input directory is correct")
    sys.exit(1)



#test1 = FrameInterpolationTest("/home/hhwu/project/leying/innocompute/build/demos/frame_interp/innocompute_examples_frame_interp")
demo = FrameInterpolationTest(args.bin_file)
demo.setDataset(args.dataset_dir)
print(f"true VMAF: {demo.scene_vmaf()}");
print(f"true VMAF: {demo.scene_vmaf()}");


