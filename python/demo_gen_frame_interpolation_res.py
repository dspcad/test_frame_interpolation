from frame_interpolation_test import FrameInterpolationTest
import os, sys
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='frame interpolation arguments')

# Required positional argument
parser.add_argument('--input_dir', type=str,
                    help='A required input of images')

parser.add_argument('--bin_file', type=str,
                    help='A required bin file for frame interpolation')

parser.add_argument('--fps', type=int, default=15,
                    help='An optional integer fps argument and default is 15')



args = parser.parse_args()
print(args.input_dir)
if(not os.path.isdir(args.input_dir)):
    print(f"{args.input_dir} cannot be found ... make sure input directory is correct")
    sys.exit(1)



#test1 = FrameInterpolationTest("/home/hhwu/project/leying/innocompute/build/demos/frame_interp/innocompute_examples_frame_interp")
test1 = FrameInterpolationTest(args.bin_file)


output_dir = "res"
frame_interpolation_video = "frame_interpolation_output.avi"
original_video = "original_output.avi"
comp_video = "output.mp4"

test1.interpolate_frame(args.input_dir, output_dir)
test1.create_video(os.path.join(test1.root_dir, output_dir), frame_interpolation_video)
test1.create_video(args.input_dir, original_video)

#os.system(f"ffmpeg -i {original_video} -i {frame_interpolation_video} -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map '[vid]' -c:v libx264 -crf 23 -preset veryfast {comp_video}")

os.system(f"ffmpeg -i {original_video} -i {frame_interpolation_video} -filter_complex hstack -c:v libx264 -crf 23 -preset veryfast {comp_video}")
