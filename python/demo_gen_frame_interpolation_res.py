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

#demo.SPREAD    = 2.5616899487413622
#demo.LOD       = 4 
#demo.THRESHOLD = 0.0
#demo.KERNEL    = 5
#demo.STRIDE    = 2
#demo.NGRID     = 16




#demo.SPREAD    = 1
#demo.LOD       = 5
demo.SPREAD    = 1.8823612233732991
demo.LOD       = 5
demo.THRESHOLD = 0.0
demo.KERNEL    = 9
demo.STRIDE    = 1
demo.NGRID     = 4



demo.fps = args.fps
output_dir = "res"
frame_interpolation_video = "output_frame_interpolation.avi"
original_video = "output_original.avi"
comp_video = "output.mp4"


demo.info()
demo.interpolate_frame(args.dataset_dir, output_dir)
demo.create_video(os.path.join(demo.root_dir, output_dir), frame_interpolation_video)
demo.create_video(args.dataset_dir, original_video)

#os.system(f"ffmpeg -i {original_video} -i {frame_interpolation_video} -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map '[vid]' -c:v libx264 -crf 23 -preset veryfast {comp_video}")

os.system(f"ffmpeg -i {original_video} -i {frame_interpolation_video} -filter_complex hstack -c:v libx264 -crf 23 -preset veryfast {comp_video}")
