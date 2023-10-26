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
demo.run_bayesian_opt()

demo.applyBestPredictedParams()
#demo.eval()

print(demo.best_params)

output_dir = "res"
frame_interpolation_video = "output_frame_interpolation.avi"
original_video = "output_original.avi"
comp_video = "output.mp4"


demo.interpolate_frame(args.dataset_dir, output_dir)
demo.create_video(os.path.join(demo.root_dir, output_dir), frame_interpolation_video)
demo.create_video(args.dataset_dir, original_video)

os.system(f"ffmpeg -i {original_video} -i {frame_interpolation_video} -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map '[vid]' -c:v libx264 -crf 23 -preset veryfast {comp_video}")
demo.info()


