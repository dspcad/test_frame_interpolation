from frame_interpolation_test import FrameInterpolationTest
import os, sys
import argparse

# Instantiate the parser
parser = argparse.ArgumentParser(description='bayesian optimization for frame interpolation in terms of PSNR')

# Required positional argument
parser.add_argument('--dataset_list', type=str,
                    help='A required input of images')

parser.add_argument('--bin_file', type=str, 
                    help='A required bin file for frame interpolation')


parser.add_argument('--calls', type=int, default="20",
                    help='A opitonal number for running the algorithm')



funs = ["PSNR","SSIM","VMAF"]


args = parser.parse_args()


if(not os.path.isfile(args.dataset_list)):
    print(f"{args.dataset_list} cannot be found ... make sure dataset_list file exists")
    sys.exit(1)




demo = FrameInterpolationTest(args.bin_file)
demo.n_calls = args.calls;


with open(args.dataset_list, 'r') as f:
    dataset_list = [dataset.strip() for dataset in f.readlines() if dataset.strip()]

#dataset_list = ["Genshin/1600x900_fsr2on_uioff_60fps/map_wild/2023-10-04_11-30-30/view",
#                "Genshin/1600x900_fsr2on_uioff_60fps/map_wild/2023-10-04_11-41-34/view", 
#                "Genshin/1600x900_fsr2on_uioff_60fps/map_wild/2023-10-04_12-00-54/view"]


demo.setDatasetList(dataset_list)
demo.info()
demo.run_bayesian_opt()

print(demo.best_params)






######################################################
# Create the videos of original and interpolated one #
######################################################
#output_dir = "res"
#frame_interpolation_video = "output_frame_interpolation.avi"
#original_video = "output_original.avi"
#comp_video = "output.mp4"
#
#
#demo.interpolate_frame(args.dataset_dir, output_dir)
#demo.create_video(os.path.join(demo.root_dir, output_dir), frame_interpolation_video)
#demo.create_video(args.dataset_dir, original_video)
#
#os.system(f"ffmpeg -i {original_video} -i {frame_interpolation_video} -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map '[vid]' -c:v libx264 -crf 23 -preset veryfast {comp_video}")
#demo.info()


