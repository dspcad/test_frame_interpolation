import argparse
import os
import sys
from natsort import natsorted 
import cv2
import shutil

# Instantiate the parser
parser = argparse.ArgumentParser(description='frame interpolation arguments')

# Required positional argument
parser.add_argument('--input_dir', type=str,
                    help='A required input of images')

parser.add_argument('--fps', type=int, default=15,
                    help='An optional integer fps argument and default is 15')


parser.add_argument('--bin_file', type=str,
                    help='A required bin file for frame interpolation')


args = parser.parse_args()
print(args.input_dir)
if(not os.path.isdir(args.input_dir)):
    print(f"{args.input_dir} cannot be found ... make sure input directory is correct")
    sys.exit(1)


# Root directory of this script
root_dir = os.path.dirname(os.path.realpath(__file__))
print(f"root dir: {root_dir}")

# Frame interpolation params specified for games
# =============
# King of Glory
# SCALE = 0.5
# LOD = 4.0
# KERNEL = 9
# STRIDE = 2
# =============
# GenshinImpact
#SCALE = 0.5
#LOD = 5.0
#THRESHOLD = 0.0
#KERNEL = 7
#STRIDE = 1
#NGRID  =

SCALE  = 0.15063696560609532
LOD    = 0.9925937951308552
THRESHOLD = 0.0
KERNEL = 8
STRIDE = 2
NGRID  = 4




def create_video(img_dir, video_name):
    files = os.listdir(img_dir)
    files = [f for f in files if f.endswith(".png")]
    natsort_file_names = natsorted(files)
    print(natsort_file_names)

    frame = cv2.imread(os.path.join(img_dir, files[0]))
    height, width, layers = frame.shape

    #fps = 15
    fps = args.fps
    print(f"FPS of video is {fps}")
    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

    for f in natsort_file_names:
        print(f"img: {f}")
        video.write(cv2.imread(os.path.join(img_dir, f)))

    cv2.destroyAllWindows()
    video.release()


def interpolate_frame(img_dir,output_name):
    files = os.listdir(img_dir)
    files = [f for f in files if f.endswith(".png")]
    natsort_file_names = natsorted(files)


    output_path = os.path.join(root_dir, output_name);
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)

    os.mkdir(output_path)
    print(natsort_file_names)
    for i in range(0,len(natsort_file_names)):

        if i%2==1 and i<len(natsort_file_names)-1:
            input1_name = os.path.join(img_dir, natsort_file_names[i-1])
            input2_name = os.path.join(img_dir, natsort_file_names[i+1])
            print(f"{args.bin_file} --input1 {input1_name} --input2 {input2_name} --scaleFactor {SCALE} --lod {LOD} --threshold {THRESHOLD} --kernel {KERNEL} --stride {STRIDE} --ngrid {NGRID} --out {natsort_file_names[i]} ")
            os.system("{} --input1 {} --input2 {} --scaleFactor {} --lod {} --threshold {} --kernel {} --stride {} --ngrid {} --out {}".format(
                    args.bin_file,
                    input1_name,
                    input2_name,
                    SCALE,
                    LOD,
                    THRESHOLD,
                    KERNEL,
                    STRIDE,
                    NGRID,
                    natsort_file_names[i],
                )
            )



            shutil.copy(natsort_file_names[i], output_path)
            os.remove(natsort_file_names[i])
        else:
            frame = cv2.imread(os.path.join(img_dir, natsort_file_names[i]))
            cv2.imwrite(os.path.join(output_path, natsort_file_names[i]), frame)
        print(f"{i}: {natsort_file_names[i]}")


output_name = "res"
frame_interpolation_video = "frame_interpolation_output.avi"
original_video = "original_output.avi"
comp_video = "output.mp4"

interpolate_frame(args.input_dir, output_name)
create_video(os.path.join(root_dir, output_name), frame_interpolation_video)
create_video(args.input_dir, original_video)

#create_video(args.input_dir)

os.system(f"ffmpeg -i {original_video} -i {frame_interpolation_video} -filter_complex '[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]' -map '[vid]' -c:v libx264 -crf 23 -preset veryfast {comp_video}")

