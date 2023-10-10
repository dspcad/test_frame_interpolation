import argparse
import os
import sys
from natsort import natsorted 
import cv2

# Instantiate the parser
parser = argparse.ArgumentParser(description='frame interpolation arguments')

# Required positional argument
parser.add_argument('--input_dir', type=str,
                    help='A required input of images')

parser.add_argument('--fps', type=int, default=15,
                    help='An optional integer fps argument and default is 15')


args = parser.parse_args()
print(args.input_dir)
if(not os.path.isdir(args.input_dir)):
    print(f"{args.input_dir} cannot be found ... make sure input directory is correct")
    sys.exit(1)


# Root directory of this script
root_dir = os.path.dirname(os.path.realpath(__file__))
print(f"root dir: {root_dir}")


def create_video(img_dir):
    files = os.listdir(img_dir)
    files = [f for f in files if f.endswith(".png")]
    natsort_file_names = natsorted(files)
    print(natsort_file_names)

    frame = cv2.imread(os.path.join(img_dir, files[0]))
    height, width, layers = frame.shape

    #fps = 15
    fps = args.fps
    print(f"FPS of video is {fps}")
    video = cv2.VideoWriter("output.avi", 0, fps, (width,height))

    for f in natsort_file_names:
        print(f"img: {f}")
        video.write(cv2.imread(os.path.join(img_dir, f)))

    cv2.destroyAllWindows()
    video.release()


create_video(args.input_dir)
