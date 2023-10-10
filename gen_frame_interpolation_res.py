import subprocess
import socket
import os
import glob
import sys
import time
import subprocess
import shutil
import concurrent.futures

from utils import get_psnr
from utils import get_ssim

# Root directory of this script
root_dir = os.path.dirname(os.path.realpath(__file__))

# List of directories to iterate over, by default, we use the one of the demos
# but in case of "--ci" we run over all the data from the shared filesystem
directories = [
    (
        "GenshinImpact",
        os.path.join(
            root_dir,
            "../../demos/frame_interp/assets/com.miHoYo.GenshinImpact_2023.09.19_13.41.45_OpenGLES_SM8550_OFPC_v2.5_Frame25911-25960",
        ),
    )
]
if "--ci" in sys.argv:
    directories = ["dir1", "dir2", "dir3"]

# Frame interpolation binary path
frame_interp_bin = os.path.join(
    root_dir, "../../build/demos/frame_interp/innocompute_examples_frame_interp"
)

# Internal version of ImageMagick compare
magick_bin = os.path.join(root_dir, "../../3rdparty/magick")

# PSNR threshold
PSNR_THRESHOLD = 0

# SSIM threshold
SSIM_THRESHOLD = 0

# Frame interpolation params specified for games
# =============
# King of Glory
# SCALE = 0.5
# LOD = 4.0
# KERNEL = 9
# STRIDE = 2
# =============
# GenshinImpact
SCALE = 0.5
LOD = 5.0
KERNEL = 7
STRIDE = 1


def create_video(input_dir):
    # Get a list of all PNG files in the input directory
    files = sorted(os.listdir(input_dir))
    files = [f for f in files if f.endswith(".png")]

    # Build the ffmpeg command
    output_file = os.path.join(input_dir, "output.mp4")
    command = [
        "ffmpeg",
        "-fflags",
        "+genpts",
        "-framerate",
        "10",
        "-i",
        os.path.join(input_dir, "%d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "0",
        "-y",
        output_file,
    ]

    # Call the ffmpeg command
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compute_file(tmp_dir, name, directory, png_files, i):
    input1_path = os.path.join(directory, png_files[i])
    input2_path = os.path.join(directory, png_files[i + 2])
    gt_path = os.path.join(directory, png_files[i + 1])

    print("Checking images: {} {}".format(i, i + 2))

    # Check if the current and next png files exist
    if os.path.isfile(input1_path) and os.path.isfile(input2_path):
        # Make sure the files exist
        if os.path.exists(input1_path) and os.path.exists(input2_path):
            out_path = "out_interp_{}.png".format(i + 1)
            # Call our algorithm
            os.system(
                "{} --input1 {} --input2 {} --scaleFactor {} --lod {} --kernel {} --stride {} --out {} > /dev/null 2>&1".format(
                    frame_interp_bin,
                    input1_path,
                    input2_path,
                    SCALE,
                    LOD,
                    KERNEL,
                    STRIDE,
                    out_path,
                )
            )

            # Calculate the PSNR
            psnr = get_psnr(out_path, gt_path)
            ssim = get_ssim(out_path, gt_path)

            # Check if it's valid
            if psnr < PSNR_THRESHOLD:
                print(
                    "ERROR: Image Quality failed! (PSNR={} < {})".format(
                        psnr, PSNR_THRESHOLD
                    )
                )
                # Copy the output failed image to a specific file so it's saved for analysis
                shutil.copy(out_path, "failed_{}_{}".format(name, i))
                sys.exit(1)

            shutil.copy(out_path, os.path.join(tmp_dir, "{}.png".format(i + 1)))
            os.remove(out_path)

            return i, psnr, ssim


# Change to the root directory of the project
os.chdir(root_dir + "/../../")

# Check if we are running on CI servers (inside a docker), if so, then make sure to trigger a build
if "docker" in open("/proc/1/cgroup").read():
    os.system("rm -rf build")
    os.system("cmake -S . -B build -D CMAKE_BUILD_TYPE=Debug")
    os.system("cmake --build build -j8")

# Check if frame interpolation binary is available
if not os.path.exists(frame_interp_bin):
    print(
        "ERROR: '{}' does not exist. Please build the project.".format(frame_interp_bin)
    )
    sys.exit(1)

# Check if the Linux tool ffmpeg is installed on the system
result = subprocess.run(
    ["which", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
if result.returncode != 0:
    print(
        "ERROR: 'ffmpeg' is not installed. Please install with 'sudo apt-get install ffmpeg'"
    )
    sys.exit(1)

# Iterate over each directory
for directory in directories:
    tmp_dir = os.path.join(root_dir, "../../tmp/")
    interp_video_file = os.path.join(
        root_dir, "../../interp_{}.mp4".format(directory[0])
    )
    orig_video_file = os.path.join(
        root_dir, "../../original_{}.mp4".format(directory[0])
    )

    # Create a video from the original frame
    create_video(directory[1])
    shutil.move(os.path.join(directory[1], "output.mp4"), orig_video_file)

    # Create a directory to store frames in order to create a video afterward
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    # Copy the non-interpolated frame (even) to the tmp
    files = glob.glob(directory[1] + "/*[02468].png")
    for file in files:
        shutil.copy(file, tmp_dir)

    print("==== Starting test for '{}' ====".format(directory[0]))

    # Get a list of all png files in the directory
    png_files = [f for f in os.listdir(directory[1]) if f.endswith(".png")]
    png_files.sort(key=lambda x: int(x[:-4]))

    # Track all the values for average computation
    list_ssim = []
    list_psnr = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                compute_file, tmp_dir, directory[0], directory[1], png_files, i
            )
            for i in range(0, len(png_files) - 2, 2)
        ]
        for future in concurrent.futures.as_completed(futures):
            i, psnr, ssim = future.result()
            list_psnr.append(psnr)
            list_ssim.append(ssim)

    # Calculate the averages
    average_psnr = 0
    if len(list_psnr) > 0:
        average_psnr = sum(list_psnr) / len(list_psnr)
    average_ssim = 0
    if len(list_ssim) > 0:
        average_ssim = sum(list_ssim) / len(list_ssim)

    print("Average PSNR={}".format(average_psnr))
    print("Average SSIM={}".format(average_ssim))

    print("==== End of test ====")

    # Create the video and remove the temporary directory
    create_video(tmp_dir)
    shutil.move(os.path.join(tmp_dir, "output.mp4"), interp_video_file)
    #shutil.rmtree(tmp_dir)

    # Create a report directory with all the necessary information
    report_dir = os.path.join(root_dir, "../../report_{}/".format(directory[0]))
    if os.path.exists(report_dir):
        shutil.rmtree(report_dir)
    os.mkdir(report_dir)

    shutil.move(interp_video_file, os.path.join(report_dir, "interp.mp4"))
    shutil.move(orig_video_file, os.path.join(report_dir, "original.mp4"))

    with open(os.path.join(report_dir, "report.txt"), "w") as f:
        f.write("TEST_NAME={}\n".format(directory[0]))
        f.write("DIRECTORY={}\n".format(directory[1]))
        f.write("AVERAGE_PSNR={}\n".format(average_psnr))
        f.write("AVERAGE_SSIM={}\n".format(average_ssim))

print("TESTS PASSED")
