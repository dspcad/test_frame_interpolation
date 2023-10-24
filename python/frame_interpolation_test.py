import os, sys, cv2, shutil
from natsort import natsorted

class FrameInterpolationTest:
    "A framework to test frame interpolation and fine tune the parameters of the algorithm"

    def __init__(self, bin_file):
        "Initialize a test with a binary file and the default parameters"

        self.bin_file = bin_file

        self.root_dir = os.path.dirname(os.path.realpath(__file__)) 

        self.SCALE     = 0.5
        self.LOD       = 5.0
        self.THRESHOLD = 0.0
        self.KERNEL    = 7
        self.STRIDE    = 1
        self.NGRID     = 32

        self.fps = 15


        self.info()


    def info(self):
        print("=========================");
        print(f"  root dir: {self.root_dir}")
        print(f"  SCALE: {self.SCALE}")
        print(f"  LOD: {self.LOD}")
        print(f"  THRESHOLD: {self.THRESHOLD}")
        print(f"  KERNEL: {self.KERNEL}")
        print(f"  STRIDE: {self.STRIDE}")
        print(f"  NGRID: {self.NGRID}")
        print("=========================");
 

    def setAlgoParams(self, params) :
        "Set the parameters of the algorithms"
        self.SCALE     = params['SCALE']
        self.LOD       = params['LOD']
        self.THRESHOLD = params['THRESHOLD']
        self.KERNEL    = params['KERNEL']
        self.STRIDE    = params['STRIDE'] 
        self.NGRID     = params['NGRID']


        

    def create_video(self, img_dir, video_name):
        "Create a video of img_dir"
        files = os.listdir(img_dir)
        files = [f for f in files if f.endswith(".png")]
        natsort_file_names = natsorted(files)
        print(natsort_file_names)

        frame = cv2.imread(os.path.join(img_dir, files[0]))
        height, width, layers = frame.shape

        #fps = 15
        print(f"FPS of video is {self.fps}")
        video = cv2.VideoWriter(video_name, 0, self.fps, (width,height))

        for f in natsort_file_names:
            print(f"img: {f}")
            video.write(cv2.imread(os.path.join(img_dir, f)))

        cv2.destroyAllWindows()
        video.release()


    def interpolate_frame(self, img_dir,output_dir):
        "Interpolate the frames of even number and clone the frames of odd number"
        files = os.listdir(img_dir)
        files = [f for f in files if f.endswith(".png")]
        natsort_file_names = natsorted(files)
    
    
        output_path = os.path.join(self.root_dir, output_dir);
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
    
        os.mkdir(output_path)
        print(natsort_file_names)
        for i in range(0,len(natsort_file_names)):
    
            if i%2==1 and i<len(natsort_file_names)-1:
                input1_name = os.path.join(img_dir, natsort_file_names[i-1])
                input2_name = os.path.join(img_dir, natsort_file_names[i+1])
                print(f"{self.bin_file} --input1 {input1_name} --input2 {input2_name} --scaleFactor {self.SCALE} --lod {self.LOD} --threshold {self.THRESHOLD} --kernel {self.KERNEL} --stride {self.STRIDE} --ngrid {self.NGRID} --out {natsort_file_names[i]} ")
                os.system("{} --input1 {} --input2 {} --scaleFactor {} --lod {} --threshold {} --kernel {} --stride {} --ngrid {} --out {}".format(
                        self.bin_file,
                        input1_name,
                        input2_name,
                        self.SCALE,
                        self.LOD,
                        self.THRESHOLD,
                        self.KERNEL,
                        self.STRIDE,
                        self.NGRID,
                        natsort_file_names[i],
                    )
                )
    
    
    
                shutil.copy(natsort_file_names[i], output_path)
                os.remove(natsort_file_names[i])
            else:
                frame = cv2.imread(os.path.join(img_dir, natsort_file_names[i]))
                cv2.imwrite(os.path.join(output_path, natsort_file_names[i]), frame)
            print(f"{i}: {natsort_file_names[i]}")

