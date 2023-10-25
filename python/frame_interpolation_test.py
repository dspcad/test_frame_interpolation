from multipledispatch import dispatch
import os, sys, cv2, shutil
import skopt
from skopt import gp_minimize
from natsort import natsorted

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import random
import numpy as np
np.random.seed(237)

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

        self.best_params = {}

        self.fps = 15

        #Bayesian Optimization parameters
        self.n_calls = 20
        self.batch_size = 5

        self.SPACE=[
            skopt.space.Real(0.0, 1.0, name='SCALE', prior='uniform'),
            skopt.space.Real(0.0, 5.0, name='LOD'),
            skopt.space.Integer(3, 9, name='KERNEL'),
            skopt.space.Integer(1, 9, name='STRIDE'),
            skopt.space.space.Categorical([4,8,16,32], name='NGRID')]

        self.obj_fun = "PSNR"
        self.dataset_dir = None
        self.dataset_file_names = None
        self.tmp_interpolated_frame = "tmp_frame_interp_res.png"

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
        print("")
        print(f"  n_calls: {self.n_calls}")
        print(f"  batch_size: {self.batch_size}")
        print("=========================");
 

    def setAlgoParams(self, params) :
        "Set the parameters of the algorithms"
        self.SCALE     = params['SCALE']
        self.LOD       = params['LOD']
        self.THRESHOLD = params['THRESHOLD']
        self.KERNEL    = params['KERNEL']
        self.STRIDE    = params['STRIDE'] 
        self.NGRID     = params['NGRID']



    def setDataset(self, dataset_dir):
        self.dataset_dir = dataset_dir
        files = os.listdir(self.dataset_dir)
        files = [f for f in files if f.endswith(".png")]
        self.dataset_file_names = natsorted(files)

        

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

    

    @dispatch(str, str, str)
    def interpolate_frame(self, img_1, img_2, interpolated_res) -> None:
        "Interpolate the img_1 and img_2 based on the parameters"
        input1_name = os.path.join(self.root_dir, img_1)
        input2_name = os.path.join(self.root_dir, img_2)
        print(f"{self.bin_file} --input1 {input1_name} --input2 {input2_name} --scaleFactor {self.SCALE} --lod {self.LOD} --threshold {self.THRESHOLD} --kernel {self.KERNEL} --stride {self.STRIDE} --ngrid {self.NGRID} --out {interpolated_res} ")
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
                interpolated_res,
            )
        )


    @dispatch(str, str)
    def interpolate_frame(self, img_dir, output_dir) -> None:
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


    def psnr_or_ssim(self, x, img_1_path, img_2_path, ground_truth_path):
        "Evaluate the objective function PSRN or SSIM"
        #PARAMS
        SCALE  = x[0]
        LOD    = x[1]
        THRESHOLD = 0.0
        KERNEL = x[2]
        STRIDE = x[3]
        NGRID  = x[4]
    
    
        print(f"SCALE: {SCALE}   LOD: {LOD}  THRESHOLD: {THRESHOLD}  KERNEL: {KERNEL}  STRIDE: {STRIDE}   NGRID: {NGRID}")
    
    
        os.system("{} --input1 {} --input2 {} --scaleFactor {} --lod {} --threshold {} --kernel {} --stride {} --ngrid {} --out {}".format(
                        self.bin_file,
                        img_1_path,
                        img_2_path,
                        SCALE,
                        LOD,
                        THRESHOLD,
                        KERNEL,
                        STRIDE,
                        NGRID,
                        self.tmp_interpolated_frame,
                    )
                )
    
    
        interpolated_img = cv2.imread(self.tmp_interpolated_frame)
        ground_truth_img = cv2.imread(ground_truth_path)
    
    
        #print(interpolated_img.shape)
        #print(ground_truth_img.shape)
    
        #print(type(interpolated_img))
    
    
        if self.obj_fun == "PSNR":
            return sk_psnr(ground_truth_img, interpolated_img)
        elif self.obj_fun == "SSIM":
            return sk_ssim(ground_truth_img, interpolated_img, data_range=ground_truth_img.max() - ground_truth_img.min(), channel_axis=2)
    

    def f(self, x):
        "The wrapper of the objective function"
        batch = random.choices(self.dataset_file_names[1:-1], k=self.batch_size)
    
        tot=[]
        for f in batch:
            ground_truth_path = os.path.join(self.dataset_dir, f)
            img_1_path        = os.path.join(self.dataset_dir, self.dataset_file_names[self.dataset_file_names.index(f)-1])
            img_2_path        = os.path.join(self.dataset_dir, self.dataset_file_names[self.dataset_file_names.index(f)+1])
            val               = -self.psnr_or_ssim(x, img_1_path, img_2_path, ground_truth_path)
    
            tot.append(val)
    
        print(f"Batch average loss: {np.mean(tot)}")
        return np.mean(tot)
    


    def run_bayesian_opt(self):
        assert self.dataset_dir
        assert self.dataset_file_names

        res = gp_minimize(self.f,          # the function to minimize
                  self.SPACE,              # the bounds on each dimension of x
                  acq_func="PI",           # the acquisition function
                  n_calls=self.n_calls,    # the number of evaluations of f
                  n_random_starts=5,       # the number of random initialization points
                  noise=0.1**2,            # the noise level (optional)
                  random_state=1234)       # the random seed


        #print(res)
        print("Best Predicted Params:")
        print(f"    SCALE: {res.x[0]}")
        print(f"      LOD: {res.x[1]}")
        print(f"   KERNEL: {res.x[2]}")
        print(f"   STRIDE: {res.x[3]}")
        print(f"    NGRID: {res.x[4]}")
        print(f"{self.obj_fun}: {-res.fun}")


        self.best_params['SCALE']  = res.x[0]
        self.best_params['LOD']    = res.x[1]
        self.best_params['KERNEL'] = res.x[2]
        self.best_params['STRIDE'] = res.x[3]
        self.best_params['NGRID']  = res.x[4]

        os.remove(self.tmp_interpolated_frame)


    def eval(self):
        assert self.dataset_dir
        assert self.dataset_file_names
        tot=[]
        params = [self.SCALE, self.LOD, self.KERNEL, self.STRIDE, self.NGRID]
        for i in range(0,len(self.dataset_file_names)):
            if i%2==1 and i<len(self.dataset_file_names)-1:
                ground_truth_path = os.path.join(self.dataset_dir, self.dataset_file_names[i])
                img_1_path        = os.path.join(self.dataset_dir, self.dataset_file_names[i-1])
                img_2_path        = os.path.join(self.dataset_dir, self.dataset_file_names[i+1])
                val               = -self.psnr_or_ssim(params, img_1_path, img_2_path, ground_truth_path)

                tot.append(val)

        print(len(tot))
        print(np.mean(tot))

    def applyBestPredictedParams(self):
        self.SCALE    = self.best_params['SCALE']  
        self.LOD      = self.best_params['LOD']    
        self.KERNEL   = self.best_params['KERNEL'] 
        self.STRIDE   = self.best_params['STRIDE'] 
        self.NGRID    = self.best_params['NGRID']  


