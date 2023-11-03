from multipledispatch import dispatch
import os, sys, cv2, shutil
import platform
import skopt
from skopt import gp_minimize
from natsort import natsorted
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import random
import numpy as np
np.random.seed(237)
import matplotlib.pyplot as plt

SUPRESS_MSG = ''
if platform.system() == 'Linux':
    SUPRESS_MSG = '1>/dev/null'
elif platform.system() == 'Windows':
    SUPRESS_MSG = ' > nul'

class FrameInterpolationTest:
    "A framework to test frame interpolation and fine tune the parameters of the algorithm"

    def __init__(self, bin_file):
        "Initialize a test with a binary file and the default parameters"

        self.bin_file = bin_file

        self.root_dir = os.path.dirname(os.path.realpath(__file__)) 

        self.SCALE     = 0.5
        self.SPREAD    = 2.0
        self.LOD       = 3.0
        self.THRESHOLD = 0.0
        self.KERNEL    = 9
        self.STRIDE    = 2
        self.NGRID     = 16

        self.best_params = {}
        self.tot_hist=None

        self.fps = 30

        #Bayesian Optimization parameters
        self.n_calls = 10
        self.batch_size = 5

        self.SPACE=[
            skopt.space.Real(0.0, 1.0, name='SCALE', prior='uniform'),
            skopt.space.space.Categorical([0,1,2,3,4,5], name='LOD'),
            skopt.space.space.Categorical([3,5,7,9], name='KERNEL'),
            skopt.space.space.Categorical([1,3,5,7,9], name='STRIDE'),
            skopt.space.space.Categorical([4,8,16], name='NGRID')]

        self.obj_fun = "PSNR"
        self.dataset_dir = None
        self.dataset_file_names = None
        self.tmp_interpolated_frame = "tmp_frame_interp_res.png"

        self.info()
        self.report = "test_res"
        self.eval_psnr = 0
        self.eval_ssim = 0


    def info(self):
        print("=========================");
        print(f"  root dir: {self.root_dir}")
        print(f"  SCALE: {self.SCALE}")
        print(f"  SPREAD: {self.SPREAD}")
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
        self.SPREAD    = params['SPREAD']
        self.LOD       = params['LOD']
        self.THRESHOLD = params['THRESHOLD']
        self.KERNEL    = params['KERNEL']
        self.STRIDE    = params['STRIDE'] 
        self.NGRID     = params['NGRID']



    def setDataset(self, dataset_dir):
        self.dataset_dir = dataset_dir

        self.report = self.dataset_dir.replace("/","_")
        if self.report[-1] == '_':
            self.report = self.report+"report"
        else:
            self.report = self.report+"_report"

        print("========= Dataset ==========");
        print(f"  Dataset: {self.dataset_dir}")
        print("===========================");

        if os.path.isdir(self.report):
            shutil.rmtree(self.report)

        os.mkdir(self.report)

        
        files = os.listdir(self.dataset_dir)
        files = [f for f in files if f.endswith(".png")]
        self.dataset_file_names = natsorted(files)

        

    def create_video(self, img_dir, video_name):
        "Create a video of img_dir"
        files = os.listdir(img_dir)
        files = [f for f in files if f.endswith(".png")]
        natsort_file_names = natsorted(files)
        #print(natsort_file_names)

        frame = cv2.imread(os.path.join(img_dir, files[0]))
        height, width, layers = frame.shape

        #fps = 15
        print(f"FPS of {video_name} is {self.fps}")
        video = cv2.VideoWriter(video_name, 0, self.fps, (width,height))

        for f in tqdm(natsort_file_names, position=0, leave=True):
            #print(f"img: {f}")
            video.write(cv2.imread(os.path.join(img_dir, f)))

        cv2.destroyAllWindows()
        video.release()

    

    @dispatch(str, str, str)
    def interpolate_frame(self, img_1, img_2, interpolated_res) -> None:
        "Interpolate the img_1 and img_2 based on the parameters"
        input1_name = os.path.join(self.root_dir, img_1)
        input2_name = os.path.join(self.root_dir, img_2)
        #print(f"{self.bin_file} --input1 {input1_name} --input2 {input2_name} --scaleFactor {self.SCALE} --spread {self.SPREAD} --lod {self.LOD} --threshold {self.THRESHOLD} --kernel {self.KERNEL} --stride {self.STRIDE} --ngrid {self.NGRID} --out {interpolated_res} ")
        

        os.system(
                "{} --input1 '{}' --input2 '{}' --scaleFactor {} --spread {} --lod {} --threshold {} --kernel {} --stride {} --ngrid {} --out {} {}".format(
                    self.bin_file,
                    input1_name,
                    input2_name,
                    self.SCALE,
                    self.SPREAD,
                    self.LOD,
                    self.THRESHOLD,
                    self.KERNEL,
                    self.STRIDE,
                    self.NGRID,
                    interpolated_res,
                    SUPRESS_MSG
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
        #print(f"output: {output_path}")
        for i in tqdm(range(0,len(natsort_file_names)), position=0, leave=True):
    
            if i%2==1 and i<len(natsort_file_names)-1:
                input1_name = os.path.join(img_dir, natsort_file_names[i-1])
                input2_name = os.path.join(img_dir, natsort_file_names[i+1])
                

                os.system("{} --input1 '{}' --input2 '{}' --scaleFactor {} --spread {} --lod {} --threshold {} --kernel {} --stride {} --ngrid {} --out {} {}".format(
                        self.bin_file,
                        input1_name,
                        input2_name,
                        self.SCALE,
                        self.SPREAD,
                        self.LOD,
                        self.THRESHOLD,
                        self.KERNEL,
                        self.STRIDE,
                        self.NGRID,
                        interpolated_res,
                        SUPRESS_MSG
                    )
                )

    
    
                #print(f"{os.path.join(self.root_dir,natsort_file_names[i])} {output_path}")
                #shutil.copy(os.path.join(self.root_dir,natsort_file_names[i]), output_path)
                shutil.copy(natsort_file_names[i], output_path)
                os.remove(natsort_file_names[i])
            else:
                frame = cv2.imread(os.path.join(img_dir, natsort_file_names[i]))
                cv2.imwrite(os.path.join(output_path, natsort_file_names[i]), frame)


    def psnr_or_ssim(self, x, img_1_path, img_2_path, ground_truth_path, obj_fun='PSNR'):
        "Evaluate the objective function PSRN or SSIM"
        #PARAMS
        SCALE  = 0.5
        SPREAD = x[0]
        LOD    = x[1]
        THRESHOLD = 0.0
        KERNEL = x[2]
        STRIDE = x[3]
        NGRID  = x[4]
    
    
    
        #print(f"{self.bin_file} --input1 '{img_1_path}' --input2 '{img_2_path}' --scaleFactor {SCALE} --spread {SPREAD} --lod {LOD} --threshold {THRESHOLD} --kernel {KERNEL} --stride {STRIDE} --ngrid {NGRID} --out {self.tmp_interpolated_frame} {SUPRESS_MSG}")
        os.system("{} --input1 '{}' --input2 '{}' --scaleFactor {} --spread {} --lod {} --threshold {} --kernel {} --stride {} --ngrid {} --out {} {}".format(
                        self.bin_file,
                        img_1_path,
                        img_2_path,
                        SCALE,
                        SPREAD,
                        LOD,
                        THRESHOLD,
                        KERNEL,
                        STRIDE,
                        NGRID,
                        self.tmp_interpolated_frame,
                        SUPRESS_MSG
                    )
                )

    
        interpolated_img = cv2.imread(self.tmp_interpolated_frame)
        ground_truth_img = cv2.imread(ground_truth_path)
    
    
        #print(interpolated_img.shape)
        #print(ground_truth_img.shape)
    
        #print(type(interpolated_img))
    
    
        if obj_fun == "PSNR":
            return sk_psnr(ground_truth_img, interpolated_img)
        elif obj_fun == "SSIM":
            return sk_ssim(ground_truth_img, interpolated_img, data_range=ground_truth_img.max() - ground_truth_img.min(), channel_axis=2)
    

    def f(self, x):
        "The wrapper of the objective function"
        batch = random.choices(self.dataset_file_names[1:-1], k=self.batch_size)
    
        tot=[]
        for f in batch:
            ground_truth_path = os.path.join(self.dataset_dir, f)
            img_1_path        = os.path.join(self.dataset_dir, self.dataset_file_names[self.dataset_file_names.index(f)-1])
            img_2_path        = os.path.join(self.dataset_dir, self.dataset_file_names[self.dataset_file_names.index(f)+1])
            val               = -self.psnr_or_ssim(x, img_1_path, img_2_path, ground_truth_path, self.obj_fun)
    
            tot.append(val)
    
        batch_val = np.mean(tot)
        print(f"Batch average loss {self.obj_fun}: {-batch_val}")
        return batch_val
    


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
        print(f"    {self.obj_fun}: {-res.fun}")


        self.best_params['SCALE']  = res.x[0]
        self.best_params['LOD']    = res.x[1]
        self.best_params['KERNEL'] = res.x[2]
        self.best_params['STRIDE'] = res.x[3]
        self.best_params['NGRID']  = res.x[4]

        os.remove(self.tmp_interpolated_frame)


    def eval(self):
        assert self.dataset_dir
        assert self.dataset_file_names
        tot_psnr=[]
        tot_ssim=[]
        self.tot_hist=np.zeros(256);
        params = [self.SCALE, self.LOD, self.KERNEL, self.STRIDE, self.NGRID]
        print(f"Start evaluating {self.dataset_dir} with the following parameters")
        self.info()
        for i in tqdm(range(0,len(self.dataset_file_names)), position=0, leave=True):
            if i%2==1 and i<len(self.dataset_file_names)-1:
                ground_truth_path = os.path.join(self.dataset_dir, self.dataset_file_names[i])
                img_1_path        = os.path.join(self.dataset_dir, self.dataset_file_names[i-1])
                img_2_path        = os.path.join(self.dataset_dir, self.dataset_file_names[i+1])
                val_psnr          = -self.psnr_or_ssim(params, img_1_path, img_2_path, ground_truth_path, "PSNR")
                val_ssim          = -self.psnr_or_ssim(params, img_1_path, img_2_path, ground_truth_path, "SSIM")

        
                interpolated_img = cv2.imread(self.tmp_interpolated_frame)
                ground_truth_img = cv2.imread(ground_truth_path)

                abs_diff = abs(ground_truth_img - interpolated_img)
                count, bins_count = np.histogram(abs_diff, bins=256, range=(0,255))
                #print(f"count: {count}    bins_count:{bins_count}")
                self.tot_hist = self.tot_hist + count
                #print(f"hist: {self.tot_hist}")
                

                tot_psnr.append(val_psnr)
                tot_ssim.append(val_ssim)

        self.eval_psnr = -np.mean(tot_psnr)
        self.eval_ssim = -np.mean(tot_ssim)
        print(f"Total images is {len(tot_psnr)}")
        print(f"The average PSRN: {self.eval_psnr}")
        print(f"The average SSIM: {self.eval_ssim}")

    def saveHist(self):
        plt.bar(list(np.arange(0, 256, 1, dtype=int)), self.tot_hist, color ='maroon',  width = 0.4)
        plt.savefig(self.report+".png", dpi=199) 
        shutil.move(self.report+".png", f"{self.report}/")

    def saveBestParams(self): 
        print(f"report name: {self.report}")
        with open(self.report+".txt", 'w') as f:
            print(self.best_params, file=f)
            print(f"Eval PSNR: {self.eval_psnr}", file=f)
        shutil.move(self.report+".txt", f"{self.report}/")
      

    def applyBestPredictedParams(self):
        self.SCALE    = self.best_params['SCALE']  
        self.LOD      = self.best_params['LOD']    
        self.KERNEL   = self.best_params['KERNEL'] 
        self.STRIDE   = self.best_params['STRIDE'] 
        self.NGRID    = self.best_params['NGRID']  



