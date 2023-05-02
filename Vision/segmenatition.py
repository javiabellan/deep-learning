import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

from torchvision import models
import torchvision.transforms as T
import cv2

import numpy as np
import time
from matplotlib import cm
import matplotlib.pyplot as plt

classes=['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Returns a numpy matrix (uint8) of shape [21, 3] (21 RGB colors)
def get_pallete(colormap="hsv", num_colors=21):
    cmap = cm.get_cmap("hsv", 21)
    colors_normalized = cmap( np.arange(0, cmap.N) )[:,:-1]

    return ( colors_normalized * 255 ).astype(np.uint8)


#def print_times(times_tuple):
#    first = times_tuple[0]
#    last  = times_tuple[-1]
#
#    fps  = 1.0 / (last-first)
#    
#
#    for i in range(len(times_tuple)-1):
#        
#        duration = times_tuple[i+1] - times_tuple[i]
#        
#        print(f'{duration:0.4f}\t', end='')
#
#    print(f"FPS: {fps:0.4f}")


class SegModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = models.segmentation.fcn_resnet50(pretrained=True, aux_loss=False).cuda()
        self.ppmean=torch.Tensor([0.485, 0.456, 0.406])
        self.ppstd=torch.Tensor([0.229, 0.224, 0.225])
        self.preprocessor=T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        self.cmap=torch.from_numpy(get_pallete()).cuda()


    def forward(self, x):
        """x is a pytorch tensor"""

        #x=(x-self.ppmean)/self.ppstd #uncomment if you want onnx to include pre-processing
        isize=x.shape[-2:]
        x=self.net.backbone(x)['out']
        x=self.net.classifier(x)
        #x=nn.functional.interpolate(x, isize, mode='bilinear') #uncomment if you want onnx to include interpolation
        return x

    def export_onnx(self, onnxpath):
        """onnxpath: string, path of output onnx file"""

        x=torch.randn(1,3,360,640).cuda() #360p size
        input=['image']
        output=['probabilities']
        torch.onnx.export(self, x, onnxpath, verbose=False, input_names=input, output_names=output, opset_version=11)
        print('Exported to onnx')

    def infer_video(self, fname, view=True, savepath=None):
        """
        fname: path of input video file/camera index
        view(bool): whether or not to display results
        savepath (string or None): if path specified, output video is saved
        """


        ########################## SET VIDEO SOURCE
        with record_function("CREATE_VIDEO"):
            video=cv2.VideoCapture(fname)
            
            ret,frame=video.read()
            if not ret:
                print(f'Cannot read input file/camera {fname}')
                quit()

        ########################## SET VIDEO OUTPUT
        dst=None
        if savepath is not None:
            dst=self.getvideowriter(savepath, video)


        with record_function("PREPARE NET"):
            self.net.eval()

        frames=5

        with torch.no_grad(): # We just inferring, no need to calculate gradients
            while ret:

                with record_function("INFER_FRAME"):
                    out_frame = self.infer_frame(video)
                
                if view:
                    cv2.imshow('segmentation', out_frame)
                    if cv2.waitKey(1)==ord('q'):
                        break
                if dst:
                    dst.write(out_frame)

                frames -= 1
                if frames==0:
                    break

            video.release()
            if dst:
                dst.release()



    def infer_frame(self, video):
        """
        frame: numpy array containing un-pre-processed video frame (dtype is uint8)
        benchamrk: bool, whether or not to calculate inference time
        """

        ################################## 1) READ FRAME
        with record_function("1_read_frame_cpu"):
            ret,frame=video.read()   # 0.0007 secs

        ################################## 2) PROCCESS FRAME (in CPU) 
        with record_function("2_process_frame_cpu"):
            rgb=frame[...,::-1].copy()
            processed=self.preprocessor(rgb)[None] # 0.03 secs        

        ################################## 3) MOVE TO GPU
        with record_function("3_move_to_gpu"):
            processed = processed.cuda() #transfer to GPU <-- does not use zero copy   # 0.003 secs

        ################################## 4) RUN NEURAL NET (in GPU)
        with record_function("4_model_inference_gpu"):
            inferred  = self(processed) #infered.shape = torch.Size([1, 21, 135, 240])  # 0.005 secs

        ################################## 5) POSTPROCESS (in GPU)
        with record_function("5_post_process_gpu"):
            mask_ints = inferred.argmax(dim=1) #mask_ints.shape = torch.Size([1, 135, 240])  # 0 secs   
            mask_rgb = self.cmap[mask_ints] # mask_rgb.shape = [1, 135, 240, 3]  # 0 secs

        ################################## 6) MOVE BACK TO CPU
        with record_function("6_move_to_cpu"):
            mask_rgb = mask_rgb.cpu().numpy() # mask_rgb.shape = [1, 135, 240, 3] # 0.3 secs

        ################################## 7) SHOW IMAGE + SEG_MASK
        with record_function("7_show_img_cpu"):
            mask_rgb = mask_rgb[0,...]                        # mask_rgb.shape = [135, 240, 3]
            mask_rgb = cv2.resize(mask_rgb, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
            overlaid = cv2.addWeighted(frame, 0.7, mask_rgb, 0.3, 0.0)

        return overlaid


    def getvideowriter(self, savepath, srch):
        """
        Simple utility function for getting video writer
        savepath: string, path of output file
        src: a cv2.VideoCapture object
        """
        fps=srch.get(cv2.CAP_PROP_FPS)
        width=int(srch.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(srch.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc=int(srch.get(cv2.CAP_PROP_FOURCC))
        dst=cv2.VideoWriter(savepath, fourcc, fps, (width, height))
        return dst

if __name__=='__main__':

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:

        with record_function("INITIALIZATION"):
            model=SegModel()
            #model.export_onnx('./segmodel.onnx')

        with record_function("INFERENCE_LOOP"):
            model.infer_video(fname=0, view=True, savepath=None)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")

