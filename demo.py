#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[2]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

###################
#import tensorflow
#import keras
#config = tensorflow.ConfigProto() 
#config.inter_op_parallelism_threads = 1 
#keras.backend.set_session(tensorflow.Session(config=config))


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

os.environ["CUDA_VISIBLE_DEVICES"]="[2,3]"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon_0254.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# In[3]:


from samples.balloon.balloon import BalloonConfig

class InferenceConfig(BalloonConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 2
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.0
    DETECTION_MAX_INSTANCES = 300
    DETECTION_NMS_THRESHOLD = 1.0
    POST_NMS_ROIS_INFERENCE = 300
config = InferenceConfig()
config.display()


# In[4]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# In[5]:


class_names = ['BG', 'Basketball', 'BasketballDunk', 'Biking', 'CliffDiving', 'CricketBowling',
               'Diving', 'Fencing', 'FloorGymnastics', 'GolfSwing', 'HorseRiding',
               'IceDancing', 'LongJump', 'PoleVault', 'RopeClimbing', 'SalsaSpin',
               'SkateBoarding', 'Skiing', 'Skijet', 'SoccerJuggling', 'Surfing', 'TennisSwing', 'TrampolineJumping',
               'VolleyballSpiking', 'WalkingWithDog']


# In[ ]:


import numpy as np
import scipy.io

kosongan = np.zeros((100,100), dtype=np.float64)


# In[ ]:


scipy.io.savemat('out.mat', mdict={'a': kosongan})


# In[ ]:


import scipy.io
mat = scipy.io.loadmat('testlist01_video_list.mat')


# In[ ]:


mat['video']


# In[ ]:


classes = mat['video']['class'][0]
classes[100][0]


# In[ ]:


classess = mat['video']['name'][0]
classess[100][0]


# In[ ]:


folder_name = mat['video']['class'][0]
class_name = mat['video']['name'][0]


# In[ ]:


import glob
from pathlib import Path
import numpy as np
import scipy.io

mat = scipy.io.loadmat('testlist01_video_list.mat')
folder_name = mat['video']['class'][0]
class_name = mat['video']['name'][0]

main_path = '/media/ee401_2/SPML/Danny/Mask_RCNN-master/UCF101/images/'
image2 = skimage.io.imread(os.path.join(IMAGE_DIR, 'ski.jpg'))
main_path_save = '/media/ee401_2/SPML/Danny/Mask_RCNN-master/UCF101/label_images/'

for i2 in range(0,914):
    


    all_header_files = glob.glob(main_path+'/'+folder_name[i2][0]+'/'+class_name[i2][0]+'/*.jpg')
    
    #print(all_header_files)
    
    list_gambar = len(all_header_files)
    j=0
    
    
    for j in range(0,len(all_header_files)):
        
        kosongan = np.zeros((300,100), dtype=np.float64)
        sekorrr = np.zeros((300,25), dtype=np.float64)
        flag3 = 0
        flag4 = 0
        flag5 = 0
        flag6 = 0
        flag7 = 0
        flag8 = 0
        flag9 = 0
        flag10 = 0
        flag11 = 0
        flag12 = 0
        flag13 = 0
        flag14 = 0
        flag15 = 0
        flag16 = 0
        flag17 = 0
        flag18 = 0
        flag19 = 0
        flag20 = 0
        flag21 = 0
        flag22 = 0
        flag23 = 0
        flag24 = 0
        flag25 = 0
        flag26 = 0

        #print(all_header_files[j])
        image1 = skimage.io.imread(all_header_files[j])
            
        image=([image1, image2])

        results = model.detect(image, verbose=1)

        r = results[0]
        i=0

        detected = r['class_ids'].size
        #print(detected)
        for i in range(0,detected):
            bbox = r['rois'][i]
            kelas = r['class_ids'][i]
            skor = r['scores'][i]

            if r['class_ids'][i] == 1 :
                # Kolom 3
                kosongan[flag3][4] = bbox[0]
                kosongan[flag3][5] = bbox[1]
                kosongan[flag3][6] = bbox[2]
                kosongan[flag3][7] = bbox[3]
                sekorrr[flag3][1] = skor
                flag3 += 1
            elif r['class_ids'][i] == 2 :
                # Kolom 2
                kosongan[flag4][8] = bbox[0]
                kosongan[flag4][9] = bbox[1]
                kosongan[flag4][10] = bbox[2]
                kosongan[flag4][11] = bbox[3]
                sekorrr[flag4][2] = skor
                flag4 += 1
            elif r['class_ids'][i] == 3 :
                # Kolom 2
                kosongan[flag5][12] = bbox[0]
                kosongan[flag5][13] = bbox[1]
                kosongan[flag5][14] = bbox[2]
                kosongan[flag5][15] = bbox[3]
                sekorrr[flag5][3] = skor
                flag5 += 1
            elif r['class_ids'][i] == 4 :
                # Kolom 2
                kosongan[flag6][16] = bbox[0]
                kosongan[flag6][17] = bbox[1]
                kosongan[flag6][18] = bbox[2]
                kosongan[flag6][19] = bbox[3]
                sekorrr[flag6][4] = skor
                flag6 += 1
            elif r['class_ids'][i] == 5 :
                # Kolom 2
                kosongan[flag7][20] = bbox[0]
                kosongan[flag7][21] = bbox[1]
                kosongan[flag7][22] = bbox[2]
                kosongan[flag7][23] = bbox[3]
                sekorrr[flag7][5] = skor
                flag7 += 1
            elif r['class_ids'][i] == 6 :
                # Kolom 2
                kosongan[flag8][24] = bbox[0]
                kosongan[flag8][25] = bbox[1]
                kosongan[flag8][26] = bbox[2]
                kosongan[flag8][27] = bbox[3]
                sekorrr[flag8][6] = skor
                flag8 += 1
            elif r['class_ids'][i] == 7 :
                # Kolom 2
                kosongan[flag9][28] = bbox[0]
                kosongan[flag9][29] = bbox[1]
                kosongan[flag9][30] = bbox[2]
                kosongan[flag9][31] = bbox[3]
                sekorrr[flag9][7] = skor
                flag9 += 1
            elif r['class_ids'][i] == 8 :
                # Kolom 2
                kosongan[flag10][32] = bbox[0]
                kosongan[flag10][33] = bbox[1]
                kosongan[flag10][34] = bbox[2]
                kosongan[flag10][35] = bbox[3]
                sekorrr[flag10][8] = skor
                flag10 += 1
            elif r['class_ids'][i] == 9 :
                # Kolom 2
                kosongan[flag11][36] = bbox[0]
                kosongan[flag11][37] = bbox[1]
                kosongan[flag11][38] = bbox[2]
                kosongan[flag11][39] = bbox[3]
                sekorrr[flag11][9] = skor
                flag11 += 1
            elif r['class_ids'][i] == 10 :
                # Kolom 2
                kosongan[flag12][40] = bbox[0]
                kosongan[flag12][41] = bbox[1]
                kosongan[flag12][42] = bbox[2]
                kosongan[flag12][43] = bbox[3]
                sekorrr[flag12][10] = skor
                flag12 += 1
            elif r['class_ids'][i] == 11 :
                # Kolom 2
                print('TUYULLLLLLLLLLLLLL')
                kosongan[flag13][44] = bbox[0]
                kosongan[flag13][45] = bbox[1]
                kosongan[flag13][46] = bbox[2]
                kosongan[flag13][47] = bbox[3]
                sekorrr[flag13][11] = skor
                flag13 += 1
            elif r['class_ids'][i] == 12 :
                # Kolom 2
                kosongan[flag14][48] = bbox[0]
                kosongan[flag14][49] = bbox[1]
                kosongan[flag14][50] = bbox[2]
                kosongan[flag14][51] = bbox[3]
                sekorrr[flag14][12] = skor
                flag14 += 1
            elif r['class_ids'][i] == 13 :
                # Kolom 2
                kosongan[flag15][52] = bbox[0]
                kosongan[flag15][53] = bbox[1]
                kosongan[flag15][54] = bbox[2]
                kosongan[flag15][55] = bbox[3]
                sekorrr[flag15][13] = skor
                flag15 += 1
            elif r['class_ids'][i] == 14 :
                # Kolom 2
                kosongan[flag16][56] = bbox[0]
                kosongan[flag16][57] = bbox[1]
                kosongan[flag16][58] = bbox[2]
                kosongan[flag16][59] = bbox[3]
                sekorrr[flag16][14] = skor
                flag16 += 1
            elif r['class_ids'][i] == 15 :
                # Kolom 2
                kosongan[flag17][60] = bbox[0]
                kosongan[flag17][61] = bbox[1]
                kosongan[flag17][62] = bbox[2]
                kosongan[flag17][63] = bbox[3]
                sekorrr[flag17][15] = skor
                flag17 += 1
            elif r['class_ids'][i] == 16 :
                # Kolom 2
                kosongan[flag18][64] = bbox[0]
                kosongan[flag18][65] = bbox[1]
                kosongan[flag18][66] = bbox[2]
                kosongan[flag18][67] = bbox[3]
                sekorrr[flag18][16] = skor
                flag18 += 1
            elif r['class_ids'][i] == 17 :
                # Kolom 2
                kosongan[flag19][68] = bbox[0]
                kosongan[flag19][69] = bbox[1]
                kosongan[flag19][70] = bbox[2]
                kosongan[flag19][71] = bbox[3]
                sekorrr[flag19][17] = skor
                flag19 += 1
            elif r['class_ids'][i] == 18 :
                # Kolom 2
                kosongan[flag20][72] = bbox[0]
                kosongan[flag20][73] = bbox[1]
                kosongan[flag20][74] = bbox[2]
                kosongan[flag20][75] = bbox[3]
                sekorrr[flag20][18] = skor
                flag20 += 1
            elif r['class_ids'][i] == 19 :
                # Kolom 2
                kosongan[flag21][76] = bbox[0]
                kosongan[flag21][77] = bbox[1]
                kosongan[flag21][78] = bbox[2]
                kosongan[flag21][79] = bbox[3]
                sekorrr[flag21][19] = skor
                flag21 += 1
            elif r['class_ids'][i] == 20 :
                # Kolom 2
                kosongan[flag22][80] = bbox[0]
                kosongan[flag22][81] = bbox[1]
                kosongan[flag22][82] = bbox[2]
                kosongan[flag22][83] = bbox[3]
                sekorrr[flag22][20] = skor
                flag22 += 1
            elif r['class_ids'][i] == 21 :
                # Kolom 2
                kosongan[flag23][84] = bbox[0]
                kosongan[flag23][85] = bbox[1]
                kosongan[flag23][86] = bbox[2]
                kosongan[flag23][87] = bbox[3]
                sekorrr[flag23][21] = skor
                flag23 += 1
            elif r['class_ids'][i] == 22 :
                # Kolom 2
                kosongan[flag24][88] = bbox[0]
                kosongan[flag24][89] = bbox[1]
                kosongan[flag24][90] = bbox[2]
                kosongan[flag24][91] = bbox[3]
                sekorrr[flag24][22] = skor
                flag24 += 1
            elif r['class_ids'][i] == 23 :
                # Kolom 2
                kosongan[flag25][92] = bbox[0]
                kosongan[flag25][93] = bbox[1]
                kosongan[flag25][94] = bbox[2]
                kosongan[flag25][95] = bbox[3]
                sekorrr[flag25][23] = skor
                flag25 += 1
            elif r['class_ids'][i] == 24 :
                # Kolom 2
                kosongan[flag26][96] = bbox[0]
                kosongan[flag26][97] = bbox[1]
                kosongan[flag26][98] = bbox[2]
                kosongan[flag26][99] = bbox[3]
                sekorrr[flag26][24] = skor
                flag26 += 1

        print('Ke = ', i2)

        if not os.path.exists(main_path_save):
            os.makedirs(main_path_save)
        if not os.path.exists(main_path_save+'/'+folder_name[i2][0]):
            os.makedirs(main_path_save+'/'+folder_name[i2][0])
        if not os.path.exists(main_path_save+'/'+folder_name[i2][0]+'/'+class_name[i2][0]):
            os.makedirs(main_path_save+'/'+folder_name[i2][0]+'/'+class_name[i2][0])

        save_mat_path = main_path_save+'/'+folder_name[i2][0]+'/'+class_name[i2][0]+'/'+all_header_files[j][-9:-4]
        print('aaaaaaaaa = ', save_mat_path)
        

        scipy.io.savemat(save_mat_path +'.mat', mdict={'boxes': kosongan, 'scores': sekorrr})


# In[ ]:




