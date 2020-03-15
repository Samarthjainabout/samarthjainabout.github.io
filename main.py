from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image
from resizeimage import resizeimage
from skimage import color
from skimage import io
import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
from array import array
import statistics
from splitter import *

for filename in glob.glob('data/membrane/train/label/*.png'): #assuming gif
    
    #cover.save(filename, im.format)
    
    
    im = cv2.imread(filename)
    ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
    cv2.imwrite(filename, thresh1)
    
for filename in glob.glob('data/membrane/train/image/*.png'): #assuming gif
    
    #cover.save(filename, im.format)
    
    
    im = cv2.imread(filename,0)
    im = cv2.equalizeHist(im)
    
    cv2.imwrite(filename, im)
    
for filename in glob.glob('data/membrane/test/*.png'): #assuming gif
    
    #cover.save(filename, im.format)
    
    
    im = cv2.imread(filename,0)
    im = cv2.equalizeHist(im)
    
    cv2.imwrite(filename, im)

"""upper is for contrast enhancement of images"""

data_gen_args = dict(rotation_range=0.6,
                    width_shift_range=0.07,
                    height_shift_range=0.07,
                    shear_range=0.09,
                    zoom_range=0.07,
                    horizontal_flip=True,
                    fill_mode='nearest')

target_size=(1024,1024)
myGene = trainGenerator(1,'data/membrane/train','image','label',data_gen_args,save_to_dir = 'data/membrane/train/aug',target_size=target_size)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=10000,epochs=4 ,callbacks=[model_checkpoint])

#predict using stored model
model.load_weights("unet_membrane.hdf5")
testGene = testGenerator("data/membrane/test",target_size=target_size)
results = model.predict_generator(testGene,23,verbose=1)
saveResult("data/membrane/test",results)

#black and white all predicted values
for filename in glob.glob('data/membrane/test/*_predict.png'): #assuming gif
    
    #cover.save(filename, im.format)
    
    
    im = cv2.imread(filename)
    ret,thresh1 = cv2.threshold(im,127,255,cv2.THRESH_BINARY)
    cv2.imwrite(filename, thresh1)
    
#measure lenght of path image    
path="data/membrane/test/6"
left=array("i")
right=array("i")
image_in=cv2.imread(path+"_predict.png")
image_in=cv2.cvtColor(image_in,cv2.COLOR_BGR2GRAY)

cv2.imshow('image',image_in)
cv2.waitKey(0)
cv2.destroyWindow('image')
for i in range(image_in.shape[0]):
    counter=0
    counter2=0
    for j in range(image_in.shape[1]):
        if image_in[i,j] < 100:
            if j>(image_in.shape[1])*.5 and j<(image_in.shape[1])*.75:
                counter2 += 1#right pillar
                
            elif j<(image_in.shape[1])*.5 and j>(image_in.shape[1])*.25:
                
                counter += 1#left pillar
    
    right.append(counter2)
    left.append(counter)
    
elements = np.array(right)

mean = np.mean(elements, axis=0)
sd = np.std(elements, axis=0)

final_list_right = [x for x in right if (x > mean - 2 * sd)]
final_list_right = [x for x in final_list_right if (x < mean + 2 * sd)]

elements = np.array(left)

mean = np.mean(elements, axis=0)
sd = np.std(elements, axis=0)

final_list_left = [x for x in left if (x > mean - 2 * sd)]
final_list_left = [x for x in final_list_left if (x < mean + 2 * sd)]

#print(final_list_left,final_list_right)

print(np.mean(final_list_left)*.5,np.mean(final_list_right)*.5)

#display visual measurements
disp(path,target_size)

    
