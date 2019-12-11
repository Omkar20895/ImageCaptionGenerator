"""
This is a utility file to retrieve image features from a pre-trained
Inception V3 Convolutional Neural Network and store them in pickle files. 

The process takes around 1 hour to run on a normal desktop.
"""

import cv2
import glob
import pickle

from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

"""
Reading the images and resizing for CNN
"""
images = []
image_names = []
image_dir = './Flickr_Data/Images/*.jpg'
for img in glob.glob(image_dir):
    x = img.split("/")
    name = x[8].split("\\")
    image_names.append(name[1])
    cv_img = cv2.imread(img)
    img_resize = cv2.resize(cv_img, (299,299))
    images.append(img_resize)

x = pd.DataFrame.from_dict(new_dict,orient='index').reset_index()
x.rename(columns={'index':'image'}, inplace=True)

"""
Creating instance of a pretrained Inception V3 Convolutional Neural Network
to retrieve image features.
"""
model = InceptionV3(weights='imagenet', include_top=True)
model_new = Model(model.input, model.layers[-2].output)

image_dict = {}
for i in range(len(images)):
    x = preprocess_input(images[i])
    x = np.resize(x,(1,299,299,3))
    preds = model_new.predict(x)
    preds = np.reshape(preds, preds.shape[1])
    image_dict[image_names[i]] = preds

"""
Creating seperate dataframes for train, test and validation images
to create pickle files
"""
train_images= {}
test_images = {}
dev_images = {}

for i in range(len(train_df)):
    if train_df.iloc[i][0] in image_dict.keys():
        train_images[train_df.iloc[i][0]] = image_dict.get(train_df.iloc[i][0])

for i in range(len(test_df)):
    if test_df.iloc[i][0] in image_dict.keys():
        test_images[test_df.iloc[i][0]] = image_dict.get(test_df.iloc[i][0])

for i in range(len(dev_df)):
    if dev_df.iloc[i][0] in image_dict.keys():
        dev_images[dev_df.iloc[i][0]] = image_dict.get(dev_df.iloc[i][0])

"""
Storing the image features in respective pickle files
"""

pickle.dump(train_images, open("encoded_train_images.pkl", "wb"))
pickle.dump(test_images, open("encoded_test_images.pkl", "wb"))
pickle.dump(dev_images, open("encoded_dev_images.pkl","wb"))
