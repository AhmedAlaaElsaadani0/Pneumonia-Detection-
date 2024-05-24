# %% [markdown]
# # This Assignment was done by :
# ### Mahmoud Labib Elsamadony____20208029
# ### Ahmed Alaa Elsaadani____20208003
# ### Abdelrahman Mezar____20208018
# ### Mohamed Amr Shehab___20208041
# ### Mohamed Ahmed Seif___20208025

# %% [markdown]
# # RSNA Pneumonia Detection
# 1- The dataset provided in two folder:
# - train_images.zip
# - test_images.zip.
# 
# </br>
# 
# 2- train.csv:
# - the training set. Contains patientIds and bounding box / target information.
# - patientId :- A patientId. Each patientId corresponds to a unique image.
# - x :- the upper-left x coordinate of the bounding box.
# - y :- the upper-left y coordinate of the bounding box.
# - width :-the width of the bounding box.
# - height :- the height of the bounding box.
# - Target :- the binary Target, indicating whether this sample has evidence of pneumonia

# %% [markdown]
# ## First we view the data 

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydicom
from PIL import Image
import os
from ultralytics import YOLO

# %%
df = pd.read_csv('dataset/train_labels.csv')
df.head()

# %%
# replace nans with 0
df.fillna(0, inplace=True)
df.head()

# %%
df.describe()

# %%
# Plotting the first image

path_to_image = df['patientId'][0]

# Load the .dcm image
ds = pydicom.dcmread(f'dataset/train_images/{path_to_image}.dcm')

# Access image data
image = ds.pixel_array

# Display the image
plt.imshow(image, cmap='gray')
plt.show()

# %%
# Plotting the first image with bounding box

# Load the .dcm image
path_to_image = df['patientId'][4]
ds = pydicom.dcmread(f'dataset/train_images/{path_to_image}.dcm')

# Access image data
image = ds.pixel_array

# Display the image
plt.imshow(image, cmap='gray')

# Add bounding box

# Get the target values
x = df['x'][4]
y = df['y'][4]
width = df['width'][4]
height = df['height'][4]

# Create a Rectangle patch

rect = plt.Rectangle((x, y), width, height, edgecolor='r', facecolor='none')

# Add the patch to the Axes
plt.gca().add_patch(rect)

plt.show()

# %%
# compare between the number of images with pneumonia and without pneumonia
print(df['Target'].value_counts())
plt.bar(['Non Pneumonia', 'Pneumonia'], df['Target'].value_counts());

# %%
# count the .dcm files in the dataset/train_images folder
print(len(os.listdir('dataset/train_images')))

# %%
# check if any of the patientId in the train_labels.csv file are in the dataset/test_images folder
test_images = os.listdir('dataset/test_images')

for patientId in df['patientId']:
    if f'{patientId}.dcm' in test_images:
        print(patientId)

# %%
df.shape

# %%
# count the number of unique patients
df['patientId'].nunique()

# %%
print(len(os.listdir('dataset/test_images')))

# %% [markdown]
# - Nan values are replaced with zeros
# - number of images with non-pneumonia are double the images with pneumonia (imbalanced data)
# - number of unique patients in the dataframe is equal to the number of training images (no redundancy)

# %% [markdown]
# ## Convert the .dcm images to a format that YOLOv8 can work with
# 
# - First, we extract the image data from .dcm file
# - Second, we will Normalize the pixel values to be from 0 - 255 (leading to faster convergence)
# - we then convert it to jpg format

# %%
# convert the .dcm files to .jpg files

def convert_dcm_to_jpg(input_folder, output_folder):
    for file in os.listdir(input_folder):
        if os.path.isfile(f'{output_folder}/{file.split(".")[0]}.jpg'):
            continue
        ds = pydicom.dcmread(f'{input_folder}/{file}')
        img = ds.pixel_array
        # Normalize the pixel values to the range [0, 255]
        image_normalized = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        image_normalized = image_normalized.astype(np.uint8)  # Convert to unsigned 8-bit integer type

        # Convert the numpy array to a PIL image
        image = Image.fromarray(image_normalized)

        filename = file.split('.')[0]
        image.save(f'{output_folder}/{filename}.jpg')

# %%
convert_dcm_to_jpg('dataset/train_images', 'dataset/images/train')

# %%
convert_dcm_to_jpg('dataset/test_images', 'dataset/images/test')

# %% [markdown]
# ## Normalizing the labels (YOLO Format)

# %%
def normalize_bboxs(df):
    img_size = 1024 
    
    # normalize the bounding boxes
    df['width'] = df['width'] / img_size
    df['height'] = df['height'] / img_size
    # Centering X , Y coordinates and normalize them
    df['x'] = df['x'] / img_size + df['width'] / 2
    df['y'] = df['y'] / img_size + df['height'] / 2

    
def denormalize_bbox(rx, ry, rw, rh):
    img_size = 1024 
    
    x = (rx-rw/2)*img_size
    y = (ry-rh/2)*img_size
    w = rw*img_size
    h = rh*img_size
    
    return x, y, w, h

# %%
normalize_bboxs(df)

df.head()

# %% [markdown]
# ## Preparing The labels folder

# %%
# Converting the df to .txt files per patientId each patientId in separate file (as there might be multiple bounding boxes per patientId)

def convert_df_to_txt_files(df, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for patientId in df['patientId'].unique():
        patient_df = df[df['patientId'] == patientId].copy()
        if patient_df['Target'].values[0] == 0:
            # if the patient doesn't have pneumonia, output empty txt file
            with open(f'{output_folder}/{patientId}.txt', 'w') as f:
                f.close()
            continue
        patient_df.drop(columns='patientId', inplace=True)
        # getting the target column to the first column
        cols = patient_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        patient_df = patient_df[cols]
        patient_df['Target'] = 0
        patient_df.to_csv(f'{output_folder}/{patientId}.txt', index=False, header=False, sep=' ')


# %%
convert_df_to_txt_files(df, 'dataset/labels/train')

# %% [markdown]
# ## Splitting the data into train_1 , train_2 , train_3 and validation (Hardware constraints)

# %%
## split the data into train_1 , train_2 , train_3 and validation sets and without shuffling the patientIds

# get the unique patientIds
patientIds = df['patientId'].unique()

train_1 = patientIds[:9000]
train_2 = patientIds[9000:18000]
train_3 = patientIds[18000:24000]
validation = patientIds[24000:]

# get the patientIds for each set
train_1_df = df[df['patientId'].isin(train_1)].copy()
train_2_df = df[df['patientId'].isin(train_2)].copy()
train_3_df = df[df['patientId'].isin(train_3)].copy()
validation_df = df[df['patientId'].isin(validation)].copy()


# convert the train_1_df, train_2_df, train_3_df and validation_df to .txt files
# convert_df_to_txt_files(train_1_df, 'dataset/labels/train_1')
# convert_df_to_txt_files(train_2_df, 'dataset/labels/train_2')
# convert_df_to_txt_files(train_3_df, 'dataset/labels/train_3')
# convert_df_to_txt_files(validation_df, 'dataset/labels/validation')

# copy the images to the respective folders
def copy_images_to_folder(df, input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for patientId in df['patientId'].unique():
        os.system(f'cp {input_folder}/{patientId}.jpg {output_folder}/{patientId}.jpg')

# copy_images_to_folder(train_1_df, 'dataset/images/train', 'dataset/images/train_1')
# copy_images_to_folder(train_2_df, 'dataset/images/train', 'dataset/images/train_2')
# copy_images_to_folder(train_3_df, 'dataset/images/train', 'dataset/images/train_3')
# copy_images_to_folder(validation_df, 'dataset/images/train', 'dataset/images/validation')



# %% [markdown]
# ## Training The YOLOv8 Algorithm 
# 
# - we ran each part `(train_1 , train_2 , train_3)` of the data for two epochs to achieve better results 
# - at the end of each part we get the `best.pt` from the previous one for the next training session
# - that lead to better losses , precision and recall

# %%
model = YOLO("dataset/best3.pt")

res = model.train(data="data.yaml", epochs=1) # one final epoch to finish training

# %% [markdown]
# ## Predicting Bounding Boxes

# %%
model = YOLO("best3.pt")
results = model('dataset/images/test',conf=0.1,stream=True)

def get_id_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

for result in results:
    id = get_id_from_path(result.path)
    with open(f'dataset/predictions/{id}.txt', 'w') as f:
        if len(result.xywh) == 0:
            f.write(f'{id}\n')
        else:
            with open(f'dataset/predictions/{id}.txt', 'w') as f:
                
                x, y, w, h = denormalize_bbox(*result.xywh.tolist())
                confidence = result.conf
                f.write(f'{id} {confidence} {x} {y} {w} {h} \n')



