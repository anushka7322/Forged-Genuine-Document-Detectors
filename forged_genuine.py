#!/usr/bin/env python
# coding: utf-8

# In[753]:


import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# In[757]:


import numpy as np
import pandas as pd
import cv2
import pytesseract 
from PIL import Image


# In[758]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


# # Step 1. Data Collection
# 

# In[759]:


import os
import cv2


# In[760]:


dataset_dir = 'Receipts'


# In[761]:


dataset_dir


# In[762]:


abc = os.listdir(dataset_dir)


# In[763]:


abc


# In[764]:


def load_data():
  images = []
  labels = []

  # Iterating through the sub-directories
  for sub_dir in os.listdir(dataset_dir):
      sub_dir_path = os.path.join(dataset_dir, sub_dir)
#       abc.append(sub_dir_path)

      #check if the sub-directory is a valid label (genuine or forged)
      if(os.path.isdir(sub_dir_path)):
        # Load images from sub-directory
        for image_file in os.listdir(sub_dir_path):
          image_path = os.path.join(sub_dir_path, image_file)
#           print(image_path)

          # Read the image and store it in the 'images' list
          image = cv2.imread(image_path)
          images.append(image)

          # Determine the label based on the sub-directory
          if sub_dir == 'genuine':
            labels.append(0)    # Assign label 0 for genuine images
          elif sub_dir == 'forged':
            labels.append(1)    # Assign label 1 for manipulated images

  return images, labels

  # Call the function to load the data
images, labels = load_data()

  #Print the number of images and their corresponding labels
print("Total images:", len(images))
print("Total labels:", len(labels))


# In[765]:


len(images)


# In[185]:


# image_path = os.path.join(dataset_dir, forged)


# # Step 2. Data Preprocessing
# 

# In[766]:


import tempfile


# In[790]:


#preprocess the images(e.g.. resize, convert to standardized format)
def preprocess_images(images):
  processed_images =[]
  for image in images:

    # Preprocess the image (e.g., resize, convert to grayscale)
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Convert the processed image to a PIL image object
        pil_image = Image.fromarray(processed_image)
        processed_images.append(pil_image)
  return processed_images

#call the function to preprocess the images
processed_images = preprocess_images(images)

#extract relevant metadata from the images
def extract_metadata(images):
    metadata = []
    for image in images:
        image_metadata = {}
#         print(os.path.join(dataset_dir, os.listdir(dataset_dir)[0]))
        
        # Save the image path in the metadata
        for i in range(len(os.listdir(dataset_dir))):
            image_metadata['image_path'] = os.path.join(dataset_dir, os.listdir(dataset_dir)[i-1])
        
        # Save the image numpy array to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_filename = tmp_file.name
            Image.fromarray(image).save(tmp_filename, format='JPEG')
        
        # Open the image using PIL
        pil_image = Image.open(tmp_filename)
        
        # Extract EXIF metadata
        exif_data = pil_image._getexif()
        
        # Check if EXIF data exists
        if exif_data is not None:
            for tag, value in exif_data.items():
                tag_name = Image.TAGS.get(tag, tag)
                image_metadata[tag_name] = value
        
        # Extract other metadata fields
        image_metadata['Filepath'] = tmp_filename  # Example: Store the filepath
        
        # Extract creation date and modification date
        image_stats = os.stat(tmp_filename)
        creation_date = image_stats.st_ctime
        modification_date = image_stats.st_mtime
        image_metadata['CreationDate'] = creation_date
        image_metadata['ModificationDate'] = modification_date
        
        # Append the image metadata to the list
        metadata.append(image_metadata)
    return metadata


#call the function to extract metadata from the images
metadata = extract_metadata(images)


# In[791]:


processed_images


# In[792]:


metadata


# # Step 3. Perform OCR

# In[793]:


# Perform OCR to extract text from the images using pytesseract
def perform_ocr(images):
    extracted_text = []
    for image in images:
        # Convert the image to PIL image object
#         pil_image = Image.fromarray(image.astype(np.uint8)).convert('L')
        
        
        # Use pytesseract to perform OCR on each image and extract the text
        text = pytesseract.image_to_string(image)
        extracted_text.append(text)
    return extracted_text

#call the function to perform ocr on the images
extracted_text = perform_ocr(processed_images)


# In[ ]:


print(extracted_text[1])


# # Step 3.2 Detect Signature and Stamp
# 

# In[795]:


import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
import numpy as np


# In[871]:


def detect_signature(image_path):
    # parameter to remove small size connected pixels
    constant_parameter_1 = 84
    constant_parameter_2 = 250
    constant_parameter_3 = 100
    
    # parameter to remove big size connected pixels
    constant_parameter_4 = 18
    
    for filename in os.listdir(image_path):
        image_path = os.path.join(filename)
    
        # Read input image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold_value, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Connected components analysis by scikit-learn framework
        blobs = thresholded > thresholded.mean()
        blobs_labels = measure.label(blobs, background=1)

        # Calculate the area of the text
        total_area = np.sum(blobs_labels > 0)
        average = total_area / np.max(blobs_labels)

        # Experimental-based ratio calculation to determine presence of a signature
        a4_small_size_outliar_constant = ((average / constant_parameter_1) * constant_parameter_2) + constant_parameter_3
        a4_big_size_outliar_constant = a4_small_size_outliar_constant * constant_parameter_4

        # Remove connected pixels smaller than a4_small_size_outliar_constant
        pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)

        # Remove connected pixels bigger than a4_big_size_outliar_constant
        component_sizes = np.bincount(pre_version.ravel())
        too_small = component_sizes > a4_big_size_outliar_constant
        too_small_mask = too_small[pre_version]
        pre_version[too_small_mask] = 0

        # Determine if a signature is present based on the connected components
        has_signature = np.max(pre_version) > 0

        return has_signature


# # Step 4. Feature Extraction

# In[898]:


# Define a function to extract features
def extract_features(metadata, extracted_text):
    features = []
    
    
    c_date = []
    m_date = []
    text = []
    sign = []
    for i in range(len(metadata)):
        image_features = []
        
        
        
        
        # Extract features from metadata, extracted text, signatures, and stamps
        # Append the features to the 'image_features' list
        
        # Extract features from metadata
        creation_date = metadata[i]['CreationDate']
        modification_date = metadata[i]['ModificationDate']
        print('iiii:',modification_date)
        # Add metadata features to the 'image_features' list
        c_date.append(creation_date)
        m_date.append(modification_date)
        
        # Extract features from extracted text
        text_length = len(extracted_text[i])
        # Add text-related features to the 'image_features' list
        text.append(text_length)
        print('text', text_length)
        
        
        print('path:::::', metadata[i]['image_path'])
        # Extract features from signatures and stamps
        has_signature = detect_signature(metadata[i]['image_path'])
        print('signature:::', has_signature)
#         has_stamp = detect_stamp(metadata[i]['image_path'])
        # Add signature-related and stamp-related features to the 'image_features' list
        sign.append(int(has_signature))
#         image_features.append(int(has_stamp))
        
        
#         features.append(image_features)
#     return features
    return c_date, m_date, text, sign
        


# Call the function to extract features
# features = extract_features(metadata, extracted_text)

c_date, m_date, text, sign = extract_features(metadata, extracted_text)

# Convert the features and labels to NumPy arrays
features = np.array(features)
labels = np.array(labels)

c_date = np.array(c_date)
m_date = np.array(m_date)
text = np.array(text)
sign = np.array(sign)


# In[899]:


# Convert the features and labels to NumPy arrays
c_date = np.array(c_date)
m_date = np.array(m_date)
text = np.array(text)
sign = np.array(sign)


# In[883]:


feature1_tensor = tf.convert_to_tensor(c_date)
feature2_tensor = tf.convert_to_tensor(m_date)
feature3_tensor = tf.convert_to_tensor(text)
feature4_tensor = tf.convert_to_tensor(sign)


# In[900]:


combined_features = np.column_stack((c_date, m_date, text, sign))


# In[901]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)


# # Step 5. Building Neural Network

# In[908]:


from keras.models import Sequential
from keras.layers import Dense, Dropout

# Create a sequential model
model = Sequential()

# Add input layer
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))

# Add hidden layers
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))

# Add output layer
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[849]:


# Create a sequential model
model = Sequential()

# Add Convolution layer 1
model.add(Conv2D(12,(3,3), activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D((2,2)))

# Add Convolution layer 2
model.add(Conv2D(24, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Add flatten layer
model.add(Flatten())

# Add Dense layer 
model.add(Dense(24, activation='relu'))

# Add output layer
model.add(Dense(1, activation='sigmoid'))


# In[857]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[909]:


model.summary()


# In[ ]:


# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_test, y_test))

