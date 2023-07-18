# Forged-Genuine-Document-Detectors
# OCR and Receipt Verification Model

This repository contains code for an OCR (Optical Character Recognition) and receipt verification model. The model is trained to extract text from receipts and verify their authenticity.

## Introduction

The OCR and receipt verification model is designed to automate the process of extracting information from receipts and determining whether they are genuine or forged. It combines computer vision techniques, deep learning, and fraud detection algorithms to provide accurate results.

## Features

- Optical Character Recognition (OCR): The model utilizes OCR techniques to extract text from receipt images, enabling automatic data extraction.
- Receipt Verification: The model includes a fraud detection component that analyzes various features of the receipt to determine its authenticity.
- Signature Detection: The model can detect signatures on receipts, allowing for signature verification and fraud detection.
- Metadata Extraction: The model extracts metadata from receipts, such as creation date, modification date, and other relevant information.

## Dataset

The dataset used for training and testing the model consists of two classes: "genuine" and "forged". The "genuine" class contains images of genuine receipts, while the "forged" class contains manipulated/forged receipts.

The dataset directory structure is as follows:


Receipts/
├── genuine/
│ ├── image1.jpg
│ ├── image2.jpg
│ └── ...
└── forged/
├── image1.jpg
├── image2.jpg
└── ...



## Requirements

To run the code in this repository, you need the following dependencies:

- Python 3
- TensorFlow
- Keras
- OpenCV
- pytesseract
- scikit-image
- NumPy
- pandas
- Matplotlib



## Usage

1. Clone this repository:

2. Install the dependencies:


3. Prepare the dataset:

   - Place your receipt images in the appropriate directories (`genuine` and `forged`) inside the `Receipts` folder.

4. Preprocess the dataset:

   - Run the `preprocess.py` script to preprocess the images and extract relevant metadata.

5. Train the model:

   - Run the `train_model.py` script to train the OCR and receipt verification model.

6. Evaluate the model:

   - Run the `evaluate_model.py` script to evaluate the model's performance on the test set.

7. Make predictions:

   - Run the `predict.py` script to make predictions on new receipt images.

## Results

The model achieved an accuracy of X% on the test set and successfully detected X% of the forged receipts.

## License

This project is licensed under the [MIT License](LICENSE).



