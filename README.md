# Email Spam Classification using SVM

This project is an implementation of a Support Vector Machine (SVM) classifier to classify emails as spam or non-spam, implemented in a Jupyter Notebook.

# Overview
The aim of this project is to develop a machine learning model that can accurately classify emails as spam or non-spam using the SVM algorithm. The model is trained on a dataset of labeled emails.

The project consists of the following components:

* Data Loading and Exploration
* Data Preprocessing
* Feature Extraction
* Model Training and Evaluation
* Model Testing

## Requirements
To get started with this project, you will need to install Python and the following libraries:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* nltk
* string

You can install these libraries using pip:

Copy code
```python
pip install pandas numpy scikit-learn matplotlib seaborn nltk string
```

## Dataset
The dataset used in this project is [Spam Email](https://www.kaggle.com/datasets/mfaisalqureshi/spam-email). This dataset contains a collection of labeled emails that can be used for training and testing the classifier.

## Running the Notebook
To run the Jupyter Notebook, open the *Email Spam SVM .ipynb* file in Jupyter Notebook or Jupyter Lab. Follow the instructions in the notebook to execute each cell and run the code.

The notebook consists of the following sections:

## Data loading and exploration
In this section, we load the dataset and explore its contents. We also perform some basic exploratory data analysis to gain insights into the data.

## Data preprocessing
In this section, we preprocess the data to prepare it for training the model. We perform tasks such as removing stop words, stemming, and tokenization.

## Feature extraction
In this section, we extract features from the preprocessed data. We use the Bag of Words approach to represent the emails as vectors of word frequencies.

## Model training and evaluation
In this section, we train the SVM classifier on the training data and evaluate its performance using cross-validation.

## Model testing
In this section, we test the trained model on a test set of emails and evaluate its performance using metrics such as accuracy, precision, and recall.

## Conclusion
This project demonstrates the use of SVM algorithm to classify emails as spam or non-spam. The implementation can be further improved by using more sophisticated techniques such as kernel functions or other machine learning algorithms.
