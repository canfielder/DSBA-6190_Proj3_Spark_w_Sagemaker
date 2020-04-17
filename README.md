<p float="right">
  <img src="https://user-images.githubusercontent.com/38056064/79609060-1277ae00-80c4-11ea-8c8a-b19308b4dd45.jpg" width="425" />
</p>


# Predicting Spam Messages Using SageMaker Spark
This repo contains the source code for DSBA 6190 Project 3. The goal this was to employ a distributed platform towards a task suited to the platform. I chose the following:

* __Platform__: Spark (via Sagemaker Spark) on Amazon Sagemaker 
* __Task__: Develop a model to identify Spam messages

# Environment
The model development of this project was executed in Amazon SageMaker, using the SageMaker Spark library. The model developed was an XGBoost model, used for the binary classification of spam vs ham messages. The SageMaker Spark contains an XGBoost Estimator equipped for Spark DataFrames. If I was to use a model other than XGBoost, I both have access models in the Pyspark ML library, and SageMaker allows the adaptation of any Sagemaker algorithm to a model which is compatible with Spark.

# Data
## Source
The data for this project was supplied by the Kaggle **SMS Spam Collection Dataset**, found [here](https://www.kaggle.com/uciml/sms-spam-collection-dataset). The original dataset is hosted on the UCI Machine Learning Repository, and was created by Tiago A. Almeida and José María Gómez Hidalgo ([link](http://www.dt.fee.unicamp.br/~tiago//smsspamcollection/)). 

This data contains 5,574 messages, manually tagged ham (legitimate) or as spam. While the size of this data set does not justify the use of Spark, in the real world, this type of problem could generate the scale of data to justify using a Spark platform.

## Import
The data was uploaded to a dedicated S3 bucket directly from Kaggle, via the Kaggle API. To use the official Kaggle API follow the instruction found at the GitHub page, [here](https://github.com/Kaggle/kaggle-api). While straightforward, full usage does require generating a Kaggle Credential  JSON file and placing correctly on your machine.

I used a Cloud9 instance for this step. Once the Kaggle API was installed, and the credential JSON properly located, the Makefile in this repo uses the commands **kaggle_download**, **extract**, and **s3_upload** to properly import the data. 

# Data Processing and Model Deployment
The full modeling process is contained in the notebook **spam_identification.ipynb**.
The notebook performs the following main tasks:

1. Initialize Spark Session (v2.3.2)
2. Preprocess and normalize the text data.
3. Tokenize and perform TF-IDF transformation on the text data via Pyspark pipeline.
4. Train and deploy the XGBoost model on the training data.
5. Evaluate the performance on the model on the test data.
6. Delete the deployed model endpoint.

Further description of the process is contained within the notebook.

# Results
The XGBoost model developed performed well, returning an accuracy of 0.980. Furthermore, the F1 Score was also 0.980, indicating high precision  and recall. The following Confusion Matrix and ROC curve was generated from the test data when evaluating the model.

<p >
  <img src="https://user-images.githubusercontent.com/38056064/79607581-6d5bd600-80c1-11ea-8b28-ced7b1d58f36.png" width="425" />
  <img src="https://user-images.githubusercontent.com/38056064/79607548-603ee700-80c1-11ea-8360-e4e74af92924.png" width="425" /> 
</p>

