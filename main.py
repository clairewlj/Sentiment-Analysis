#This project is based on movie reviews on IMDb.
import urllib.request
import pandas as pd

#define URLs
test_data_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/testdata.txt"
train_data_url = "https://dl.dropboxusercontent.com/u/8082731/datasets/UMICH-SI650/training.txt"

#define local file names
test_data_file_name = 'test_data.csv'
train_data_file_name = 'train_data.csv'

#download files using urlib
test_data_f = urllib.request.urlretrieve(test_data_url,test_data_file_name)
train_data_f = urllib.request.urlretrieve(train_data_url,train_data_file_name)

#read and load files into data frames for processing
test_data_df = pd.read_csv(test_data_file_name,header=None,delimiter="\t",quoting=3)
test_data_df.columns = ['Text']
train_data_df = pd.read_csv(train_data_file_name,header=None,delimiter='\t',quoting=3)
train_data_df.columns = ['Sentiment','Text']

train_data_df.head()
