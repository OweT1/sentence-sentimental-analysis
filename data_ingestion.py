import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Setting of variables
seed = 42
train_val_test_split = [0.75, 0.10, 0.15]
train_split = train_val_test_split[0]
val_split = train_val_test_split[1]
test_split = train_val_test_split[2]

# Reading in of data
data = pd.read_csv("data/dataset.csv", header = None, names = ["Label", "Text"])
data = data[['Text', 'Label']]

# Splitting of Data - We will split the data into 75% training, 10% validation and 15% testing.
training_data, testing_data = train_test_split(data, test_size=test_split, random_state=seed, stratify=data['Label'])
training_data, validation_data = train_test_split(training_data, test_size=val_split/(1-test_split), random_state=seed, stratify=training_data['Label'])

training_data.reset_index(drop=True).to_csv('data/training_data.csv', index=False)
testing_data.reset_index(drop=True).to_csv('data/testing_data.csv', index=False)
validation_data.reset_index(drop=True).to_csv('data/validation_data.csv', index=False)