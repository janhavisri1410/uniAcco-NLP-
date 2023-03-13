# Import Libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import torch
import re
import os
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support as score
import matplotlib.pyplot as plt

# Importing the required Transformer models
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, logging

# Disable from connecting to Weights and Biases and removing the Warnings shown
os.environ["WANDB_DISABLED"] = "true"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
logging.set_verbosity_warning()

df_train = pd.read_csv('train.csv', header=0, names=['labels', 'title', 'description'])
df_test = pd.read_csv('test.csv', header=0, names= ['labels', 'title', 'description'])

# drop any null values
df_train = df_train.dropna()
df_test = df_test.dropna()

# Combine Title and Description Column
df_train['text'] = df_train['title'] + " " + df_train['description']
df_test['text'] = df_test['title'] + " " + df_test['description']

# we have new combined text column hence we drop the title n Description column
df_train = df_train.drop(['title', 'description'], axis=1)
df_test = df_test.drop(['title', 'description'], axis=1)

#Lables
CLASS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Tech"}

#Changing the labes to start from 0 rather than 1
df_train['labels'] = df_train['labels'] - 1
df_test['labels'] = df_test['labels'] - 1

# Regular Expression for extract only the words and spaces
df_train['text'] = df_train['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df_test['text'] = df_test['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

# Regular Expression to remove all the Unnessasry words using the StopWords lib
stop_words = set(stopwords.words('english'))
df_train['text'] = df_train['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
df_test['text'] = df_test['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Regular Expression to convert all words to lowercase
df_train['text'] = df_train['text'].apply(lambda x: x.lower())
df_test['text'] = df_test['text'].apply(lambda x: x.lower())


train_length_1 = len(np.where(df_train.applymap(lambda x: str(x).isspace() == True))[0])
train_length_2 = len(np.where(df_train.applymap(lambda x: str(x).isspace() == True))[1])
print("The number of strings with just whitespaces in the training dataframe are '{}' in the class column and '{}' in the text column".format(train_length_1, train_length_2))

test_length_1 = len(np.where(df_test.applymap(lambda x: str(x).isspace() == True))[0])
test_length_2 = len(np.where(df_test.applymap(lambda x: str(x).isspace() == True))[1])
print("The number of strings with just whitespaces in the testing dataframe are '{}' in the class column and '{}' in the text column".format(test_length_1, test_length_2))

model = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model)

# This is a helper function to tokenize the input plain text for input and attention data
def tokenize(dataset):
    return tokenizer(dataset['text'], truncation=True)

# Function to convert the dataframe to a dataset and then, perform tokenization on the dataset using the above helper function (tokenize)
def df_to_ds(dataframe):
    dataset = Dataset.from_pandas(dataframe, preserve_index=False)
    tokenized_ds = dataset.map(tokenize, batched=True)
    tokenized_ds = tokenized_ds.remove_columns('text')
    
    return tokenized_ds

df_train_in, df_val_in = train_test_split(df_train[['labels', 'text']], test_size=0.2, random_state=42)

#Tokenize our Training, validation and testing set
tokenized_train = df_to_ds(df_train_in)
tokenized_val = df_to_ds(df_val_in)
tokenized_test = df_to_ds(df_test)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
transformer = AutoModelForSequenceClassification.from_pretrained(model, num_labels=4)

training_args = TrainingArguments(output_dir="./results", save_strategy = 'epoch', optim="adamw_torch", learning_rate=2e-5, per_device_train_batch_size=16, per_device_eval_batch_size=16, num_train_epochs=5, weight_decay=0.01)
trainer = Trainer (model=transformer, args=training_args, train_dataset=tokenized_train, eval_dataset=tokenized_val, tokenizer=tokenizer, data_collator=data_collator)
trainer.train()

tokenized_tester = tokenized_test.remove_columns('labels')
predictions = trainer.predict(tokenized_test)


#Confusion Matrix - 
preds_flat = [np.argmax(x) for x in predictions[0]]
print(len(preds_flat))
cm = confusion_matrix(df_test['labels'], preds_flat)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


#Evaluating our results
precision, recall, fscore, support = score(df_test['labels'], preds_flat)
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))