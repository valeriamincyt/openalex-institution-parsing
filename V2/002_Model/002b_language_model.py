#002b_languaje_model
#!pip install datasets
#!pip install transformers==4.26.0
#!pip install tensorflow==2.11.0

# %%
import pickle
import json
import pandas as pd
pd.set_option("display.max_colwidth", None)
import numpy as np

from collections import Counter
from math import ceil
from sklearn.model_selection import train_test_split

# %%
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from datasets import load_dataset
from transformers import create_optimizer, TFAutoModelForSequenceClassification, DistilBertTokenizer
from transformers import DataCollatorWithPadding, TFDistilBertForSequenceClassification
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

print("TensorFlow version:", tf.__version__)

# %%
base_save_path = "./"
iteration_save_path = f"{base_save_path}institutional_affiliation_classification/"
rutaDatos = "../Datos/"
curr_model_artifacts_location = f"{rutaDatos}institution_tagger_v2_artifacts/"

filepath_model = f"{base_save_path}models/"

# %% [markdown]
# ### Loading Affiliation Dictionary

# %%
# Loading the affiliation (target) vocab
with open(f"{curr_model_artifacts_location}affiliation_vocab.pkl","rb") as f:
    affiliation_vocab = pickle.load(f)
    
inverse_affiliation_vocab = {i:j for j,i in affiliation_vocab.items()}

# %%
with open(f"{curr_model_artifacts_location}affiliation_vocab.pkl","wb") as f:
    pickle.dump(affiliation_vocab, f)

# %%
print('len(affiliation_vocab): -------------------------------------')
print(len(affiliation_vocab))

# %% [markdown]
# ### Tokenizing Affiliation String

# %%
# Loading the standard DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", return_tensors='tf')

# %%
# Using the HuggingFace library to load the dataset
train_dataset = load_dataset("parquet", data_files={'train': f'{rutaDatos}train_data.parquet'})
val_dataset = load_dataset("parquet", data_files={'val': f'{rutaDatos}val_data.parquet'})

# %%
MAX_LEN = 256

def preprocess_function(examples):
    return tokenizer(examples["processed_text"], truncation=True, padding=True, 
                     max_length=MAX_LEN)

# %%
# Tokenizing the train dataset
tokenized_train_data = train_dataset.map(preprocess_function, batched=False)

# %%
tokenized_train_data.cleanup_cache_files()

# %%
# Tokenizing the validation dataset
tokenized_val_data = val_dataset.map(preprocess_function, batched=False)

# %%
tokenized_val_data.cleanup_cache_files()

# %% [markdown]
# ### Creating the model

# %%
# Hyperparameters to tune
batch_size = 512
num_epochs = 15
batches_per_epoch = len(tokenized_train_data["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)

# %%
# Allow for use of multiple GPUs
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='tf')

    # Turning dataset into TF dataset
    tf_train_dataset = tokenized_train_data["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "label"],
    shuffle=True,
    batch_size=batch_size, 
    collate_fn=data_collator)

    # Turning dataset into TF dataset
    tf_val_dataset = tokenized_val_data["val"].to_tf_dataset(
    columns=["input_ids", "attention_mask", "label"],
    shuffle=False,
    batch_size=512,
    collate_fn=data_collator)

    # Using HuggingFace library to create optimizer
    lr_scheduler = PolynomialDecay(
    initial_learning_rate=5e-5, end_learning_rate=5e-7, decay_steps=total_train_steps)


    opt = Adam(learning_rate=lr_scheduler)
    
    # Loading the DistilBERT model and weights with a classification head
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(affiliation_vocab))
    print('Tipo del modelo: -------------------------------------')
    print(type(model))
    
    model.compile(optimizer=opt)

# %%
model.fit(tf_train_dataset, validation_data=tf_val_dataset, epochs=num_epochs)

# %%
tf_save_directory = f"{base_save_path}all_strings_language_model_15epochs"

# %%
# Saving the model, tokenizer, and affiliation (target) vocab
tokenizer.save_pretrained(tf_save_directory)
model.save_pretrained(tf_save_directory)
with open(f"{tf_save_directory}/vocab.pkl", "wb") as f:
    pickle.dump(affiliation_vocab, f)

# %%
print('FINALIZADO OK')