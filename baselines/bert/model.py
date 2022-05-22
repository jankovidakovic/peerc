import os
from random import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

# Set parameters:
from transformers import TFDistilBertModel, DistilBertConfig, DistilBertTokenizerFast

from dataset import load_datasets

params = {'MAX_LENGTH': 25,
          'EPOCHS': 6,
          'LEARNING_RATE': 5e-5,
          'FT_EPOCHS': 2,
          'OPTIMIZER': 'adam',
          'FL_GAMMA': 2.0,
          'FL_ALPHA': 0.2,
          'BATCH_SIZE': 64,
          'NUM_STEPS': len(X_train.index) // 64,
          'DISTILBERT_DROPOUT': 0.2,
          'DISTILBERT_ATT_DROPOUT': 0.2,
          'LAYER_DROPOUT': 0.2,
          'KERNEL_INITIALIZER': 'GlorotNormal',
          'BIAS_INITIALIZER': 'zeros',
          'POS_PROBA_THRESHOLD': 0.5,
          'ADDED_LAYERS': 'Dense 256, Dense 32, Dropout 0.2',
          'LR_SCHEDULE': '5e-5 for 6 epochs, Fine-tune w/ adam for 2 epochs @2e-5',
          'FREEZING': 'All DistilBERT layers frozen for 6 epochs, then unfrozen for 2',
          'CALLBACKS': '[early_stopping w/ patience=0]',
          'RANDOM_STATE': 42
          }

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(params['RANDOM_STATE'])

# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(params['RANDOM_STATE'])

# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(params['RANDOM_STATE'])

# 4. Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed=params['RANDOM_STATE'])

MAX_LENGTH = params['MAX_LENGTH']
BATCH_SIZE = params['BATCH_SIZE']


# Define function to encode text data in batches
def batch_encode(tokenizer, texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    """""""""
    A function that encodes a batch of texts and returns the texts'
    corresponding encodings and attention masks that are ready to be fed
    into a pre-trained transformer model.

    Input:
        - tokenizer:   Tokenizer object from the PreTrainedTokenizer Class
        - texts:       List of strings where each string represents a text
        - batch_size:  Integer controlling number of texts in a batch
        - max_length:  Integer controlling max number of words to tokenize in a given text
    Output:
        - input_ids:       sequence of texts encoded as a tf.Tensor object
        - attention_mask:  the texts' attention mask encoded as a tf.Tensor object
    """""""""

    input_ids = []
    attention_mask = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding='max_length',  # implements dynamic padding
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )

        input_ids.extend(inputs['input_ids'])
        attention_mask.extend(inputs['attention_mask'])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)


def train():
    train, dev = load_datasets()

    X_train, y_train = train["text"], train["label_val"]
    X_dev, y_dev = dev["text"], dev["label_val"]

    BERT_NAME = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_NAME)

    # Encode X_train
    X_train_ids, X_train_attention = batch_encode(tokenizer, X_train.tolist())

    # Encode X_valid
    X_valid_ids, X_valid_attention = batch_encode(tokenizer, X_dev.tolist())

    config = DistilBertConfig(dropout=params['DISTILBERT_DROPOUT'],
                              attention_dropout=params['DISTILBERT_ATT_DROPOUT'],
                              output_hidden_states=True)

    distilBERT = TFDistilBertModel.from_pretrained(BERT_NAME)

    # Freeze DistilBERT layers to preserve pre-trained weights
    for layer in distilBERT.layers:
        layer.trainable = False

    # Build model
    model = build_model(distilBERT)

    model.summary()

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      mode='min',
                                                      min_delta=0,
                                                      patience=0,
                                                      restore_best_weights=True)

    # Train the model
    train_history1 = model.fit(
        x=[X_train_ids, X_train_attention],
        y=y_train,
        epochs=params['EPOCHS'],
        batch_size=params['BATCH_SIZE'],
        steps_per_epoch=params['NUM_STEPS'],
        validation_data=([X_valid_ids, X_valid_attention], y_valid),
        callbacks=[early_stopping],
        verbose=1
    )


def build_model(transformer, max_length=params['MAX_LENGTH']):
    # Define weight initializer with a random seed to ensure reproducibility
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=params['RANDOM_STATE'])

    # Define input layers
    input_ids_layer = tf.keras.layers.Input(shape=(max_length,),
                                            name='input_ids',
                                            dtype='int32')
    input_attention_layer = tf.keras.layers.Input(shape=(max_length,),
                                                  name='input_attention',
                                                  dtype='int32')

    # DistilBERT outputs a tuple where the first element at index 0
    # represents the hidden-state at the output of the model's last layer.
    # It is a tf.Tensor of shape (batch_size, sequence_length, hidden_size=768).
    last_hidden_state = transformer([input_ids_layer, input_attention_layer])[0]

    # We only care about DistilBERT's output for the [CLS] token, which is located
    # at index 0.  Splicing out the [CLS] tokens gives us 2D data.
    cls_token = last_hidden_state[:, 0, :]

    D1 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                 seed=params['RANDOM_STATE']
                                 )(cls_token)

    X = tf.keras.layers.Dense(256,
                              activation='relu',
                              kernel_initializer=weight_initializer,
                              bias_initializer='zeros'
                              )(D1)

    D2 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                 seed=params['RANDOM_STATE']
                                 )(X)

    X = tf.keras.layers.Dense(32,
                              activation='relu',
                              kernel_initializer=weight_initializer,
                              bias_initializer='zeros'
                              )(D2)

    D3 = tf.keras.layers.Dropout(params['LAYER_DROPOUT'],
                                 seed=params['RANDOM_STATE']
                                 )(X)

    # Define a single node that makes up the output layer (for binary classification)
    output = tf.keras.layers.Dense(1,
                                   activation='sigmoid',
                                   kernel_initializer=weight_initializer,  # CONSIDER USING CONSTRAINT
                                   bias_initializer='zeros'
                                   )(D3)

    # Define the model
    model = tf.keras.Model([input_ids_layer, input_attention_layer], output)

    # Compile the model
    model.compile(tf.keras.optimizers.Adam(lr=params['LEARNING_RATE']),
                  loss=focal_loss(),
                  metrics=['accuracy'])

    return model

def focal_loss(gamma=0.5, alpha=0.5):
    """""""""
    Function that computes the focal loss.
    Code adapted from https://gist.github.com/mkocabas/62dcd2f14ad21f3b25eac2d39ec2cc95
    """""""""

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

