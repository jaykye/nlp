import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

train_data, validation_data, test_data = tfds.load('imdb_reviews', split=('train[:60%]', 'train[60%:]', 'test'), as_supervised=True)

train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])


# 여기서 dataset는 dictionary이다.
type(dataset)

train_dataset, test_dataset = dataset['train'], dataset['test']
train_dataset.element_spec
#
# for example, label in train_dataset.take(1):  # TakeDataset은 generator처럼 작동한다. -- don't need to load all.
#     print('text: ', example.numpy())
#     print('label: ', label.numpy())
#
# # Train data는 섞어야한다.
# # Train Test 둘 다 batch로 만들자.
# BUFFER_SIZE = 10000  # 얘가 전체 데이터 크기보다 커야한다.
# BATCH_SIZE = 64
# train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#
# for example, label in train_dataset.take(1):  # batch로 만들었으니 하나당 64개가 된다.
#     print('texts: ', example.numpy()[:2])
#     print()
#     print('labels: ', label.numpy()[:2])
#
# # 이제 텍스트 전체를 encoding해서 int로 바꾸자.
# # Vocabulary를 형성하는 것과 동시에 text를 int로 변환한다.
# SEQ_LEN = 1000
#
# from transformers import AutoTokenizer
#
# (x for x in range(100))

