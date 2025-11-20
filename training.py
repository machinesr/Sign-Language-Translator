import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Masking,
    Layer,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

DATA_PATH = "data"
BATCH_SIZE = 32
TARGET_FRAMES = 45
SPATIAL_JITTER_STD = 0.005
TEST_SPLIT_SIZE = 0.2
RANDOM_SEED = 42

print("Scanning data directory...")
sequences = []
sign_folders = sorted([f for f in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))])

label_encoder = LabelEncoder()
label_encoder.fit(sign_folders)
np.save("labels.npy", label_encoder.classes_)
print(f"Found {len(label_encoder.classes_)} classes: {label_encoder.classes_}")

for label_int, sign_name in enumerate(label_encoder.classes_):
    sign_path = os.path.join(DATA_PATH, sign_name)
    for sample_file in os.listdir(sign_path):
        if sample_file.endswith(".npy"):
            filepath = os.path.join(sign_path, sample_file)
            sequences.append((filepath, label_int))

random.seed(RANDOM_SEED)
random.shuffle(sequences)

print(f"Found {len(sequences)} total samples...")

train_seq, test_seq = train_test_split(
    sequences, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED, stratify=[s[1] for s in sequences]
)

print(f"Training samples: {len(train_seq)}, Validation samples: {len(test_seq)}")

try:
    sample_data = np.load(sequences[0][0])
    FEATURE_VECTOR_SIZE = sample_data.shape[1]
    print(f"Detected feature vector size: {FEATURE_VECTOR_SIZE}")
except Exception as e:
    print(f"Could not auto-detect feature size from {sequences[0][0]}: {e}")
    print("Please ensure 'data' folder is not empty. Exiting.")
    exit()

num_classes = len(label_encoder.classes_)


def spatial_jitter(sequence):
    nan_mask = tf.math.is_nan(sequence)
    noise = tf.random.normal(shape=tf.shape(sequence), mean=0.0, stddev=SPATIAL_JITTER_STD, dtype=tf.float32)
    jittered_sequence = sequence + tf.where(nan_mask, 0.0, noise)
    return jittered_sequence


def temporal_crop(sequence, is_training):
    seq_len = tf.shape(sequence)[0]

    if seq_len > TARGET_FRAMES:
        if is_training:
            max_offset = seq_len - TARGET_FRAMES
            offset = tf.random.uniform(shape=(), minval=0, maxval=max_offset, dtype=tf.int32)
            sequence = sequence[offset : offset + TARGET_FRAMES]
        else:
            sequence = sequence[:TARGET_FRAMES]

    return sequence


def load_and_process(filepath, label, is_training):
    def _load_npy(filepath_bytes):
        filepath = filepath_bytes.numpy().decode("utf8")

        try:
            data = np.load(filepath).astype(np.float32)

            if data.shape[0] == 0:
                print(f"\n\n[WARNING] Empty file (0 frames): {filepath}")

            return data

        except Exception as e:
            print(f"\n\n[CRITICAL ERROR] Failed to load file: {filepath}")
            print(f"Error details: {e}")
            print("This file is likely corrupt or 0 bytes. Please delete it or re-generate it.")
            raise e

    sequence = tf.py_function(_load_npy, [filepath], tf.float32)
    sequence.set_shape([None, FEATURE_VECTOR_SIZE])
    label = tf.cast(label, tf.int32)

    if is_training:
        sequence = spatial_jitter(sequence)

    sequence = temporal_crop(sequence, is_training)
    # sequence = temporal_crop(sequence, False)

    label_one_hot = tf.one_hot(label, num_classes, dtype=tf.float32)

    return sequence, label_one_hot


def create_dataset(file_label_list, is_training):
    filepaths = [item[0] for item in file_label_list]
    labels = [item[1] for item in file_label_list]

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if is_training:
        dataset = dataset.shuffle(len(filepaths), seed=RANDOM_SEED)

    dataset = dataset.map(
        lambda fp, lbl: load_and_process(fp, lbl, is_training=is_training), num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.padded_batch(
        BATCH_SIZE,
        padded_shapes=([TARGET_FRAMES, FEATURE_VECTOR_SIZE], [num_classes]),
        padding_values=(np.nan, 0.0),
        drop_remainder=False,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


print("Creating tf.data pipelines...")
train_dataset = create_dataset(train_seq, is_training=True)
val_dataset = create_dataset(test_seq, is_training=False)

print("Sample batch from training data:")
for x_batch, y_batch in train_dataset.take(1):
    print(f"x_batch shape: {x_batch.shape}")
    print(f"y_batch shape: {y_batch.shape}")
    print(f"NaNs in batch (should be > 0 if padding occurred): {np.isnan(x_batch.numpy()).sum()}")

    nan_count = np.isnan(x_batch.numpy()).sum()
    total_count = np.prod(x_batch.shape)
    print(f"Percentage of NaN values in batch: {100 * nan_count / total_count:.2f}%")


class PositionalEncoding(Layer):
    def __init__(self, max_seq_len, embedding_dim, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        pos = np.arange(max_seq_len)[:, np.newaxis]
        i = np.arange(embedding_dim)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embedding_dim))
        angle_rads = pos * angle_rates

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        x_zeroed = tf.where(tf.math.is_nan(x), 0.0, x)

        seq_len = tf.shape(x)[1]
        return x_zeroed + self.pos_encoding[:, :seq_len, :]


class TransformerEncoderBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, activation="relu"),
                Dense(embed_dim),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False, mask=None):
        attn_output = self.att(inputs, inputs, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


def build_transformer_model(input_shape, num_classes, num_heads=4, ff_dim=256, num_blocks=8, dropout_rate=0.2):
    inputs = Input(shape=input_shape)
    x = Masking(mask_value=np.nan)(inputs)
    x = PositionalEncoding(max_seq_len=input_shape[0], embedding_dim=input_shape[1])(x)

    for _ in range(num_blocks):
        x = TransformerEncoderBlock(embed_dim=input_shape[1], num_heads=num_heads, ff_dim=ff_dim, rate=dropout_rate)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


model_input_shape = (TARGET_FRAMES, FEATURE_VECTOR_SIZE)

print(f"Building model with input shape: {model_input_shape}")
model = build_transformer_model(model_input_shape, num_classes, num_heads=2, ff_dim=128, num_blocks=2, dropout_rate=0.3)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

print("Starting model training...")

early_stopping = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
model_checkpoint = ModelCheckpoint("sign_model.h5", monitor="val_loss", save_best_only=True, verbose=1)

history = model.fit(
    train_dataset, validation_data=val_dataset, epochs=150, callbacks=[early_stopping, model_checkpoint]
)

print("Training completed.")
print("Best Model Saved as 'sign_model.h5'")
print("Label Mapping Saved as 'labels.npy'")
