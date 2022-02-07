# Import the needed libraries
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np

# Set the configurations
VECTOR_SIZE = 1
VOCAB_SIZE = 10000
SEQ_LENGTH = 100
NUM_EPOCHS = 5
BATCH_SIZE = 1
EMBEDDING_DIM = 16
INPUT_DATA_PATH = r'/data/WatchHistory.csv'
CHECKPOINT_PATH = r'/Models/client_model.ckpt'
TF_LITE_MODEL_DIRECTORY = "tflite_client_model_v1"

class Model(tf.Module):

    def __init__(self, llst_vocab):
        try:
            self.model = tf.keras.Sequential([
                tf.keras.layers.TextVectorization(
                    max_tokens=VOCAB_SIZE, output_mode='int', 
                    output_sequence_length=SEQ_LENGTH, vocabulary = llst_vocab),
                tf.keras.layers.Embedding(VOCAB_SIZE + 1, 16, name="embedding"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu', name='dense_1'),
                tf.keras.layers.Dense(128, activation='relu', name='dense_2'),
                tf.keras.layers.Dense(3, name='dense_3')
            ])

            self.model.compile(
                optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

        except Exception as e:
            raise e
            
    # Input will be a string eg: b'Mr. Beast is the best' and an encoded label vector eg: [0.,0.,1.]
    @tf.function(input_signature=[
      tf.TensorSpec([None, VECTOR_SIZE], tf.string),
      tf.TensorSpec([None, 3], tf.float32),
    ])
    def train(self, pstr_input_string, parr_target_label):
        try:
            # We use tf.GradientTape to record all operations for automatic differentiation
            with tf.GradientTape() as tape:
                # Make the predction for the given input string
                prediction = self.model(pstr_input_string)
                
                # Get the loss for the training step
                loss = self.model.loss(parr_target_label, prediction)
                
            # Calculate the gradients for the trainable variables using the loss
            # Returns gradients in the same dimensions as the model parameters
            gradients = tape.gradient(loss, self.model.trainable_variables)
            
            # Apply the new gradients to the model parameters
            self.model.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))
            result = {"loss": loss}
            return result
        except Exception as e:
            raise e

    # Inference function takes and input eg: b'Mr. Beast is the best' and returns the output probabilities
    @tf.function(input_signature=[
      tf.TensorSpec([None, VECTOR_SIZE], tf.string),
    ])
    def infer(self, pstr_input_string):
        try:
            logits = self.model(pstr_input_string)
            probabilities = tf.nn.softmax(logits, axis=-1)
            return {
                "output": probabilities,
                "logits": logits
                }

        except Exception as e:
            raise e

    # Save function takes the checkpoint path as input where it saves the model
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def save(self, checkpoint_path):
        try:
            # Reads the layer names
            tensor_names = [weight.name for weight in self.model.weights]

            # Reads the weights associated with the layers
            tensors_to_save = [weight.read_value() for weight in self.model.weights]

            # tensor_names and tensors_to_save should be of the same length
            tf.raw_ops.Save(
                filename=checkpoint_path, tensor_names=tensor_names,
                data=tensors_to_save, name='save')

            return {"checkpoint_path": checkpoint_path}

        except Exception as e:
            raise e

    # Restore returns a dictionary of tensors stored using the save funciton
    @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
    def restore(self, checkpoint_path):
        try:
            restored_tensors = {}
            for var in self.model.weights:
                restored = tf.raw_ops.Restore(
                file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
                name='restore')
                var.assign(restored)
                restored_tensors[var.name] = restored
            return restored_tensors

        except Exception as e:
            raise e

def get_vocabulary(pobj_train_data):
    try:  
        vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_mode='int',
            output_sequence_length=SEQ_LENGTH)
        
        # Call adapt to build the vocabulary.
        vectorize_layer.adapt(pobj_train_data)

        # Return the vocabulary
        return vectorize_layer.get_vocabulary()

    except Exception as e:
            raise e

def get_data():
    try:
        # Read the input csv into a dataframe
        ldf_yt_history = pd.read_csv(INPUT_DATA_PATH, names=['Title','Time','Label'])
        # Split the dataset into training and test
        train_data, test_data, train_labels, test_labels = train_test_split(
            ldf_yt_history['Title'], ldf_yt_history['Label'], test_size=0.25)

        # Reshape the data to suit the model
        train_data = train_data.values.reshape(len(train_data),1)
        test_data = test_data.values.reshape(len(test_data),1)
        train_labels = train_labels.values.reshape(len(train_labels),1)
        test_labels = test_labels.values.reshape(len(test_labels),1)

        # Convert labels to one-hot categorical vectors
        train_labels = tf.keras.utils.to_categorical(train_labels)
        test_labels = tf.keras.utils.to_categorical(test_labels)

        return train_data, test_data, train_labels, test_labels

    except Exception as e:
            raise e

def convert_to_tf_lite(pobj_model):
    try:
        tf.saved_model.save(
            pobj_model,
            TF_LITE_MODEL_DIRECTORY,
            signatures={
                'train':
                    pobj_model.train.get_concrete_function(),
                'infer':
                    pobj_model.infer.get_concrete_function(),
                'save':
                    pobj_model.save.get_concrete_function(),
                'restore':
                    pobj_model.restore.get_concrete_function(),
            })

        # Convert the model to tflite
        converter = tf.lite.TFLiteConverter.from_saved_model(TF_LITE_MODEL_DIRECTORY)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        converter.experimental_enable_resource_variables = True
        tflite_model = converter.convert()
        return tflite_model

    except Exception as e:
            raise e


if __name__ == "__main__":
    try:
        train_data, test_data, train_labels, test_labels = get_data()
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        train_dataset = train_dataset.batch(BATCH_SIZE)

        llst_vocab = get_vocabulary(train_data)

        lobj_model = Model(llst_vocab)
        larr_losses = np.zeros([NUM_EPOCHS])

        for i in range(NUM_EPOCHS):
            for input_string,label in train_dataset:
                ldict_result = lobj_model.train(input_string, label)
                larr_losses[i] = ldict_result['loss']
            if (i + 1) % 10 != 0:
                print(f"Epoch: {i+1}")
                print(f"\tloss: {larr_losses[i]:.4f}")

        # Save the trained weights to a checkpoint.
        lobj_model.save(CHECKPOINT_PATH)

        # Finally convert the model to tflite
        _ = convert_to_tf_lite(lobj_model)
        
    except Exception as e:
        raise e
