import numpy as np

import os
import tensorflow as tf
from utils import load_data_set, load_image_test, map_prediction_to_mask, compare_accuracy, labeled_prediction_to_image, \
    count_files_in_folder, files_in_folder

from DeepPanelTest import generate_output_template


def load_tflite():
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="pretrained.model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']

    print(" - Loading test data")
    testing_files_path = "./dataset/test/raw"
    testing_num_files = count_files_in_folder(testing_files_path)
    TESTING_BATCH_SIZE = testing_num_files
    raw_dataset = load_data_set()
    test_raw_dataset = raw_dataset['test']
    test = test_raw_dataset.map(load_image_test)
    test_dataset = test.batch(TESTING_BATCH_SIZE)
    for images, true_masks in test_dataset:
        pass  # Hack needed to be able to extrac images and true masks from map datasets
    images = images.numpy()
    true_masks = true_masks.numpy()
    # We need to manipulate here

    random_input = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = images

    num_images = images.shape[0]

    predictions = []
    for image_idx in range(0, num_images):
        img = np.expand_dims(input_data[image_idx, :, :, :], axis=0)
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output_data)

    print(f" - Let's transform predictions into labeled values.")
    labeled_predictions = []
    for image_index in range(len(predictions)):
        prediction = predictions[image_index]
        prediction = np.squeeze(prediction)
        predicted_mask = map_prediction_to_mask(prediction)
        labeled_predictions.append(predicted_mask)

    print(f" - Saving labeled images into ./output folder")
    predicted_index = 0
    output_path = "./output/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for predicted_result in labeled_predictions:
        prediction_as_image = labeled_prediction_to_image(predicted_result)
        prediction_as_image.save(f"{output_path}{predicted_index:03d}.jpg")
        prediction_as_image.close()
        print(f"    - Image with index {predicted_index} saved.")
        predicted_index += 1
    print(f" - Generating sample output page")
    generate_output_template()


if __name__ == "__main__":
    load_tflite()
