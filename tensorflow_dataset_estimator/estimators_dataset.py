import os 
import six.moves.urllib.request as request
import tensorflow as tf

tf_version = tf.__version__
print("Tensorflow version: {}".format(tf_version))
assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"

# Fetch and store Training and test dataset files.
PATH_MODEL = "model"
PATH_DATASET = "dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "iris_training.csv"
FILE_TEST = PATH_DATASET + os.sep + "iris_test.csv"
URL_TRAIN = "http://download.tensorflow.org/data/iris_training.csv"
URL_TEST = "http://download.tensorflow.org/data/iris_test.csv"

def download_dataset(url, file):
    if not os.path.exists(PATH_DATASET):
        os.makedirs(PATH_DATASET)

    if not os.path.exists(PATH_MODEL):
        os.makedirs(PATH_MODEL)

    if not os.path.exists(file):
        data = request.urlopen(url).read()
        with open(file, "wb") as f:
            f.write(data)
            f.close()

download_dataset(URL_TRAIN, FILE_TRAIN)
download_dataset(URL_TEST, FILE_TEST)

tf.logging.set_verbosity(tf.logging.INFO)

# The CSV features in our training and test data.
features_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth']

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):

    def decode_csv(line):
            parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
            label = parsed_line[-1] # Last element is the label
            del parsed_line[-1]
            features = parsed_line
            d = dict(zip(features_names, features)),label
            return d

    dataset = (tf.data.TextLineDataset(file_path)
                .skip(1)
                .map(decode_csv))

    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(repeat_count) #Repeats dataset this # times
    dataset = dataset.batch(32) # Batch size to use
    iterator = dataset.make_one_shot_iterator() #Creates an Iterator for enumerating the elements of this dataset.
    batch_features, batch_labels = iterator.get_next()

    return batch_features, batch_labels

# Create the feature_columns, which specifies the input to our model.
# All our input features are numeric, so use numeric_column for each one.

feature_columns = [tf.feature_column.numeric_column(k) for k in features_names]


# Create a deep neural network regression classifier.
# Use the DNN Classifier pre-made estimator
classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns, # The input features to our model.
        hidden_units=[10,10], # Two layers, each with 10 neurons.
        n_classes=3,
        model_dir=PATH_MODEL) # Path to where checkpoints etc are stored.

# Train our model, use the previously function my_input_fn.
# Input to training is a file with training example.
# Stop training after 8 iterations of train data (epochs)

classifier.train(
        input_fn = lambda: my_input_fn(FILE_TRAIN, True, 8))

# Evaluate our model using the examples contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
evaluate_result = classifier.evaluate(
        input_fn = lambda: my_input_fn(FILE_TEST, False, 4))

print("Evaluation results")
for key in evaluate_result:
    print(" {}, was: {}".format(key, evaluate_result[key]))

# Predict the type of some Iris flowers.
# Let's predict the examples in FILE_TEST, repeat only once.
predict_results = classifier.predict(
        input_fn = lambda: my_input_fn(FILE_TEST, False, 1))

print("Predictions on test file")
for prediction in predict_results:
    # Will print the predicted class, i.e: 0, 1, or 2 if the prediction
    # is Iris Sentos, Vericolor, Virginica, respectively.
    print(prediction["class_ids"][0])


        
# Let create a memory dataset for prediction.
prediction_input = [[5.9, 3.0, 4.2, 1.5],
                    [6.9, 3.1, 5.4, 2.1],
                    [5.1, 3.3, 1.7, 0.5]]


def new_input_fn():
    def decode(x):
        x = tf.split(x, 4) # Need to split into our 4 features.
        # When prediction, we don't need (or have) any labels.
        return dict(zip(features_names, x)) # Then build a dict fro them.

    # The form_tensor_slices function will use a memory structure as input
    dataset = tf.data.Dataset.from_tensor_slices(prediction_input)
    dataset = dataset.map(decode)
    iterator = dataset.make_one_shot_iterator()
    next_feature_batch = iterator.get_next()
    return next_feature_batch, None # In prediction ,we have no labels

# Predict all our prediction_input
predict_results = classifier.predict(input_fn=new_input_fn)

# Print results
print("Predictions on memory data")
for idx, prediction in enumerate(predict_results):
    type = prediction["class_ids"][0] # Get the predicted class (index)

    if type == 0:
        print("I think: {}, is Iris Sentosa".format(prediction_input[idx]))
    elif type == 1:
        print("I think: {}, is Iris Versicolor".format(prediction_input[idx]))
    else:
        print("I think: {}, is Iris Virginica".format(prediction_input[idx]))

