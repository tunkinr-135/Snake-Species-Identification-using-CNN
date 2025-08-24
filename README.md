# Snake-Species-Identification-using-CNN
## Objective
This project leverages Convolutional Neural Networks (CNNs), a powerful deep learning technique for image recognition, to automatically identify snake species from images. The primary objective is to build a model capable of classifying snake images into their respective species.
## 📌 Dataset
* The project uses the "Snake Dataset India", which is downloaded from KaggleHub. 
* The dataset is organized into train and test directories, each containing two subdirectories: Venomous and Non-venomous.
* The total number of classes is 2.
* The images are resized to 224x224 pixels for the model.
## 🛠️Project Workflow
### Setup and Data Acquisition:
Essential Python libraries like tensorflow, matplotlib, and scikit-learn are installed.
The "Snake Dataset India" is downloaded directly from KaggleHub and its local path is printed.
### Data Preprocessing and Augmentation:
* Image Augmentation: An ImageDataGenerator is used to augment the training data, creating new variations of images by applying transformations such as rotation, shifting, shearing, and zooming. This helps the model generalize better.
Rescaling: Both training and testing images are rescaled to a range of [0, 1] by dividing the pixel values by 255.0.
* Data Loading: The flow_from_directory function is used to load images in batches from the train and test directories. For the confusion matrix, the test_generator is configured with shuffle=False to maintain the order of predictions.
* Class Weights: To address class imbalance, class weights are computed and applied during the training process.
### Model Building and Training:
* Transfer Learning: The project utilizes a pre-trained MobileNetV2 model as a base. The include_top parameter is set to False to exclude the original classification layers.     * Model Architecture: A new classification head is added on top of the base model. This consists of a GlobalAveragePooling2D layer, a Dense layer with relu activation, and a final Dense output layer with softmax activation for the two classes.
* Freezing Layers: The layers of the MobileNetV2 base model are initially frozen (base_model.trainable = False), so only the new layers are trained.
* Compilation: The model is compiled with the adam optimizer and categorical_crossentropy loss function. The primary metric for evaluation is accuracy.
* Training: The model is trained for 20 epochs with an EarlyStopping callback to prevent overfitting. This callback monitors validation loss and stops training if it doesn't improve for three consecutive epochs, restoring the best-performing weights.

## Evaluation and Prediction:
* Test Accuracy: The final model is evaluated on the test set, achieving a test accuracy of approximately 80.86%.
* Confusion Matrix: A normalized confusion matrix is generated to visualize the model's performance on each class (non-venomous vs. venomous).
* Classification Report: A classification report provides detailed metrics, including precision, recall, and f1-score, for each class. The overall accuracy of the model on the test set is approximately 86%.

Single Image Prediction: A function is provided to load and preprocess a single image and use the trained model to predict whether the snake is venomous or non-venomous.


