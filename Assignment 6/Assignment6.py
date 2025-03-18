import tensorflow as tf
from tensorflow.keras import layers, models
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

#Load dataset
dataset = load_dataset("beans")

#Resize and normalize function
def preprocess_data(batch):
    images = [tf.image.resize(img, (128, 128)).numpy() / 255.0 for img in batch['image']]
    labels = batch['labels']
    return np.array(images), np.array(labels)

train_images, train_labels = preprocess_data(dataset['train'])
val_images, val_labels = preprocess_data(dataset['validation'])
test_images, test_labels = preprocess_data(dataset['test'])

#CNN model
model = models.Sequential([
    #Convolutional block 1: layer that detects low-level features, includes input layer that defines expected image input
    layers.Conv2D(32, (3, 3), activation='relu',  input_shape=(128, 128, 3)),
    #downsampling to focus on dominant features
    layers.MaxPooling2D((2, 2)),
    #Convolutional block 2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    #Convolutional block 3
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    #Flattening: converts 3d feature maps into 1d vector for dense layers
    layers.Flatten(),
    #Dense layer to combine features for activation
    layers.Dense(128, activation='relu'),
    #50% dropout to prevent overfitting
    layers.Dropout(0.5),
    #Output layer
    layers.Dense(3, activation='softmax')
])

#Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train model
history = model.fit(train_images, train_labels, 
                    validation_data=(val_images, val_labels),
                    epochs=10, batch_size=32)

#Model evaluation
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy:.2f}")

#Classification Report and Confusion Matrix
predictions = np.argmax(model.predict(test_images), axis=1)
print(classification_report(test_labels, predictions, target_names=dataset['train'].features['labels'].names))

conf_matrix = confusion_matrix(test_labels, predictions)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=dataset['train'].features['labels'].names, 
            yticklabels=dataset['train'].features['labels'].names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

#Plot training history
plt.figure(figsize=(12, 4))
#Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
#Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()
