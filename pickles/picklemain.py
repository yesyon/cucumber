import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a new model instance
model = create_model()

# Data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=20, 
    zoom_range=0.2, 
    shear_range=0.2, 
    horizontal_flip=True, 
    validation_split=0.2  # Use 20% of data for validation
)

train_generator = train_datagen.flow_from_directory(
    'data/train',  # Path to training data
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='binary', 
    subset='training'  # Use this subset for training
)

validation_generator = train_datagen.flow_from_directory(
    'data/train',  # Path to validation data
    target_size=(64, 64), 
    batch_size=32, 
    class_mode='binary', 
    subset='validation'  # Use this subset for validation
)

# Train the model
model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10  # Adjust the number of epochs as needed
)

# Save the trained model
model.save('cucumber_model.h5')
