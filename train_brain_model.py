import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- CONFIGURATION ---
# Dataset Structure Expectation:
# brain_dataset/
# ├── Training/
# │   ├── glioma_tumor/
# │   ├── meningioma_tumor/
# │   ├── no_tumor/
# │   └── pituitary_tumor/
# └── Testing/
#     ├── glioma_tumor/
#     ├── meningioma_tumor/
#     ├── no_tumor/
#     └── pituitary_tumor/

DATASET_DIR = 'brain_dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'Training')
VAL_DIR = os.path.join(DATASET_DIR, 'Testing')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_CLASSES = 4

def create_model(num_classes):
    # Use EfficientNetB0 as base
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def main():
    print("Checking for dataset...")
    if not os.path.exists(TRAIN_DIR):
        print(f"ERROR: Dataset not found at {TRAIN_DIR}")
        print(f"Please create a folder '{DATASET_DIR}' with 'Training' and 'Testing' subfolders.")
        return

    # 1. Data Generators
    print("Setting up data generators...")
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print("Class mappings:", train_generator.class_indices)

    # 2. Build Model
    print("Building EfficientNetB0 model...")
    model = create_model(NUM_CLASSES)
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 3. Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join('models', 'brain_tumor_classifier_v2.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

    # 4. Train (Phase 1)
    print("Starting Phase 1 training (Frozen Base)...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # 5. Fine-Tuning (Phase 2)
    print("\nStarting Phase 2 training (Fine-Tuning)...")
    
    model.trainable = True
    # Freeze all except last 20 layers
    for layer in model.layers[:-20]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    print("Training complete.")
    print(f"Best model saved to models/brain_tumor_classifier_v2.keras")

if __name__ == "__main__":
    main()
