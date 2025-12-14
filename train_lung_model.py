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
DATASET_DIR = 'lung_dataset'
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'valid') # Use 'valid' folder for validation

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30 # Increased epochs
LEARNING_RATE = 1e-3 # Slightly higher initial LR
NUM_CLASSES = 4

def create_model(num_classes):
    # Load EfficientNetB0 pre-trained on ImageNet
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers initially
    base_model.trainable = False
        
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x) # Increased dense layer size
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create final model
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def main():
    print("Checking for dataset...")
    if not os.path.exists(TRAIN_DIR):
        print(f"ERROR: Dataset not found at {TRAIN_DIR}")
        return

    # 1. Data Generators
    print("Setting up data generators...")
    
    # Enhanced Augmentation
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
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
    print("Building EfficientNetB0-based model...")
    model = create_model(NUM_CLASSES)
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # 3. Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join('models', 'lung_cancer_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10, # Increased patience
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # 4. Train Phase 1 (Frozen)
    print("Starting Phase 1 training (Frozen Base)...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    # 5. Fine-Tuning
    print("\nStarting Phase 2 training (Fine-Tuning)...")
    # Unfreeze the top 30 layers
    model.trainable = True
    for layer in model.layers[:-30]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=1e-5), # Low LR for fine-tuning
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    history_fine = model.fit(
        train_generator,
        epochs=20, # Increased fine-tuning epochs
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    print("Training complete.")
    print(f"Best model saved to models/lung_cancer_model.keras")

if __name__ == "__main__":
    main()
