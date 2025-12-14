import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATASET_DIR = 'eye_dataset' 
IMAGES_DIR = os.path.join(DATASET_DIR, 'train_images')
CSV_FILE = os.path.join(DATASET_DIR, 'train.csv')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

def create_model(num_classes):
    # Use EfficientNetB0 as base
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def main():
    print("Checking for dataset...")
    if not os.path.exists(IMAGES_DIR) or not os.path.exists(CSV_FILE):
        print(f"ERROR: Dataset not found!")
        return

    # 1. Load CSV
    print("Loading data...")
    df = pd.read_csv(CSV_FILE)
    df['filename'] = df['id_code'].astype(str) + '.png'
    df['label'] = df['diagnosis'].astype(str)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    # 2. Data Generators
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=IMAGES_DIR,
        x_col="filename",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=IMAGES_DIR,
        x_col="filename",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    # Calculate class weights
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label']
    )
    class_weights_dict = dict(enumerate(class_weights))
    print(f"Class Weights: {class_weights_dict}")

    # 3. Create and Compile Model
    num_classes = len(train_df['label'].unique())
    print(f"Building EfficientNetB0 model for {num_classes} classes...")
    
    model = create_model(num_classes)
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join('models', 'diabetic_retinopathy_model.keras'),
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
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        validation_steps=len(val_df) // BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # 5. Fine-Tuning (Phase 2)
    print("\nStarting Phase 2 training (Fine-Tuning)...")
    
    model.trainable = True
    # Freeze all except last 30 layers
    for layer in model.layers[:-30]:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    history_fine = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        steps_per_epoch=len(train_df) // BATCH_SIZE,
        validation_steps=len(val_df) // BATCH_SIZE,
        class_weight=class_weights_dict,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # 6. Save Model
    save_path = os.path.join('models', 'diabetic_retinopathy_model.keras')
    os.makedirs('models', exist_ok=True)
    model.save(save_path)
    print(f"SUCCESS! Model saved to: {save_path}")

if __name__ == "__main__":
    main()
