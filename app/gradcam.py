import tensorflow as tf
import numpy as np
import cv2
import os

def get_gradcam_heatmap(model, img_array, target_layer_name=None):
    """
    Generates Grad-CAM heatmap for a given image and model.
    Supports both nested VGG16 (Brain) and standard MobileNetV2 (Eye).
    """
    # Strategy for EfficientNetB0 (All models now use EfficientNetB0)
    try:
        # EfficientNetB0 top_activation layer is usually 'top_activation' or 'top_conv'
        # Let's target the last convolutional layer before pooling
        if target_layer_name is None or target_layer_name in ['block5_conv3', 'out_relu']: # Handle old layer names
            target_layer_name = 'top_activation'
            
        try:
            last_conv_layer = model.get_layer(target_layer_name)
        except ValueError:
            # Fallback for EfficientNetB0 if layer name differs (e.g. trained differently)
            # Try 'top_conv' or iterate
            try:
                last_conv_layer = model.get_layer('top_conv')
            except:
                # Last resort: Last Conv2D layer
                for layer in reversed(model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Activation):
                        # EfficientNet ends with Activation (swish) usually
                        target_layer_name = layer.name
                        last_conv_layer = layer
                        break
        
        # Create a model matching Inputs -> [ConvOutput, Predictions]
        # Since we use functional API with a custom top, we need to inspect the model structure.
        # Our training scripts wrap EfficientNetB0 in a Functional Model.
        
        # model.layers[0] might be the EfficientNetB0 Functional model if nested?
        # In current training scripts:
        # model = Model(inputs=base_model.input, outputs=predictions)
        # So it is a flat Functional model? No, base_model is a Model object used in graph.
        
        # If 'model' is the full deployed model:
        # It has inputs -> efficientnet_layers -> global_pooling -> dense -> ...
        
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

    except Exception as e:
        print(f"Error building Grad-CAM model: {e}")
        return None

    # Compute Gradients
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # Get gradients
    grads = tape.gradient(top_class_channel, conv_outputs)
    
    # Global Average Pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Post-processing
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10) # Avoid div by zero
    
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, output_path, alpha=0.4):
    """
    Overlays heatmap on the original image and saves it.
    """
    # Load original image
    img = cv2.imread(img_path)
    
    # Resize heatmap to match image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Rescale heatmap to 0-255 and convert to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose
    superimposed_img = heatmap * alpha + img
    
    # Save
    cv2.imwrite(output_path, superimposed_img)
    return output_path
