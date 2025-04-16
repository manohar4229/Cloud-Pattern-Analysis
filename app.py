import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import streamlit as st
from PIL import Image
import segmentation_models_pytorch as smp

# Class definitions
class UNetPlusPlus(nn.Module):
    def __init__(self, encoder='timm-efficientnet-b4', num_classes=4):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights='noisy-student',
            in_channels=3,
            classes=num_classes,
            activation=None
        )
        
    def forward(self, x):
        return self.model(x)

# Helper functions
def mask2rle(mask):
    """Convert mask to RLE format"""
    pixels = mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle2mask(rle_string, height, width):
    """Convert RLE string to mask"""
    if not isinstance(rle_string, str) or rle_string == '':
        return np.zeros((height, width), dtype=np.uint8)
        
    rows, cols = height, width
    mask = np.zeros(rows * cols, dtype=np.uint8)
    
    rle_values = np.array([int(x) for x in rle_string.split()])
    start_pixels = rle_values[0::2] - 1  # Convert 1-indexed to 0-indexed
    run_lengths = rle_values[1::2]
    
    for start_pixel, run_length in zip(start_pixels, run_lengths):
        mask[start_pixel:start_pixel + run_length] = 1
        
    return mask.reshape(cols, rows).T  # Reshape according to F-order

def resize_image(image, target_shape):
    """Resizes an image to the specified shape (height, width)"""
    return cv2.resize(image, (target_shape[1], target_shape[0]))

def get_bounding_box(binary_mask):
    """Calculates bounding box coordinates for a binary mask"""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return 0, 0, 0, 0
        
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return y_min, y_max, x_min, x_max

def predict_uploaded_images(uploaded_files, model, label_thresholds, pixel_thresholds):
    """Process uploaded images and generate predictions"""
    results = []
    visualization_data = []
    labels = ['Fish', 'Flower', 'Gravel', 'Sugar']
    
    with torch.no_grad():
        for uploaded_file in uploaded_files[:10]:  # Process up to 10 images
            img_name = uploaded_file.name
            image = Image.open(uploaded_file).convert('RGB')
            
            # Save original image for visualization
            orig_image = np.array(image)
            
            # Preprocess for model
            resized_image = image.resize((608, 416))
            image_array = np.array(resized_image)
            image_array = image_array.transpose(2, 0, 1).astype(np.float32) / 255.0
            image_tensor = torch.tensor(image_array).unsqueeze(0).to('cpu')
            
            # Get predictions
            output = model(image_tensor)
            output = torch.sigmoid(output).cpu().numpy()[0]
            
            masks = []
            for ch in range(4):
                pred_mask = output[ch]
                max_prob = pred_mask.max()
                if max_prob < label_thresholds[ch]:
                    pred_mask = np.zeros_like(pred_mask)
                else:
                    pred_mask = (pred_mask > pixel_thresholds[ch]).astype(np.uint8)
                pred_mask = cv2.resize(pred_mask, (525, 350))
                masks.append(pred_mask)
            
            # Create visualization data
            visualization_data.append({
                'image_name': img_name,
                'original_image': orig_image,
                'masks': np.stack(masks, axis=-1)
            })
            
            # Create results dataframe
            for i, label in enumerate(labels):
                pred_mask = masks[i]
                rle = mask2rle(pred_mask) if pred_mask.sum() > 0 else ''
                results.append({'Image_Label': f'{img_name}_{label}', 'EncodedPixels': rle})
    
    return pd.DataFrame(results), visualization_data

def create_visualization(viz_data, selected_classes=None):
    """Create visualization of predictions with bounding boxes"""
    if selected_classes is None:
        selected_classes = ['Fish', 'Flower', 'Gravel', 'Sugar']
    
    class_indices = [['Fish', 'Flower', 'Gravel', 'Sugar'].index(cls) for cls in selected_classes]
    bbox_colors = ['red', 'green', 'blue', 'purple']
    
    for item in viz_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax1.imshow(item['original_image'])
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Image with masks and bboxes
        resized_image = cv2.resize(item['original_image'], (525, 350))
        # Convert from BGR to RGB if needed
        if resized_image.shape[2] == 3:
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        ax2.imshow(resized_image)
        
        combined_mask = np.zeros((350, 525, 3), dtype=np.uint8)
        
        for i in class_indices:
            mask = item['masks'][..., i]
            if np.sum(mask) > 0:
                # Make sure mask is 8-bit (uint8) before morphological operations
                mask_uint8 = mask.astype(np.uint8)
                
                # Apply closing operation to clean up mask
                kernel = np.ones((4, 4), np.uint8)
                processed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
                
                # Find connected components - ensure input is uint8
                # This line needed to be fixed
                num_components, components = cv2.connectedComponents(processed_mask.astype(np.uint8))
                
                for j in range(1, num_components):
                    component_mask = components == j
                    y_min, y_max, x_min, x_max = get_bounding_box(component_mask)
                    
                    # Only draw if we have a valid box
                    if x_max > x_min and y_max > y_min:
                        bbox = patches.Rectangle(
                            (x_min, y_min), 
                            x_max - x_min, 
                            y_max - y_min, 
                            linewidth=2, 
                            edgecolor=bbox_colors[i], 
                            facecolor='none'
                        )
                        ax2.add_patch(bbox)
                        ax2.text(
                            x_min, 
                            y_min - 5, 
                            ['Fish', 'Flower', 'Gravel', 'Sugar'][i], 
                            bbox=dict(facecolor=bbox_colors[i], alpha=0.5),
                            fontsize=8, 
                            color='white'
                        )
                
                # Add semi-transparent mask to the combined mask
                color_mask = np.zeros((350, 525, 3), dtype=np.uint8)
                if i == 0:  # Fish - red
                    color_mask[mask > 0] = [255, 0, 0]
                elif i == 1:  # Flower - green
                    color_mask[mask > 0] = [0, 255, 0]
                elif i == 2:  # Gravel - blue
                    color_mask[mask > 0] = [0, 0, 255]
                elif i == 3:  # Sugar - purple
                    color_mask[mask > 0] = [128, 0, 128]
                
                combined_mask = np.maximum(combined_mask, color_mask)
        
        # Overlay mask on image
        overlay = resized_image.copy()
        cv2.addWeighted(combined_mask, 0.4, overlay, 0.6, 0, overlay)
        ax2.imshow(overlay)
        ax2.set_title("Predictions with Bounding Boxes")
        ax2.axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)

# Streamlit App
def main():
    st.set_page_config(page_title="Cloud Pattern Analysis", layout="wide")
    
    st.title("Cloud Pattern Analysis")
    st.write("Upload images to detect and segment Fish, Flower, Gravel, and Sugar")
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    
    # Model selection
    model_path = st.sidebar.selectbox(
        "Select model checkpoint",
        ["best_network1.pth","best_network.pth"],
        index=0
    )
    
    # Advanced settings collapsible
    with st.sidebar.expander("Advanced Settings"):
        # Thresholds for each class
        st.subheader("Label Thresholds")
        fish_threshold = st.slider("Fish Label Threshold", 0.5, 1.0, 0.85, 0.01)
        flower_threshold = st.slider("Flower Label Threshold", 0.5, 1.0, 0.92, 0.01)
        gravel_threshold = st.slider("Gravel Label Threshold", 0.5, 1.0, 0.85, 0.01)
        sugar_threshold = st.slider("Sugar Label Threshold", 0.5, 1.0, 0.85, 0.01)
        
        st.subheader("Pixel Thresholds")
        fish_pixel = st.slider("Fish Pixel Threshold", 0.1, 0.5, 0.21, 0.01)
        flower_pixel = st.slider("Flower Pixel Threshold", 0.1, 0.5, 0.44, 0.01)
        gravel_pixel = st.slider("Gravel Pixel Threshold", 0.1, 0.5, 0.4, 0.01)
        sugar_pixel = st.slider("Sugar Pixel Threshold", 0.1, 0.5, 0.3, 0.01)
    
    # Visualization options
    st.sidebar.subheader("Visualization Options")
    show_classes = st.sidebar.multiselect(
        "Show classes",
        ["Fish", "Flower", "Gravel", "Sugar"],
        default=["Fish", "Flower", "Gravel", "Sugar"]
    )
    
    # Add option to disable visualization (useful for debugging)
    enable_visualization = st.sidebar.checkbox("Enable visualization", value=True)
    
    # Load model
    @st.cache_resource
    def load_model(checkpoint_path):
        try:
            model = UNetPlusPlus(encoder='timm-efficientnet-b4', num_classes=4)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            new_checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            model.load_state_dict(new_checkpoint)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    
    # File uploader
    uploaded_files = st.file_uploader("Upload up to 10 images", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    
    # Only try to load model if there are uploaded files
    model = None
    if uploaded_files:
        if os.path.exists(model_path):
            with st.spinner("Loading model..."):
                model = load_model(model_path)
        else:
            st.warning(f"Model file {model_path} not found. Please make sure it exists in the current directory.")
    
    # Handle case when files are uploaded and model is loaded
    if uploaded_files and model:
        with st.spinner("Processing images..."):
            try:
                # Set thresholds
                label_thresholds = [fish_threshold, flower_threshold, gravel_threshold, sugar_threshold]
                pixel_thresholds = [fish_pixel, flower_pixel, gravel_pixel, sugar_pixel]
                
                # Process images
                submission_df, viz_data = predict_uploaded_images(uploaded_files, model, label_thresholds, pixel_thresholds)
                
                # Show dataframe
                st.subheader("Prediction Results")
                st.dataframe(submission_df)
                
                # Download option
                st.download_button(
                    label="Download Submission CSV",
                    data=submission_df.to_csv(index=False),
                    file_name="submission.csv",
                    mime="text/csv"
                )
                
                # Create visualizations
                if enable_visualization:
                    st.subheader("Visualization")
                    try:
                        create_visualization(viz_data, show_classes)
                    except Exception as e:
                        st.error(f"Error in visualization: {e}")
                        st.text("Detailed error information:")
                        st.code(str(e))
            except Exception as e:
                st.error(f"Error during processing: {e}")
                st.text("Please check if your images are in the correct format and try again.")
    elif not model and uploaded_files:
        st.error("Model not loaded. Please check if the model file exists and is accessible.")
    elif not uploaded_files:
        st.info("Please upload some images to begin.")

if __name__ == "__main__":
    main()