import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Function to resize model image
def resize_img(img_path, output_path):
    img = Image.open(img_path)
    new_size = (576, 1024)
    resized_img = img.resize(new_size)
    resized_img.save(output_path, format='PNG')

# Function to resize garment image
def resize_img_rgba(img_path, output_path):
    img = Image.open(img_path)
    new_size = (576, 1024)
    resized_img = img.resize(new_size)
    resized_img.save(output_path, format='PNG')

def generate_and_save_masks(image_path, output_dir, garment_type):
    # Open image and convert to RGBA
    image = Image.open(image_path).convert("RGBA")
    
    # Convert image to numpy array
    g_img = np.array(image)

    # Create a mask from the alpha channel
    alpha_mask = g_img[:, :, 3]
    
    # Threshold the alpha mask
    _, binary_mask = cv2.threshold(alpha_mask, 0, 255, cv2.THRESH_BINARY)

    # Find contours on the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Number of contours found: {len(contours)}")

    if len(contours) > 0:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create blank RGBA images for each mask
        height, width = g_img.shape[:2]
        contour_img = np.zeros((height, width, 4), dtype=np.uint8)
        rectangle_img = np.zeros((height, width, 4), dtype=np.uint8)
        trapezium_img = np.zeros((height, width, 4), dtype=np.uint8)
        mask_img = np.zeros((height, width, 4), dtype=np.uint8)

        # Draw the largest contour
        cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0, 255), thickness=cv2.FILLED)
        cv2.drawContours(mask_img, [largest_contour], -1, (0, 255, 0, 255), 3)

        # Get bounding rectangle for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding rectangle
        cv2.rectangle(rectangle_img, (x, y), (x + w, y + h), (255, 0, 0, 255), thickness=cv2.FILLED)
        cv2.rectangle(mask_img, (x, y), (x + w, y + h), (255, 0, 0, 255), 2)

        # Define trapezium points (customize as needed)
        top_left = (x + int(w * 0.2), y)
        top_right = (x + int(w * 0.8), y)
        bottom_left = (x, y + h)
        bottom_right = (x + w, y + h)

        # Create an array of points for the trapezium
        trapezium_points = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)

        # Draw the trapezium
        cv2.fillPoly(trapezium_img, [trapezium_points], color=(0, 0, 255, 255))
        cv2.polylines(mask_img, [trapezium_points], isClosed=True, color=(0, 0, 255, 255), thickness=2)

        contour_path = output_dir / f'contour_mask_{image_path.name}'
        rectangle_path = output_dir / f'rectangle_mask_{image_path.name}'
        trapezium_path = output_dir / f'trapezium_mask_{image_path.name}'
        mask_path = output_dir / f'mask_outline_{image_path.name}'

        # Function to save RGBA image
        def save_rgba_image(filename, img):
            Image.fromarray(img, 'RGBA').save(filename)

        # Save the images
        save_rgba_image(contour_path, contour_img)
        save_rgba_image(rectangle_path, rectangle_img)
        save_rgba_image(trapezium_path, trapezium_img)
        save_rgba_image(mask_path, mask_img)

        # Plot all images
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(contour_img)
        plt.title("Contours")
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(rectangle_img)
        plt.title("Bounding Rectangle")
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(trapezium_img)
        plt.title("Bounding Trapezium")
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(mask_img)
        plt.title("All three masks")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    else:
        print("No contours detected. Check thresholding or object separation.")

def main():
    # Get current working directory
    dir = Path(os.getcwd())

    # Get user input for paths and garment type
    model_img_path = Path(input("Enter the path to the model image: "))
    garment_img_path = Path(input("Enter the path to the garment image: "))
    garment_type = input("Enter the garment type (top, lower, or dress): ").lower()

    # Set up directories
    model_dir = dir / 'Model_imgs'
    garment_dir = dir / 'Garment'
    resized_model_img_dir = model_dir / 'resized'
    resized_garment_img_dir = garment_dir / 'resized'
    
    # Create directories if they don't exist
    for d in [model_dir, garment_dir, resized_model_img_dir, resized_garment_img_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Set up output directory based on garment type
    if garment_type == 'top':
        output_dir = dir / 'masks/top_masks'
    elif garment_type == 'lower':
        output_dir = dir / 'masks/lower_masks'
    elif garment_type == 'dress':
        output_dir = dir / 'masks/dress_masks'
    else:
        print("Invalid garment type. Please enter 'top', 'lower', or 'dress'.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resize images
    output_model_path = resized_model_img_dir / f'model_{model_img_path.name}'  # Extracting the name
    output_garment_path = resized_garment_img_dir / f'{garment_type}_{garment_img_path.name}'

    resize_img(model_img_path, output_model_path)
    resize_img_rgba(garment_img_path, output_garment_path)

    print("Model and garment images are resized and saved successfully!")

    # Generate masks
    generate_and_save_masks(output_garment_path, output_dir, garment_type)

    print(f"Masks for {garment_type} garment image are created and saved successfully in {output_dir}!")

if __name__ == "__main__":
    main()
