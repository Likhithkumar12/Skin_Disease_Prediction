import gradio as gr
import cv2
import numpy as np
import pickle
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
vgg_model=VGG19(weights='imagenet',include_top=False,input_shape=(224,224,3))
x = GlobalAveragePooling2D()(vgg_model.output)
vgg_inputs=vgg_model.input
feature_model=Model(inputs=vgg_inputs,outputs=x)
with open('xgb1.pkl','rb')as f:
    xgb=pickle.load(f)
def kmeans_segment(image, k=4):
    pixel_values = image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)
    return segmented_image
def final_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return binary_image
def grabcut_segmentation(image):
    kernel = np.ones((5,5), dtype=np.uint8)
    gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    image_blackhat=cv2.morphologyEx(gray_image,cv2.MORPH_BLACKHAT,kernel)
    mask1 = cv2.threshold(image_blackhat, 5, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)
    image1 = cv2.inpaint(image, mask1, 20 ,cv2.INPAINT_NS)
    mask = np.zeros(image1.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, image1.shape[1] - 100, image1.shape[0] - 100)
    cv2.grabCut(image1, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image1 * mask2[:, :, np.newaxis]
    return segmented_image
def preprocess_and_extract_features(image):
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    features = feature_model.predict(image)
    return features.flatten()


def process_image(image):
    grabcut_img = grabcut_segmentation(image)
    kmeans_img = kmeans_segment(grabcut_img)
    final_img = final_mask(grabcut_img)

    print("GrabCut Image:", grabcut_img.shape)
    print("KMeans Image:", kmeans_img.shape)
    print("Final Mask Image:", final_img.shape)

    features = preprocess_and_extract_features(grabcut_img)

    print("Extracted Features:", features)

    prediction = xgb.predict([features])

    if hasattr(xgb, "predict_proba"):
        print("Prediction Probabilities:", xgb.predict_proba([features]))

    class_labels = {
        5: 'Melanocytic Nevi',
        4: 'Melanoma',
        3: 'Dermatofibroma',
        2: 'Benign Keratosis-like Lesions',
        1: 'Basal Cell Carcinoma',
        0: 'Actinic Keratoses'
    }
    predicted_class = class_labels.get(prediction[0], 'Unknown')


    print("Predicted Class:", predicted_class)

    return grabcut_img, kmeans_img, final_img, predicted_class
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(type="numpy", label="GrabCut Segmented Image"),
        gr.Image(type="numpy", label="KMeans Segmented Image"),
        gr.Image(type="numpy",label='maked Image'),
        'text',

    ],
    title="AI TOOL FOR DERMATOLOGIST",
    description="Upload an image to obtain the masked image and the skin disease."
)

interface.launch()