from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import openai
from io import BytesIO
import base64
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

load_dotenv()

key = os.getenv('API_KEY')
client = openai.OpenAI(api_key=key)

base_model = VGG16(weights='imagenet', include_top=False)

def encode_image_to_base64(image_path):
    """Encode an image to a base64 string."""
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def preprocess_image(image_path):
    """Preprocess the image for feature extraction."""  
    # Load the image with the correct target size for VGG16
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(img_array):
    """Extract features from the image using VGG16."""
    features = base_model.predict(img_array)
    return features.flatten()

def cosine_similarity(a, b):
    """Compute the cosine similarity between two feature vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compare_ultrasound_images(image1_path, image2_path):
    """Compare two ultrasound images and return a similarity score."""
    # Preprocess the images
    image1_array = preprocess_image(image1_path)
    image2_array = preprocess_image(image2_path)

    # Extract features
    features1 = extract_features(image1_array)
    features2 = extract_features(image2_array)

    # Calculate cosine similarity
    similarity = cosine_similarity(features1, features2)

    return similarity

def gpt4_analysis(similarity, threshold=0.8):
    """Generate a detailed report using GPT-4 based on the comparison."""
    if similarity > threshold:
        similarity_report = f"Las imágenes son similares con una puntuación de similitud coseno de {similarity:.2f}."
    else:
        similarity_report = f"Las imágenes no son lo suficientemente similares con una puntuación de similitud coseno de {similarity:.2f}."

    prompt = f"""
    Comparamos dos imágenes de ultrasonido para identificar estructuras principales como el hígado y la vesícula biliar. Aquí están los resultados:
    Informe de similitud: {similarity_report}
    Basado en esta comparación, proporciona un análisis detallado de las imágenes. Comenta sobre la presencia y claridad de las estructuras principales en la imagen subida por el usuario en comparación con la imagen de referencia.
    No menciones nada mas despues de llegar a la conclusion.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Eres un experto en análisis de imagenes de ultrasonido abdominal."},
            {"role": "user", "content": prompt}
        ]
    )

    total_tokens = response.usage.total_tokens
    
    return response.choices[0].message.content.strip(), total_tokens

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict')
def predict():
  if key is None:
    return Exception('API key not found')
  
  image = request.args.get('image')

  if image is None:
    return "Error"
  
  base_image = encode_image_to_base64(image)
  user_image = encode_image_to_base64(request.files['user_image'])

  similarity = compare_ultrasound_images(base_image, user_image) 
  detailed_report, total_tokens = gpt4_analysis(similarity)

  return jsonify({'similarity': similarity, 'detailed_report': detailed_report, 'total_tokens': total_tokens})

if __name__ == '__main__':
  app.run(port=5000)