from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import io
import os
from model.caption_model import generate_caption, Translate_caption
# from model.Model import generate_caption, Translator

# from tensorflow.keras.models import load_model
from keras.models import load_model

import os

file_path = "image_caption_model.pth"
print("File exists:", os.path.exists(file_path))

# Load your trained model
# model = load_model('image_caption_model1.pth')

# Initialize Flask app
app = Flask(__name__)

# Set up the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Generate caption
    caption = generate_caption(file_path)

    source_language = "en"
    target_language = "es"  # Translate to spanish
    # translator = Translator(source_lang=source_language, target_lang=target_language)
    translated_caption =Translate_caption(caption)
    # translated_caption = translator.translate(caption)
    print("\nTranslated Caption:")
    print(translated_caption)

    return render_template('index.html', caption=caption,trans_caption=translated_caption, image_file=filename)

if __name__ == '__main__':
    app.run(debug=True)
