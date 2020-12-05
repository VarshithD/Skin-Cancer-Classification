import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow import keras
import tensorflow 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import preprocess_input


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert inputted image to an array
def process_image(file_path):
	print(file_path)
	image_array = []
	image_prep = keras.preprocessing.image.load_img(file_path, target_size = (128, 128, 3))
	image_prep = keras.preprocessing.image.img_to_array(image_prep)
	image_prep = preprocess_input(image_prep)
	image_array.append(image_prep)
	return np.array(image_array)

def get_class(row):
	for c in row:
		if row[c] == 1:
			return c

# Load model and make prediction
def get_class_prediction(image_array):
	base_model = tensorflow.keras.applications.mobilenet.MobileNet()
	x = base_model.layers[-6].output
	x = Dropout(0.3)(x)
	predictions = Dense(5, activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)
	for layer in base_model.layers[:-23]:
		layer.trainable = False
	

	
	model.compile(Adam(lr=0.01), loss='categorical_crossentropy',metrics=[categorical_accuracy])
	model.load_weights('model/Model.hdf5')
	classes = {
        0 : 'Basal Cell Carcinoma',        
        1 : 'Seborrhoeic keratosis',
        2 : 'Solar lentigo',
        3 : 'Squamous cell carcinoma',
        4 : 'melanoma',
    }
	class_index = model.predict(image_array)
	class_req = max(class_index[0])
	for c in range(0,5):
		if class_index[0][c] == class_req:
			class_re = c
	print(class_index[0])
	print(class_re)
	return classes[class_re]

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        f = request.files['file']
        if f.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if f and allowed_file(f.filename):
            image_name = f.filename
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'static', secure_filename(f.filename))
            f.save(file_path)

            image = process_image(file_path)
            class_name = get_class_prediction(image).capitalize()
            return render_template('upload.html', label = class_name, img = image_name)
    return

if __name__ == '__main__':
    app.run(debug=True)