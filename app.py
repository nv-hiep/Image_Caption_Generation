import os
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename

import numpy as np
import pickle

from PIL import Image

from keras.models import model_from_json
# from keras.preprocessing import image

from tensorflow.keras.models import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Initiate the app
app = Flask(__name__)

# For session
app.secret_key = 'adahsdusagsagdakdi7quiwuwiqiuqiuqjasbd978dy698d709udjadjka'
app.config['SESSION_TYPE'] = 'filesystem'


# For uploading images
UPLOAD_FOLDER = 'static/img'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 0.5 * 1024 * 1024                # 0.5MB app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  


# Model Parameters
# The input size and output of the InceptionV3 network
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
max_length = 34

START  = 'startseq'
STOP   = 'endseq'

# Model
encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)

# load json and create model
json_file = open('model/Image_Caption_model.json', 'r')
caption_model = json_file.read()
json_file.close()
caption_model = model_from_json(caption_model)

# load weights into new model
caption_model.load_weights("model/caption-model.hdf5")


with open('model/wordtoidx.pkl', 'rb') as fp:
  wordtoidx = pickle.load(fp)


with open('model/idxtoword.pkl', 'rb') as fp:
  idxtoword = pickle.load(fp)



# We need to encode the images to create the training set.
# Later we will encode new images to present them for captioning.
def encode_image(img):
  # Resize all images to the size specified by the network
  img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
  
  # Convert image to a numpy array
  x = image.img_to_array(img)
  
  # Expand to 4D array
  x = np.expand_dims(x, axis=0) # (1,299,299,3)
  
  # Perform any preprocessing needed by InceptionV3
  x = preprocess_input(x)
  
  # Call InceptionV3 to extract the smaller feature set for the image.
  x = encode_model.predict(x) # Get the encoding vector for the image
  
  # Reshape to correct form to be accepted by LSTM captioning network.
  x = np.reshape(x, OUTPUT_DIM) # Vector of size (2048,)
  
  return x




def generate_caption(photo):
    in_text = START
    for i in range(max_length):
        sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = caption_model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idxtoword[yhat]
        in_text += ' ' + word
        if word == STOP:
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)

    return final






def is_allowed_file(filename: str) -> bool:
    """Check if the file is allowed

    Args:
        filename ([string]): [String of Filename]

    Returns:
        [boolean]: [True/False]
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def del_items() -> None:
    """Delete all image files in UPLOAD FOLDER

    Returns:
        [None]
    """
    for x in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, x))
    return None






def img_captioning(img_list: list) -> dict:
    """Generate the Image Captions

    Args:
        img_list ([list]): [List of image filenames]

    Returns:
        [dict]: [Dictionary of image filenames and their captions]
    """
    ret = {}
    for xfile in img_list:
        imgx = image.load_img(
            os.path.join(app.config['UPLOAD_FOLDER'], xfile),
            target_size=(HEIGHT, WIDTH, 3)
        )

        imgx = encode_image(imgx).reshape((1,OUTPUT_DIM)) # A vector (1,2048)
        ret[xfile] = generate_caption(imgx)
    
    return ret



# HTML
TITLE = "Image Captioning With Encoder-Decoder"
FRAME_TITLE = "Image Captioning With Encoder-Decoder"
PURPOSE_TITLE = "Generate Image Caption using Encoder-Decoder model"
GITHUB_LINK = "https://github.com/nv-hiep/DeepLearning_Projects/tree/main/Image_Caption_Generator"


@app.route('/')
def home():
    del_items()
    return render_template("index.html",
                           TITLE =TITLE,
                           FRAME_TITLE = FRAME_TITLE,
                           PURPOSE_TITLE = PURPOSE_TITLE,
                           GITHUB_LINK = GITHUB_LINK)





@app.route("/captioning", methods = ["GET", "POST"])
def captioning():
    if request.method == "POST":

        # check if the post request has the file part
        if 'files[]' not in request.files:
            flash("No file part! Add file input to the form.", "danger")
            return redirect(url_for('home')) # or return redirect(request.url) if want to go back /predict
        
        # file = request.files['file']
        files = request.files.getlist('files[]')
        if len(files) == 1 and files[0].filename == '':
            flash("No selected file! Please choose file(s) to upload.", "danger")
            return redirect(request.url)
        
        if len(files) > 3:
            flash("Too many files! Please select 3 files at max", "danger")
            return redirect(request.url)

        # MAX_CONTENT_LENGTH
        sizes = np.array([len(file.read()) for file in files])
        if np.any(sizes > MAX_CONTENT_LENGTH):
            flash("Files are too large! Please select files with size < " + str(MAX_CONTENT_LENGTH/1024./1024.) + ' MB', "danger")
            return redirect(request.url)

        filenames = []
        for file in files:
            if file and is_allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.seek(0)  # Need to have this to avoid files with 0 bytes
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                filenames.append(filename)
        
        msg = "Success! Files uploaded!" if len(files) > 1 else "Success! File uploaded!"
        flash(msg, "success")

        results = img_captioning(filenames) 

        return render_template(
            "captioning.html",
            filenames = filenames,
            results = results,
            TITLE =TITLE,
            FRAME_TITLE = FRAME_TITLE,
            PURPOSE_TITLE = PURPOSE_TITLE,
            GITHUB_LINK = GITHUB_LINK)
    
    
    return render_template("index.html",
                           TITLE =TITLE,
                           FRAME_TITLE = FRAME_TITLE,
                           PURPOSE_TITLE = PURPOSE_TITLE,
                           GITHUB_LINK = GITHUB_LINK)









@app.route('/plot')
def plot():
    data = [
    ("01-01-2020", 1597),
    ("02-01-2020", 1465),
    ("03-01-2020", 1908),
    ("04-01-2020", 896),
    ("05-01-2020", 755),
    ("06-01-2020", 423),
    ("07-01-2020", 1100),
    ("08-01-2020", 1235),
    ("09-01-2020", 1536),
    ("10-01-2020", 1498),
    ("11-01-2020", 1623),
    ("12-01-2020", 2121)
    ]

    x = [row[0] for row in data]
    y = [row[1] for row in data]

    return render_template("graph.html", labels=x, values=y)

if __name__ == '__main__':
    app.debug = True
    app.run()