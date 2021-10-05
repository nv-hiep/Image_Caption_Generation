# Image Captioning With Encoder-Decoder

## Image Captioning With Encoder-Decoder
- Generate Image Caption using Encoder-Decoder model
- Takes an image as input and generates a short text describing the content of the image.
- Build a webapp with Flask
- Deploy the Model With Flask on Heroku

## How can we train the model:
- Image --> CNN model (InceptionV3) -> Extracted Feature Vector
- Extracted Feature Vector -> DropOut/Dense/ReLU activation --> Vector (X1) --> Input_1
- Caption text -> tokenizer --> pad_sequences --> Embedding -> DropOut -> LSTM layer (X2) --> Input_2
- X1 + X2 --> Dense layer/ReLu_Activation --> Dense layer/Softmax_activation (decoder) --> outputs
- Model(inputs=[inputs1, inputs2], outputs=outputs)

1. Data:
   - Training dataset: Flickr8k Dataset
   - The dataset includes:
        - Images (https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip)
        - Captions (https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip)
2. Run the notebook: Image_Caption_Generation.ipynb:
   - Data Cleaning
   - Train a encoder-decoder model for image captioning
   - Generate the caption using an image as the input
   - Save the model and the model weight (Image_Caption_model.json and caption-model.hdf5) in directory /model
4. Buid a webapp with Flask:
   - Install Flask (https://linuxize.com/post/how-to-install-flask-on-ubuntu-20-04/)
   - To import numpy, pandas etc... Need to be inside (venv) of Flask. then:  pip3 install numpy,  pip3 install scikit-learn,  pip3 install pandas etc...
   - cd to the directory that contains "app.py"
   - Run the "app.py" (e.g: python3 app.py)
   - Open http://127.0.0.1:5000/ on localhost in browser.
5. Upload to Github
   - Upload the project to Github
   - Include the files: Procfile, requirements.txt, runtime.txt
6. Deploy the App on Heroku
   - Signup for an account on Heroku
   - Create a new app
   - Connect to Github in "Deployment method"
   - Deploy the App (click on "Deploy Branch")
   - Check the app via: https://dl-flask-image-captioning.herokuapp.com/