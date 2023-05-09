from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image
app = Flask(__name__)

model = load_model('vgg_model.h5')
model.save('vgg_model.h5')


target_img = os.path.join(os.getcwd() , 'C:\\Users\\Admin\\Desktop\\Animal Image Classification Project\\imageload\\static\\raw-img')
@app.route('/')
def index_view():
    return render_template('myimageload.html')
#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        print("111")
        file = request.files['file']
        
        #if file and allowed_file(file.filename): #Checking file format
            #filename = file.filename
        print("file is",file.filename)
        
        file_path = os.path.join('C:\\Users\\Admin\\Desktop\\Animal Image Classification Project\\imageload\\static\\raw-img', file.filename)
        file.save(file_path)
        img = read_image(file_path) #prepressing method
        #img=img.reshape(-1,180,180,3)
        prediction=model.predict(img)[0]
        print("prediction is", prediction)

           
            #'fruit' , 'prob' . 'user_image' these names we have seen in predict.html.
        return render_template('C:\\Users\\Admin\\Desktop\\Animal Image Classification Project\\imageload\\imageload\\myimageload.html', prediction_text = 'The item is a {}'. format(prediction))
        
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)