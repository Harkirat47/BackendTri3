from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource
from model.places import ImageClassifier
import sqlite3
import base64
from PIL import Image
from io import BytesIO
import urllib.request

places_api = Blueprint('places_api', __name__, url_prefix='/api/places')
api = Api(places_api)

# Initialize the model
places_model = ImageClassifier('./places')
places_model.train_model()

class ImageApi:
    class Upload(Resource):
        def post(self):
            try:
                data = request.get_json()
                image_data = data.get('image')

                conn = sqlite3.connect('sqlite.db')
                cursor = conn.cursor()

                cursor.execute('''CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, image_data BLOB)''')
                cursor.execute("INSERT INTO images (image_data) VALUES (?)", (image_data,))
                conn.commit()
                conn.close()

                return jsonify({'message': 'Image uploaded successfully'})
            except Exception as e:
                return jsonify({'error': str(e)}), 500

    class Predict(Resource):
        def post(self):
            try:
                ## database related code, for future expansion to database
                """conn = sqlite3.connect('sqlite.db')
                cursor = conn.cursor()

           
                cursor.execute("SELECT image_data FROM images ORDER BY id DESC LIMIT 1")
                image_data = cursor.fetchone()[0]
                conn.close()"""
                ## get data from frontend as URL to image
                data = request.get_json()
                ## PIL will automatically interpret base64 data if wrapped properly
                ## hence supports base64 as well as URLs
                ## certain file types in URLs, such as webp will return null
                url = data["image_data"]
                with urllib.request.urlopen(url) as url:
                    ## open image and get data by sending a request
                    with open('temp.jpg', 'wb') as f:
                        f.write(url.read())
                
                ## convert image to PIL Image
                image = Image.open('temp.jpg')
                ## get prediction
                predictions = places_model.predict_image_class(image)
                ## prediction comes in the form of a number, depending on what image it is
                ## prediction must be converted to text form to return using dictionary below
                maps = {"4": "Big Ben",
                        "2": "Burj Khalifa",
                        "8": "Christ the Redeemer",
                        "5": "Eiffel Tower",
                        "6": "Golden Gate Bridge",
                        "1": "Leaning Tower of Pisa",
                        "3": "Macchu Picchu",
                        "9": "Osaka Castle",
                        "0": "Statue of Liberty",
                        "7": "Pyramids of Giza"
                        }
                ## return text based on prediction
                return jsonify({'predictions': maps[str(predictions)]})
            except Exception as e:
                print(e)

# Add resources to API
api.add_resource(ImageApi.Upload, '/upload')
api.add_resource(ImageApi.Predict, '/predict')