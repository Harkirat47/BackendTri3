from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource # import API package
from model.places import ImageClassifier

places_api = Blueprint('places_api', __name__,
                   url_prefix='/api/places') # initialize titanic api url

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(places_api)

# Initialize the model
titanic_model = ImageClassifier()

class ImageApi:        
    class _CRUD(Resource): 
        def post(self):
            data = request.get_json()
            alive_prob = titanic_model.predict(data)
            return jsonify({'survival_probability': alive_prob})

    api.add_resource(_CRUD, '/')