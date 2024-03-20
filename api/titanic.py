from flask import Blueprint, request, jsonify
from flask_restful import Api, Resource # import API package
from model.titanics import TitanicRegression

predict_api = Blueprint('predict_api', __name__)

titanic_api = Blueprint('titanic_api', __name__,
                   url_prefix='/api/titanic') # initialize titanic api url

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(titanic_api)

# Initialize the model
titanic_model = TitanicRegression()

class TitanicApi:        
    class _CRUD(Resource): 
        def post(self):
            data = request.get_json()
            alive_prob = titanic_model.predict(data)
            return jsonify({'survival_probability': alive_prob})

    api.add_resource(_CRUD, '/')