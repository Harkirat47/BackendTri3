import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
import pandas as pd

from model.model import *

model_api = Blueprint('model_api', __name__,
                   url_prefix='/api/titanic')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(model_api)
class TitanicAPI:        
    class _Titanic(Resource):
        def post(self):
            body = request.get_json()
            name = body.get('name')
            pclass = body.get('pclass')
            sex = body.get('sex')
            age = body.get('age')
            fmem = body.get('fmem')
            fare = body.get('fare')
            embark = body.get('embark')
            passenger = pd.DataFrame({
                'name': [name],
                'pclass': [pclass],
                'sex': [sex],
                'age': [age],
                'sibsp': [fmem], 
                'parch': [fmem], 
                'fare': [fare], 
                'embarked': [embark], 
                'alone': [True if fmem == 0 else False]
            })
            initTitanic()
            return predictSurvival(passenger)
    api.add_resource(_Titanic, '/')

