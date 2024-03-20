import os
from flask import Blueprint, request, jsonify, current_app
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename
from images.model import Image  # Importing Image model
from images.model import db  # Importing db from SQLAlchemy setup file

image_api = Blueprint('image_api', __name__, url_prefix='/api/image')
api = Api(image_api)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ImageAPI:        
    class _CRUD(Resource):
        def post(self):
            if 'file' not in request.files:
                return {'message': 'No file part'}, 400

            file = request.files['file']

            if file.filename == '':
                return {'message': 'No selected file'}, 400

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))

                # Create Image object
                description = request.form.get('description')

                image = Image(name=name, place=place)  # Creating Image object
                image.description = description
            
                # Saving image to database
                try:
                    db.session.add(image)
                    db.session.commit()
                    return {'message': 'Image uploaded successfully'}, 201
                except Exception as e:
                    db.session.rollback()
                    return {'message': 'Error uploading image'}, 500
            else:
                return {'message': 'File type not allowed'}, 400

    api.add_resource(_CRUD, '/')
