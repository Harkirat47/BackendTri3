""" database dependencies to support sqliteDB examples """
from random import randrange
from datetime import date
import os, base64
import json

from __init__ import app, db
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash

from __init__ import app, db
import os
import base64


class Image(db.Model):
    __tablename__ = "images"
    id = db.Column(db.Integer, primary_key=True)
    _name = db.Column(db.String(255), unique=False, nullable=False)
    _place = db.Column(db.String, unique=False)

    def __init__(self, name, place):
        self._name = name
        self._place = place

    @property
    def name(self):
        return self._name
    
    @property
    def place(self):
        return self._place
    
    @place.setter
    def place(self, place):
        self._place = place

    def create(self, image_data):
        try:
            path = app.config['UPLOAD_FOLDER']
            # Generate a unique filename
            filename = str(self.id) + '_' + self._name + '.jpg'
            output_file_path = os.path.join(path, filename)
            # Decode base64 image data and save it to the file
            with open(output_file_path, 'wb') as output_file:
                output_file.write(base64.b64decode(image_data))

            # Update the place attribute with the filename
            self._place = filename
            db.session.add(self)
            db.session.commit()
            return self
        except Exception as e:
            db.session.rollback()
            print(f"Error: {e}")
            return None

    def read(self):
        try:
            path = app.config['UPLOAD_FOLDER']
            file = os.path.join(path, self._place)
            with open(file, 'rb') as file:
                file_encode = base64.encodebytes(file.read())
            return {
                "id": self.id,
                "name": self.name,
                "place": self._place,
                "base64": str(file_encode),
            }
        except Exception as e:
            print(f"Error: {e}")
            return None
