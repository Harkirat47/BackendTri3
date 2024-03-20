import tensorflow as tf
from tensorflow.keras import layers, models
from random import randrange
from datetime import date
import os, base64
import json

from __init__ import app, db
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
import os
import base64

# Assuming `db` is your Flask-SQLAlchemy instance
class Image(db.Model):
    __tablename__ = 'images'

    # Define the Image schema
    id = db.Column(db.Integer, primary_key=True)
    image_data = db.Column(db.Text, nullable=False)

    # Constructor of an Image object
    def __init__(self, image_data):
        self.image_data = image_data

    # Returns a string representation of the Image object
    def __repr__(self):
        return f"Image(id={self.id})"

    # CRUD create, adds a new record to the Image table
    def create(self):
        try:
            db.session.add(self)
            db.session.commit()
            return self
        except IntegrityError:
            db.session.rollback()
            return None

    # CRUD read, returns dictionary representation of Image object
    def read(self):
        return {
            "id": self.id,
            "image_data": self.image_data
        }

    db.session.commit()

class ImageClassifier:
    def __init__(self, data_path, epochs=30):
        self.data_path = data_path
        self.epochs = epochs
        self.data = None
        self.model = None

    def load_data(self):
        self.data = tf.keras.utils.image_dataset_from_directory(
            self.data_path,
            image_size=(256, 256),
            batch_size=32,
            validation_split=0.2,
            subset="training",
            seed=42,
        )

    def build_model(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(10, activation='softmax')  # 10 classes
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train_model(self):
        if self.data is None:
            print("Data not loaded. Please call load_data() first.")
            return

        train_size = int(len(self.data) * 0.7)
        val_size = int(len(self.data) * 0.2)

        train_data = self.data.take(train_size)
        val_data = self.data.skip(train_size).take(val_size)

        hist = self.model.fit(train_data, epochs=self.epochs, validation_data=val_data)

    def train(self):
        self.load_data()
        self.build_model()
        self.train_model()

# Example usage:
if __name__ == "__main__":
    classifier = ImageClassifier('/Users/shubhay/Documents/GitHub/BackendTri3/places')
    classifier.train()
