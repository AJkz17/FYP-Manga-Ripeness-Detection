from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    password = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), default='user')

class DetectionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    temperature = db.Column(db.Float, default=0.0)
    humidity = db.Column(db.Float, default=0.0)
    days_remaining = db.Column(db.String(50), default="N/A")
    harvest_msg = db.Column(db.String(200), default="")

    def __repr__(self):
        return f'<Record {self.id} - {self.prediction}>'