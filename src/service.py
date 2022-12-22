"""
----------------------------
Web-service with API
----------------------------
"""
from email import message
import os
import unicodedata


import numpy as np
from flask import Flask, render_template, request
from flask_restful import Resource, Api, reqparse

from src.utils import conf, logger, MessagesDB, ML

from tensorflow.keras.models import model_from_json

db = MessagesDB(conf)
db.init_db()

def load_model(model_path, weights_path):
    # load json and create model
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    logger.info("Loaded model from disk")
    return loaded_model


app = Flask(__name__)
api = Api(app)
model = load_model(conf.model_path, conf.weights_path)


@app.route('/messages/<string:identifier>')
def predict_label(identifier):
    msg = db.read_message(msg_id=int(identifier))

    # model predict single label
    pred = model.predict(ML.preprocessing(np.array( [msg] )))
    predicted_label = pred[0]

    return render_template('page.html', id=identifier, txt=msg['txt'], label=predicted_label)

@app.route('/feed/')
def feed():
    limit = request.args.get('limit', 10)
    limit = int(limit)
    # rank all messages and predict
    msg_ids = db.get_messages_ids(limit)

    messages = []

    for id in msg_ids:
        msg = db.read_message(msg_id=int(id)) 
        # model predict single label
        pred = model.predict(ML.preprocessing(np.array( [msg] )))
        predicted_label = pred[0]

        message = {}
        message['msg_id'] = id
        message['msg_txt'] = msg
        message['msg_pred'] = predicted_label

        messages.append(message)

    sorted_messages = sorted(messages, key=lambda d: d['msg_pred'], reverse = True) 

    return render_template('feed.html', recs=sorted_messages)

class Messages(Resource):
    def __init__(self):
        super(Messages, self).__init__()
        self.msg_ids = db.get_messages_ids()  # type: list

    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('limit', type=int, default=10, location='args')
        args = parser.parse_args()
        try:
            resp = [int(i) for i in np.random.choice(self.msg_ids, size=args.limit, replace=False)]
        except ValueError as e:
            resp = 'Error: Cannot take a larger sample than %d' % len(self.msg_ids)
        return {'msg_ids': resp}


api.add_resource(Messages, '/messages')
logger.info('App initialized')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
