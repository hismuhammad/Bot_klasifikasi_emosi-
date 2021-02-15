from flask import Flask, request, abort

from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
from keras.models import model_from_json
import os
from mainApi.engine.response import get_emotionn

app = Flask(__name__)

line_bot_api = LineBotApi('ypJtd6gIJIPwRUp8JYqIIt1XXN40pdMBCF/7KwEs8SJBKggnZJxL1D+v/0IqvC1530AHHW6+lJLtM/hoOXNgQv/PNU/WZ+cTmIaD5VrHkqKtfG27b+q3wUPieUxiys3YBSvdfO+VSAaux8nehI9apwdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('0749389a230f79e07064b3e50fe76cb3')

@app.route("/")
def index():
    return "harusnya jalan, OK"

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Please check your channel access token/channel secret.")
        abort(400)

    return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    alamat = os.path.dirname(os.path.abspath(__file__))
    print(alamat)
    json_file = open(alamat + "/mainApi/engine/model.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(alamat + "/mainApi/engine/model.h5")
    text = event.message.text
    res = get_emotionn(loaded_model,text)
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=res))

if __name__ == "__main__":
    app.run(port=8000)
