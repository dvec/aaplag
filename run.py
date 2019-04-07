import requests
import json
from flask import Flask, request, render_template
from nn import gpt2_text_transform, download_gpt2_model


app = Flask(__name__)
tkey = "trnsl.1.1.20190406T120126Z.f880cac48e1e6bdb.ce1ea65ddd619cac01e53e80bead80c5edf56401"
turl = "https://translate.yandex.net/api/v1.5/tr.json/translate"


def translate_text(text, lang):
    return json.loads(requests.get(turl, params={'key': tkey, 'lang': lang, 'text': text}).text)['text'][0]


@app.route('/', methods=['GET', 'POST'])
def mainpage():
    if request.method == 'POST':
        input = request.form.get('input')
        output = translate_text(input, 'ru-en')
        output = gpt2_text_transform(output)
        output = translate_text(output, 'en-ru')
        return render_template("main.html", input=input, output=output)
    else:
        return render_template("main.html")


if __name__ == '__main__':
    download_gpt2_model()
    app.run(host='0.0.0.0', port=80)
