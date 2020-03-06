import flask
from flask import request, jsonify, render_template, sessions
import os
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == "POST":
        file = request.files["file"]
        file.save(os.path.json(app.config['UPLOAD_FOLDER'], file.filename))
        return render_template("index1.html", message = "success")
    return render_template("index1.html", message="Not Uploaded")


if __name__ == '__main__':
    app.run()