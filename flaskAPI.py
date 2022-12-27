from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

import w2v as w
# @app.route('/',methods=['GET', 'POST',])
# def hello_world():
#     res = w.search("macron")
#     print(res)
#     return render_template('index.html', embed=res)

@app.route('/research',methods=['GET','POST'])
@cross_origin()
def research():
    if request.method == "POST":
        query = request.json['fname']
        cat,docs = w.search(query)
        print(jsonify(cat,docs))
        return jsonify(cat,docs)


app.run(debug=True)