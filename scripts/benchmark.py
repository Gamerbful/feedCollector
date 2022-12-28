import json
from xml.dom import minidom
import pickle
import snowballstemmer
from stop_words import get_stop_words
from joblib import dump, load
import sys

import numpy as np
def readXMLFile(fname):
    # convert an XML file (fname) into a list of dicts
    data = []
    en_db = minidom.parse(fname)
    items = en_db.getElementsByTagName('item')
    for i in range(len(items)):
        d={}
        title=items[i].getElementsByTagName('title')[0].childNodes[0].data
        desc=items[i].getElementsByTagName('description')[0].childNodes[0].data
        txt=items[i].getElementsByTagName('text')[0].childNodes[0].data
        d['title']=title
        d['description']=desc
        d['text']=txt
        data.append(d)
    return data


vectorizerFR = pickle.load( open("vec/vectorizerFR.pickle", "rb"))
vectorizerEN = pickle.load( open("vec/vectorizerEN.pickle", "rb"))

bestModelFR = load("model/bestModelFR.joblib")
bestModelEN = load("model/bestModelEN.joblib")

categories = np.array(["BUSINESS","SPORT","HEALTH","ART","SCIENCE","POLITIC"])


benchmarkCategories = ['ART_CULTURE', 'ECONOMIE', 'POLITIQUE', 'SANTE_MEDECINE', 'SCIENCE', 'SPORT']

indexes = [3,0,5,2,4,1]


def launch(language):
    vectorizer = vectorizerEN if language == "en" else vectorizerFR
    model = bestModelEN if language == "en" else bestModelFR
    lg = "english" if language == "en" else "french"
    data = readXMLFile("benchmark/benchmark_{}.xml".format(language))
    stemmer = snowballstemmer.stemmer(lg)
    stop_words = get_stop_words(language)

    predicted = []
    probas = []
    N = len(data)
    index = 1
    for doc in data:
        print("{} / {} docs".format(index,N), end='\r')
        text = doc['text']
        text_lower = [ w.lower() for w in text.split() ]
        stemWord = stemmer.stemWords(text_lower)
        filtered_sentence = [w for w in stemWord if not w in stop_words]
        X = vectorizer.transform([" ".join(filtered_sentence)])
        cat = benchmarkCategories[indexes.index(model.predict(X)[0])]
        proba = model.predict_proba(X)
        predicted.append(cat)
        probas.append(proba.tolist()[0])
        index += 1

    res=dict()
    res['pred']=list(predicted)
    res['probs']=list(probas)
    res['names']=['CAPPELLE','ROBIN']
    res['method']="MLP"
    res['lang']=language
    file = open("CAPPELLE_ROBIN_MLP_"+language+".res","w")
    file.write(json.dumps(res))
    file.close()


def main():
    if len(sys.argv) < 2:
        print("Usage : add argument in command line to proceed benchmark ( en | fr )")
    if sys.argv[1] not in ["en","fr"]:
        print("Usage : Available args are : en / fr")
    else:
        launch(sys.argv[1])


if __name__ == "__main__":
    main()