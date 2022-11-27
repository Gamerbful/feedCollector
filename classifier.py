import elasticSearchTest as est
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import os.path

categories = ["BUSINESS","SPORT","HEALTH","ART","SCIENCE"]

res = est.getDocuments("rss",{"match_all": {} })
corpusFR = []
yFR = []

corpusEN = []
yEN = []

hits = res["hits"]["hits"]
size = len(hits)

for cat in categories:
    res = est.getDocuments("rss",{"match": {
        "Catégorie_du_flux":cat
    } })
    print("{} : {}".format(cat,len(res["hits"]["hits"])))


    
for hit in hits:
    if hit["_source"]["data"] != None:
        if hit["_source"]["language"] == "fr":      
            corpusFR.append(hit["_source"]["data"])
            yFR.append(categories.index(hit["_source"]["Catégorie_du_flux"]))
        else:
            corpusEN.append(hit["_source"]["data"])
            yEN.append(categories.index(hit["_source"]["Catégorie_du_flux"]))


print("EN doc count : {}".format(len(corpusEN)))
print("FR doc count : {}".format(len(corpusFR)))

def vectorize(corpus):
    vectorizer2 = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))
    X2 = vectorizer2.fit_transform(corpus)  
    print(X2.toarray().shape)
    return X2.toarray(),vectorizer2.get_feature_names_out()

def splitSet(vec,y):
    return train_test_split(  vec, y, test_size=0.33, random_state=42)

if os.path.isfile("dataset/xtestEN.npy"):
    print("loading dataset EN...")
    X_train_EN = np.load("dataset/xtrainEN.npy")
    X_test_EN = np.load("dataset/xtestEN.npy")
    y_train_EN = np.load("dataset/ytrainEN.npy")
    y_test_EN = np.load("dataset/ytestEN.npy")
    print("dataset EN loaded!")
else:
    vecEN,vecFeaturesEN = vectorize(corpusEN)
    np.save("vec/vecFeaturesEN.npy",vecFeaturesEN)
    X_train_EN, X_test_EN, y_train_EN, y_test_EN = splitSet(vecEN,yEN)
    print("Dataset EN created!")
    np.save("dataset/xtrainEN.npy",X_train_EN)
    np.save("dataset/xtestEN.npy",X_test_EN)
    np.save("dataset/ytrainEN.npy",y_train_EN)
    np.save("dataset/ytestEN.npy",y_test_EN)
    print("dataset EN saved!")

if os.path.isfile("dataset/xtestFR.npy"):
    print("loading dataset FR...")
    X_train_FR = np.load("dataset/xtrainFR.npy")
    X_test_FR = np.load("dataset/xtestFR.npy")
    y_train_FR = np.load("dataset/ytrainFR.npy")
    y_test_FR = np.load("dataset/ytestFR.npy")
    print("dataset FR loaded!")
else:
    vecFR,vecFeaturesFR = vectorize(corpusFR)
    np.save("vec/vecFeaturesFR.npy",vecFeaturesFR)
    X_train_FR, X_test_FR, y_train_FR, y_test_FR = splitSet(vecFR,yFR)
    print("Dataset FR created!")
    print("Saving dataset...")
    np.save("dataset/xtrainFR.npy",X_train_FR)
    np.save("dataset/xtestFR.npy",X_test_FR)
    np.save("dataset/ytrainFR.npy",y_train_FR)
    np.save("dataset/ytestFR.npy",y_test_FR)
    print("dataset FR saved!")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

neighFR = KNeighborsClassifier(n_neighbors=3)
lrFR = LogisticRegression(random_state=0)
gnbFR = GaussianNB()
# svcFR = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# rcFR = RandomForestClassifier(max_depth=2, random_state=0)
mlpFR = MLPClassifier(random_state=1,alpha=0.0005, max_iter=200, early_stopping=True,batch_size=256)

neighEN = KNeighborsClassifier(n_neighbors=3)
lrEN = LogisticRegression(random_state=0)
gnbEN = GaussianNB()
# svcEN = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# rcEN = RandomForestClassifier(max_depth=2, random_state=0)
mlpEN = MLPClassifier(random_state=1,alpha=0.0005, max_iter=200, early_stopping=True,batch_size=256)

def predict(model,x):
    return model.predict(x)

modelsFR = [neighFR,lrFR,gnbFR,mlpFR]
modelsEN = [neighEN,lrEN,gnbEN,mlpEN]

from joblib import dump, load


def train(models,x_train,y_train):
    N = len(models)
    index = 1
    for model in models:
        print("{} / {} in training".format(index,N), end="\r")
        model.fit(x_train,y_train)
        index += 1
    print("TRAINING SUCCESSFULL")


def test(models, X_test, y_test):
    bestModel = None
    bestAcc = 0.0
    for model in models:
        yPredict = predict(model,X_test)
        acc = accuracy_score(y_test,yPredict)
        if acc>bestAcc:
            bestModel = model
            bestAcc = acc
        print(acc)
    return bestModel

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def printMetrics(model, X_test, y_test, language):
    modelName = type(model).__name__
    yPredict = model.predict(X_test)
    yPredictProba = model.predict_proba(X_test)
    acc = accuracy_score(y_test,yPredict)
    f1 = f1_score(y_test, yPredict, average="weighted")
    roc = roc_auc_score(y_test, yPredictProba,multi_class='ovr')
    print("{} {} accuracy : {}".format(modelName, language, acc))
    print("{} {} f1 score : {}".format(modelName, language, f1))
    print("{} {} roc_auc score : {}".format(modelName, language, roc))
    return confusion_matrix(y_test,yPredict)


import time
def execTimeForOneDoc(model,doc):
    start_time = time.time()
    model.predict(doc)
    print("exec time for one doc --- {} seconds ---".format(time.time() - start_time))

if os.path.isfile("model/bestModelEN.joblib"):
    print("loading best model EN...")
    bestModelEN = load("model/bestModelEN.joblib")
    print("best model EN loaded!")
else:
    train(modelsEN, X_train_EN, y_train_EN)
    bestModelEN = test(modelsEN, X_test_EN, y_test_EN)
    dump(bestModelEN,"model/bestModelEN.joblib")

if os.path.isfile("model/bestModelFR.joblib"):
    print("loading best model FR...")
    bestModelFR = load("model/bestModelFR.joblib")
    print("best model FR loaded!")
else:
    train(modelsFR, X_train_FR, y_train_FR)
    bestModelFR = test(modelsFR, X_test_FR, y_test_FR)
    dump(bestModelFR,"model/bestModelFR.joblib")




cm1 = printMetrics(bestModelEN, X_test_EN, y_test_EN, "EN")
execTimeForOneDoc(bestModelEN, [X_test_EN[0]])
cm2 = printMetrics(bestModelFR, X_test_FR, y_test_FR, "FR")
execTimeForOneDoc(bestModelFR, [X_test_FR[0]])



import seaborn as sns
from matplotlib import pyplot as plt

plt.figure()
cm1_plot = sns.heatmap(cm1/np.sum(cm1), annot=True, xticklabels=categories, yticklabels=categories)
plt.figure()
cm2_plot = sns.heatmap(cm2/np.sum(cm2), annot=True, xticklabels=categories, yticklabels=categories)

cm1_plot.figure.savefig("img/cmEN.png",dpi=400)
cm2_plot.figure.savefig("img/cmFR.png",dpi=400)