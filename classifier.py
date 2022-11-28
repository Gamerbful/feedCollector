import elasticSearchTest as est
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import os.path
import pickle


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
        elif hit["_source"]["language"] == "en":
            corpusEN.append(hit["_source"]["data"])
            yEN.append(categories.index(hit["_source"]["Catégorie_du_flux"]))


print("EN doc count : {}".format(len(corpusEN)))
print("FR doc count : {}".format(len(corpusFR)))

def vectorize(corpus,language):
    vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 1))
    X2 = vectorizer2.fit_transform(corpus)  
    pickle.dump(vectorizer2,open("vec/vectorizer{}.pickle".format(language),"wb"))
    print(X2.toarray().shape)
    return X2.toarray(),vectorizer2.get_feature_names_out()

def splitSet(vec,y):
    return train_test_split(  vec, y, test_size=0.2, random_state=2)

if os.path.isfile("vec/vecEN.pickle"):
    print("loading vectorizer EN...")
    vecEN = pickle.load(open("vec/vecEN.pickle", "rb"))
    print("vectorizer FR loaded!")
    X_train_EN, X_test_EN, y_train_EN, y_test_EN = splitSet(vecEN,yEN)
    print("dataset EN created!")
else:
    vecEN,vecFeaturesEN = vectorize(corpusEN,"EN")
    np.save("vec/vecFeaturesEN.npy",vecFeaturesEN)
    X_train_EN, X_test_EN, y_train_EN, y_test_EN = splitSet(vecEN,yEN)
    print("Dataset EN created!")
    print("saving EN vectorizer...")
    pickle.dump(vecEN, open("vec/vecEN.pickle", "wb"))
    print("EN vectorizer saved!")

if os.path.isfile("vec/vecFR.pickle"):
    print("loading vectorizer FR...")
    vecFR = pickle.load(open("vec/vecFR.pickle", "rb"))
    print("vectorizer FR loaded!")
    X_train_FR, X_test_FR, y_train_FR, y_test_FR = splitSet(vecFR,yFR)
    print("dataset FR created!")
else:
    vecFR,vecFeaturesFR = vectorize(corpusFR,"FR")
    np.save("vec/vecFeaturesFR.npy",vecFeaturesFR)
    X_train_FR, X_test_FR, y_train_FR, y_test_FR = splitSet(vecFR,yFR)
    print("Dataset FR created!")
    print("saving FR vectorizer...")
    pickle.dump(vecFR, open("vec/vecFR.pickle", "wb"))
    print("FR vectorizer saved!")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform


# gnb {'var_smoothing': 1.2328467394420658e-05}
# knn {'n_neighbors': 1}
# lr {'C': 2.195254015709299, 'penalty': 'l1'}
#mlp {'solver': 'adam', 'learning_rate': 'constant', 'hidden_layer_sizes': (16,), 'alpha': 0.05, 'activation': 'tanh'}

neighFR = KNeighborsClassifier(n_neighbors=3)
lrFR = LogisticRegression(solver='saga', penalty="l2", C=2.195254015709299, tol=1e-2, max_iter=300,random_state=0)
gnbFR = GaussianNB(var_smoothing=1.2328467394420658e-05)
# svcFR = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# rcFR = RandomForestClassifier(max_depth=2, random_state=0)
mlpFR = MLPClassifier(random_state=1,alpha=0.0001,activation="relu", early_stopping=False, max_iter=100,tol=0.00001, verbose=1, hidden_layer_sizes=(128,))

neighEN = KNeighborsClassifier(n_neighbors=3)
lrEN = LogisticRegression(solver='saga', penalty="l2", C=2.195254015709299, tol=1e-2, max_iter=200,random_state=0)
gnbEN = GaussianNB(var_smoothing=1.2328467394420658e-05)
# svcEN = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# rcEN = RandomForestClassifier(max_depth=2, random_state=0)
mlpEN = MLPClassifier(random_state=1,alpha=0.0001,activation="relu", early_stopping=False, max_iter=100,tol=0.00001, verbose=1, hidden_layer_sizes=(128,))




def predict(model,x):
    return model.predict(x)

logisticDist = dict(
    C=uniform(loc=0, scale=4),
    penalty=['l2','l1']
)

mlpDist = dict(
    hidden_layer_sizes=[(128,),(64,32,16,),(32,16,),(16,)],
    activation= ["tanh","relu"],
    solver= ["sgd","adam"],
    alpha = [0.0001,0.05],
    learning_rate = ["constant", 'adaptive']
)

knnDist = dict(
    n_neighbors = [1,3,5,7,9,12,15,18,21]
)

gnbDist = dict(
    var_smoothing= np.logspace(0,-9, num=100)
)

modelsFR = [neighFR,lrFR,gnbFR,mlpFR]
modelsEN = [neighEN,lrEN,gnbEN,mlpEN]

modelsDist = [knnDist, logisticDist, gnbDist, mlpDist]


def bestParams(clf,dist,x,y):
    rs = RandomizedSearchCV(clf, dist, random_state=0, verbose=3)
    rs.fit(x,y)
    return rs.best_params_

def PrintBestParams(models,dists,x,y):
    for i in range(len(models)):
        print(bestParams(models[i],dists[i],x,y))

# print(bestParams(neighFR,knnDist,X_train_EN, y_train_EN))
# print(bestParams(mlpFR,mlpDist,X_train_EN, y_train_EN))

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

vectorizerFR = pickle.load( open("vec/vectorizerFR.pickle", "rb"))

# print(vectorizerFR.transform([corpusFR[0]]))
