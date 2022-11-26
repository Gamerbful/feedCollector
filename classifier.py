import snowballstemmer
import elasticSearchTest as est
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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



def vectorize(corpus):
    vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 2))
    X2 = vectorizer2.fit_transform(corpus)  
    print(X2.toarray().shape)
    return X2.toarray()

vecFR = vectorize(corpusFR)
vecEN = vectorize(corpusEN)

def splitSet(vec,y):
    return train_test_split(  vec, y, test_size=0.33, random_state=42)

X_train_FR, X_test_FR, y_train_FR, y_test_FR = splitSet(vecFR,yFR)

X_train_EN, X_test_EN, y_train_EN, y_test_EN = splitSet(vecEN,yEN)

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
mlpFR = MLPClassifier(random_state=1,alpha=0.005, max_iter=1000, early_stopping=True)

neighEN = KNeighborsClassifier(n_neighbors=3)
lrEN = LogisticRegression(random_state=0)
gnbEN = GaussianNB()
# svcEN = make_pipeline(StandardScaler(), SVC(gamma='auto'))
# rcEN = RandomForestClassifier(max_depth=2, random_state=0)
mlpEN = MLPClassifier(random_state=1,alpha=0.005, max_iter=1000, early_stopping=True)

def predict(model,x):
    return model.predict(x)

modelsFR = [neighFR,lrFR,gnbFR,mlpFR]
modelsEN = [neighEN,lrEN,gnbEN,mlpEN]

def train(models,x_train,y_train):
    N = len(models)
    index = 1
    for model in models:
        print("{} / {} trained".format(index,N), end="\r")
        model.fit(x_train,y_train)
        index += 1


def test(models, X_test, y_test):
    for model in models:
        yPredict = predict(model,X_test)
        print(accuracy_score(y_test,yPredict))

train(modelsEN, X_train_EN, y_train_EN)
train(modelsFR, X_train_FR, y_train_FR)
test(modelsEN, X_test_EN, y_test_EN)
test(modelsFR, X_test_FR, y_test_FR)