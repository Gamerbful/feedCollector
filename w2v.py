from gensim.models import Word2Vec
import elasticSearchTest as est
client = est.es_client
from gensim.models import Phrases
import snowballstemmer
import pickle
import numpy as np




res = est.getDocuments("rss",{"match_all": {} })

corpusFR = []
corpusEN = []

hits = res["hits"]["hits"]
size = len(hits)

count = 0
for hit in hits:
    if hit["_source"]["data"] != None:
        if hit["_source"]["language"] == "fr":      
            corpusFR.append(hit["_source"]["data"].split())
        elif hit["_source"]["language"] == "en":
            corpusEN.append(hit["_source"]["data"].split())
        else:
            count +=1

print(count)
		

# bigramFR = Phrases(corpusFR)
# bigramEN = Phrases(corpusEN)

#  embedding dim =~ 1.6 times the square root of the number of unique elements in the category, and no less than 600.



# modelFR = Word2Vec(corpusFR, vector_size=320, window=7, min_count=1, workers=4)
# modelEN = Word2Vec(corpusEN, vector_size=320, window=7, min_count=1, workers=4)

# modelFR.save("model/word2vecFR.model")
# modelEN.save("model/word2vecEN.model")


modelFR = Word2Vec.load("model/word2vecFR.model")
modelEN = Word2Vec.load("model/word2vecEN.model")
from joblib import dump, load

vectorizerFR = pickle.load( open("vec/vectorizerFR.pickle", "rb"))
bestModelFR = load("model/bestModelFR.joblib")

categories = ["BUSINESS","SPORT","HEALTH","ART","SCIENCE","POLITIC"]

def getBestSimilarities(model, word, language,topn=10):
	try:
		stemmer = snowballstemmer.stemmer(language)
		stemWord = stemmer.stemWords(word)
		return model.wv.most_similar(positive=stemWord, topn=topn)
	except:
		return ""
# print(getBestSimilarities(modelFR, "black", "english"))
# print(getBestSimilarities(modelFR, "friday", "english"))


def getWords(sims):
	words = []
	for t in sims:
		words.append(t[0])
	return words

from sklearn.metrics.pairwise import cosine_similarity

def search(query):
	user_input = query.split()
	user_input_lower = [ w.lower() for w in user_input ]
	simsTuple = getBestSimilarities(modelFR, user_input_lower,"french",topn = 50)
	print(getBestSimilarities(modelFR, user_input_lower,"french",topn = 30))
	sims = getWords(simsTuple)
	concat = user_input_lower + sims
	X = vectorizerFR.transform([" ".join(concat)])
	cat = bestModelFR.predict(X)
	print(bestModelFR.predict_proba(X))
	print(categories[cat[0]])

	res = est.getDocuments("rss",{
		"match": {
		"Catégorie_du_flux":categories[cat[0]]
	}})
	hits = res["hits"]["hits"]
	docs = []
	for hit in hits:
		if hit["_source"]["data"] != None:
			X2 = vectorizerFR.transform([hit["_source"]["data"]])
			docs.append((hit["_source"],cosine_similarity(X,X2)[0][0]))		

	# with open('readmee.txt', 'w',encoding="utf-8") as f:
	# 	f.write(str(sorted(docs,key=lambda x: x[1],reverse=True)))
	return categories[cat[0]],sorted(docs,key=lambda x: x[1],reverse=True)


