from gensim.models import Word2Vec
import elasticSearchTest as est
client = est.es_client
from gensim.models import Phrases
import snowballstemmer

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
		

bigramFR = Phrases(corpusFR)
bigramEN = Phrases(corpusEN)

modelFR = Word2Vec(corpusFR, vector_size=128, window=10, min_count=1, workers=4)
modelEN = Word2Vec(corpusEN, vector_size=128, window=10, min_count=1, workers=4)

modelFR.save("model/word2vecFR.model")
modelEN.save("model/word2vecEN.model")


modelFR = Word2Vec.load("model/word2vecFR.model")
modelEN = Word2Vec.load("model/word2vecEN.model")

def getBestSimilarities(model, word, language,topn=10):
	try:
		stemmer = snowballstemmer.stemmer(language)
		stemWord = stemmer.stemWords(word)
		return model.wv.most_similar(positive=stemWord, topn=topn)
	except:
		return ""
# print(getBestSimilarities(modelFR, "black", "english"))
# print(getBestSimilarities(modelFR, "friday", "english"))

while True:
	user_input = input('recherche: ')
	print(getBestSimilarities(modelEN,[ w.lower() for w in user_input.split()],"english",3))


