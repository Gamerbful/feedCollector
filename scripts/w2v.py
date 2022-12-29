from gensim.models import Word2Vec
import elasticSearchTest as est
client = est.es_client
from gensim.models import Phrases
import snowballstemmer
import pickle
import numpy as np
from stop_words import get_stop_words 



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



# modelFR = Word2Vec( vector_size=300, window=5,
# 					 min_count=5,sample=6e-5, alpha=0.03, 
#                      min_alpha=0.0007, negative=20, workers=4)
# modelEN = Word2Vec( vector_size=300, window=5,
# 					 min_count=5,sample=6e-5, alpha=0.03, 
#                      min_alpha=0.0007, workers=4)

# modelFR.build_vocab(bigramFR[corpusFR],progress_per=10000)
# modelEN.build_vocab(bigramEN[corpusEN],progress_per=10000)

# modelFR.train(bigramFR[corpusFR], total_examples=modelFR.corpus_count, epochs=30, report_delay=1)
# modelEN.train(bigramEN[corpusEN], total_examples=modelEN.corpus_count ,epochs=30, report_delay=1)

# modelFR.save("model/word2vecFR.model")
# modelEN.save("model/word2vecEN.model")


modelFR = np.load("model/word2vecFR.model", allow_pickle=True)
modelEN = np.load("model/word2vecEN.model", allow_pickle=True)
from joblib import dump, load

vectorizerFR = pickle.load( open("vec/vectorizerFR.pickle", "rb"))
bestModelFR = load("model/bestModelFR.joblib")

vectorizerEN = pickle.load( open("vec/vectorizerEN.pickle", "rb"))
bestModelEN = load("model/bestModelEN.joblib")

categories = ["BUSINESS","SPORT","HEALTH","ART","SCIENCE","POLITIC"]

def getBestSimilarities(model, word, language,topn=10):
	try:
		stemmer = snowballstemmer.stemmer(language)
		stemWord = stemmer.stemWords(word)
		stop_words = get_stop_words(language)
		filtered_sentence = [w for w in stemWord if not w.lower() in stop_words]
		filtered_sentence_in_vocab = [w for w in filtered_sentence if w in model.wv.key_to_index.keys()]
		print(filtered_sentence_in_vocab)
		return model.wv.most_similar(positive=filtered_sentence_in_vocab, topn=topn)
	except:
		return ""
# print(getBestSimilarities(modelEN, ["black"], "english"))
# print(getBestSimilarities(modelEN, ["friday"], "english"))


def getWords(sims):
	words = []
	for t in sims:
		words.append(t[0])
	return words

from sklearn.metrics.pairwise import cosine_similarity

def search(query,language):
	if language == "fr":
		lg = "french"
		model = modelFR
		bestModel = bestModelFR
		vectorizer = vectorizerFR
	else:
		lg = "english"
		model = modelEN
		bestModel = bestModelEN
		vectorizer = vectorizerEN

	user_input = query.split()
	user_input_lower = [ w.lower() for w in user_input ]
	simsTuple = getBestSimilarities(model, user_input_lower,lg,topn = 100)
	print(getBestSimilarities(model, user_input_lower,lg,topn = 30))
	sims = getWords(simsTuple)
	concat = user_input_lower + sims
	X = vectorizer.transform([" ".join(concat)])
	cat = bestModel.predict(X)
	print(bestModel.predict_proba(X))
	print(categories[cat[0]])

	res = est.getDocuments("rss",{
		"bool": {
			"should": [
				{
					"match": {
						"Catégorie_du_flux":categories[cat[0]]
						}

				}
				# ,
				# {
				# 	"match": {
				# 		"language":language
				# 		}

				# },

			]
		}
		})
	hits = res["hits"]["hits"]
	docs = []
	for hit in hits:
		if hit["_source"]["data"] != None:
			X2 = vectorizer.transform([hit["_source"]["data"]])
			docs.append((hit["_source"],cosine_similarity(X,X2)[0][0]))		

	# with open('readmee.txt', 'w',encoding="utf-8") as f:
	# 	f.write(str(sorted(docs,key=lambda x: x[1],reverse=True)))
	return categories[cat[0]],sorted(docs,key=lambda x: x[1],reverse=True)


