import snowballstemmer
import elasticSearchTest as est


categories = ["BUSINESS","SPORT","HEALTH","ART","SCIENCE"]

for cat in categories:
    res = est.getDocuments("rss",{"match": {
        "Catégorie_du_flux" : cat
    } })
    print("{} : {}".format(cat,len(res["hits"]["hits"])))