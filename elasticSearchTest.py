from elasticsearch import Elasticsearch 
import utils.readJSON as rp
import datetime
from elasticsearch import helpers
import sys
es_client = Elasticsearch("http://localhost:9200")

def generate_docs():
    reader = rp.getRss()

    for chunk in reader:
        for data in chunk:
            doc = {
                "_index": "rss",
                "_id": data["id"],
                "_source": {
                    "id": data["id"],
                    "title": data["title"],
                    "description": data["description"],
                    "pubDate": data["pubDate"],
                    "link": data["link"],
                    "language": data["language"],
                    "Catégorie_du_flux": data["categorie"],
                    "Catégorie_prédite": None,
                    "date_de_collecte": datetime.datetime.now(),
                    "data": data["data"] }
                }
            yield doc

def clearServer():
    es_client.delete_by_query(index="rss", query={"match_all": {}})

def getDocuments(index, query):
    return es_client.search(index=index, query=query, size=10000)

def updateDocument(index, query, id):
    es_client.update(
        index=index,
        id=id,
        body=query
    )

if __name__=='__main__':
    if len(sys.argv) != 2:
        print ( "See the README.MD to know how to launch")
    else:
        arg = sys.argv[1]
        if arg == "GEN":
            helpers.bulk(es_client, generate_docs())
        if arg == "COUNT":
            result = es_client.count(index="rss")
            print(result.body['count'])
        if arg == "SEARCH":
            res = getDocuments("rss",{"match_all": {} } )
            print ("query hits:", len(res["hits"]["hits"]))
        if arg == "CLEAR":
            clearServer()