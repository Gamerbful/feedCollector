from elasticsearch import Elasticsearch 
import requests
import pandas as pd
import sys


from elasticsearch import helpers

es_client = Elasticsearch("http://localhost:9200")



def getes_clientVersion(es_client):
    #es_client = Elasticsearch()
    print(es_client.info())


def tes_clienttServer():
    res_clientponse = requests.get('http://localhost:9200')
    if res_clientponse.status_code != 200:
        print('ElasticSearch server not acces_clientsible !')
    else: 
        print('so far so good')


def indexData(idx, data):
    res_clientponse = requests.get('http://localhost:9200')
    if res_clientponse.status_code != 200:
        print('ElasticSearch server not acces_clientsible !')
    else:
        rs = es_client.index(index='rss', id=idx, document = data)



def searchBdy(bdy, index_name):
    try:
        hits = es_client.search(index = index_name, query = bdy)
        H = hits['hits']['hits']
        print("# hits:", len(H))
        for h in H:
            print('>>', h['_score'], ' ', h['_source']['id'], h['_source']['numb_etud'])
    except:
        print('error:', sys.exc_info()[0])
        hits=[]
    return hits

    
import readJSON as rp
import datetime
def generate_docs():
    reader = rp.getRss()

    for data in reader:
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
                "Catégorie_du_flux": None,
                "Catégorie_prédite": None,
                "date_de_collecte": datetime.datetime.now() }
            }
        yield doc

import readJSON as rj
if __name__=='__main__':
    
    # reading
    rss = rj.getRss()
    # tes_clientting
    getes_clientVersion(es_client)

    # indexing
    # for i in range(len(rss)):
    #     indexData(rss[i]['id'], rss[i])
    helpers.bulk(es_client, generate_docs())
    # es_client.delete_by_query(index="rss", query={"match_all": {}})
    result = es_client.count(index="rss")
    print(result.body['count'])
    # searching
    # print('match all')
    # bdy_match_all = {'match_all': {}}

    # searchBdy(bdy_match_all)
    # match_all = {
    #     "size": 100,
    #     "query": {
    #         "match_all": {}
    #     }
    # }
    # res_clientp = es_client.search(
    #         index = "rss",
    #         body = match_all,
    #         scroll = '2s' # length of time to keep search context
    #     )
    # print(res_clientp['hits']['hits'])
    # searching with a must
    # print('\n \n \nmust (and)')
    # bdy_must = {
    #     "bool": {
    #         'must': [
    #             {'match': {'language': 'en'}}
    #             #{'match': {'title':'Tucci samples_client sausage in crème brûlée'}}
    #             #{'match': {'id': '60d0c7096c454ae0cd6d28ae8bb12f1119eba4b69a7bc4c0d24a8cabfd51f998'}}
    #         ]
    #     }
    # }

    # searchBdy(bdy_must)

    #'Tucci samples_client sausage in crème brûlée'
    """
    for i in range(len(rss)):
        #print(key)
        print(rss[i]['title'])
    """

# id title des_clientcription pubDate link language data
# id : '60d0c7096c454ae0cd6d28ae8bb12f1119eba4b69a7bc4c0d24a8cabfd51f998'

