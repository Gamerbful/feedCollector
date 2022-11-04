from elasticsearch import Elasticsearch 
import requests
import pandas as pd
import sys




es = Elasticsearch([{'host': 'localhost', 'port': 9200, "scheme": "http"}])



def getESVersion(es):
    #es = Elasticsearch()
    print(es.info())


def testServer():
    response = requests.get('http://localhost:9200')
    if response.status_code != 200:
        print('ElasticSearch server not accessible !')
    else: 
        print('so far so good')


def indexData(idx, data):
    response = requests.get('http://localhost:9200')
    if response.status_code != 200:
        print('ElasticSearch server not accessible !')
    else:
        es.index(index='rss', id=idx, document = data)
        print(idx, "indexed")



def searchBdy(bdy, index_name):
    try:
        hits = es.search(index = index_name, query = bdy)
        H = hits['hits']['hits']
        print("# hits:", len(H))
        for h in H:
            print('>>', h['_score'], ' ', h['_source']['id'], h['_source']['numb_etud'])
    except:
        print('error:', sys.exc_info()[0])
        hits=[]
    return hits







if __name__=='__main__':
        
    # reading
    rss = pd.read_pickle('mypicklefile')
    # testing
    getESVersion(es)
    testServer()

    #indexing
    #for i in range(len(rss)):
    #    indexData(rss[i]['id'], rss[i])

    # searching
    #print('match all')
    #bdy_match_all = {'match_all': {}}
    #searchBdy(bdy_match_all)


    # searching with a must
    print('\n \n \nmust (and)')
    bdy_must = {
        "bool": {
            'must': [
                {'match': {'language': 'en'}}
                #{'match': {'title':'Tucci samples sausage in crème brûlée'}}
                #{'match': {'id': '60d0c7096c454ae0cd6d28ae8bb12f1119eba4b69a7bc4c0d24a8cabfd51f998'}}
            ]
        }
    }

    searchBdy(bdy_must)

    #'Tucci samples sausage in crème brûlée'
    """
    for i in range(len(rss)):
        #print(key)
        print(rss[i]['title'])
    """

# id title description pubDate link language data
# id : '60d0c7096c454ae0cd6d28ae8bb12f1119eba4b69a7bc4c0d24a8cabfd51f998'

