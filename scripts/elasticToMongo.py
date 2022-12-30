from pymongo import MongoClient
import elasticSearchTest as est
import os
from dotenv import load_dotenv

load_dotenv('./scripts/mongodb.env')

def get_database():
 
   # Provide the mongodb atlas url to connect python to mongodb using pymongo
   CONNECTION_STRING = os.getenv('MONGODB_URL')
 
   # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
   client = MongoClient(CONNECTION_STRING)
 
   # Create the database for our example (we will use the same database throughout the tutorial
   return client['portfolio']

def elasticToMongo():
   res = est.getDocuments("rss",{"match_all": {} })
   hits = res["hits"]["hits"]
   docs = []
   for hit in hits:
    if hit["_source"]["data"] != None:
        hit["_source"]["_id"] = hit["_source"].pop("id")
        docs.append(hit["_source"])
   # Get the database
   dbname = get_database()
   collection = dbname["docs"]
   collection.insert_many(docs)

def getAllDocs(query):
   dbname = get_database()
   collection = dbname["docs"]
   return collection.find(query)


def deleteAllDocs(query):
   dbname = get_database()
   collection = dbname["docs"]
   return collection.delete_many(query)
  