import json
import os

def getRss():
    for filename in os.listdir("rss"):
        with open("rss/"+filename) as json_file:
            yield json.load(json_file)


