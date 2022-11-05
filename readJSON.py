import json

def getRss():
    with open('mydata.json') as json_file:
        return json.load(json_file)
