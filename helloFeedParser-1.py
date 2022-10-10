#!/usr/bin/python3


#
import sys
from this import d
from bs4 import BeautifulSoup
import pickle
import urllib.request, feedparser
from langdetect import detect
#import textract
import hashlib
import numpy as np

#proxy = urllib.request.ProxyHandler({'http' : 'http://squidva.univ-ubs.fr:3128/'} )

import time
from subprocess import check_output

from matplotlib.pyplot import title



# --------------------
# CNN Collector (feedparser)
# --------------------

url_cert = 'https://www.cert.ssi.gouv.fr/alerte/feed/'
url_cnn = 'http://rss.cnn.com/rss/edition.rss'
#d = feedparser.parse(url_cert, handlers = [proxy])

# print all posts

def getDictOfFoundedProps(post):
    return {
        "title" : isInEntry(post,"title"),
        "description" : isInEntry(post,"description"),
        "pubDate" : isInEntry(post,"pubDate"),
        "link" : isInEntry(post,"link")
    }

def generateID(post, dictOfTruth):
    key = getEntry(dictOfTruth,post,'title') + getEntry(dictOfTruth,post,'description') + getEntry(dictOfTruth,post,'pubDate') + getEntry(dictOfTruth,post,'link')
    if key != "":
        return hashlib.sha256(key.encode())

def getID(hash ):
    return hash.hexdigest()

def getLanguage(post,dictOfTruth):
    txt = getEntry(dictOfTruth,post,'title') + getEntry(dictOfTruth,post,'description') 
    if txt != "":
        return detect(txt)

def dataToAscii(post, dictOfTruth ):
    link = getEntry(dictOfTruth,post,'link')
    if link != "":
        try:
            with urllib.request.urlopen(link) as f:
                return  f.read().decode('utf-8').encode("ascii","ignore")
        except:
            return ''

def isInEntry(post, name ):
    try:
        if post[name] != 0:
            return True
    except:
        return False

def getEntry(dictOfTruth, post, name):
    if( dictOfTruth[name] ):
        return post[name]
    else:
        return ""

def generateDict(dictOfTruth, post, id):
    return {  "id": id,
            "title": getEntry(dictOfTruth,post,'title'),
            "description": getEntry(dictOfTruth,post,'description'),
            "pubDate": getEntry(dictOfTruth,post,'pubDate'),
            "link": getEntry(dictOfTruth,post,'link'),
            "language": getLanguage(post, dictOfTruth),
            "data": dataToAscii(post, dictOfTruth) }


def parsingData(rss_link, repertory):
    """
    ARGS:
        rss_link -> link of rss stream
        repertory -> output repertory
    OUTPUT:

    """
    flux = feedparser.parse(rss_link)
    parsedData = []
    n = len(flux.entries)
    if  n == 0 :
        print("error: Not a rss url / No entries in the rss")
    else:
        url = flux.channel.link
        index = 0
        for post in flux.entries:
            index += 1
            dictOfTruth = getDictOfFoundedProps(post)
            id = getID(generateID(post, dictOfTruth))
            dictTest = generateDict(dictOfTruth, post, id)
            parsedData.append(dictTest)
            print(f"{index/n}% effectué", end='\r')
            if index == 1000:
                with open('mypicklefile', 'wb') as f1:
                    pickle.dump(parsedData, f1)
                break


if __name__ == "__main__":
    if len(sys.argv) != 3: 
        print("usage: python .\helloFeddParser-1.py [rssLink] [path to store pickle file]")
    else:
        parsingData(sys.argv[1], sys.argv[2])

    