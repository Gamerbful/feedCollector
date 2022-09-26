#!/usr/bin/python3


#
from bs4 import BeautifulSoup
import pickle
import urllib.request, feedparser
from langdetect import detect
import textract
import hashlib

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
flux = feedparser.parse(url_cnn)
url = flux.channel.link
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
                data = f.read().decode('utf-8')
                newFile = open('myFile.txt', "a")
                newFile.write(data)
                return textract.process("./myFile.txt")
        except:
            return

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

for post in flux.entries:
    dictOfTruth = getDictOfFoundedProps(post)
    print(dataToAscii(post, dictOfTruth))
    