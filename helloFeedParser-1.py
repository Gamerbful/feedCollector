#!/usr/bin/python3


#

import urllib.request, feedparser
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

def generateID(post):
    key = isInEntry(post,'title') + isInEntry(post,'description') + isInEntry(post,'pubDate')
    if key != "":
        return hashlib.sha256(key.encode())

def getID(hash ):
    return hash.hexdigest()

def dataToAscii(post ):
    return

def isInEntry(post, name ):
    try:
        return post[name]
    except:
        return ""
    # entrie.title
    # entrie.description
    # entrie.pubDate

for post in flux.entries:
    print(getID(generateID(post)))
    