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

def generateID(link ):
    return hashlib.sha256(link.encode())

def getID(hash ):
    return hash.hexdigest()

def retrieveData(entrie ):
    print('yeeah')

def isInEntry(entrie ):
    if (entrie.popeye == None ):
        print(False)
    # entrie.title
    # entrie.description
    # entrie.pubDate

for post in flux.entries:
    isInEntry(post)