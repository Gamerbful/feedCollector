# feedCollector
RSS-Intelligence project.
The aim of the project is to reproduce the working of a basic searchEngine using some tools of machineLearning and text processing.
Here you will have some steps to follow to first use our architecture and after to deploy it in a RESTful API.
Nothing too complicated here, if you have any errors go to the error sections there will be some error cases otherwise contact us. ⭐

# Full Guideline

- Install <strong>Elasticsearch</strong> https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html
- Install / clone this github repo https://github.com/Gamerbful/feedCollector.git

- Launch your Elasticsearch server ( on windows launch elasticsearch.bat in the bin folder of your server ) 
- Open a <strong>terminal</strong> in you favorite IDE or just in raw if you don't like using any IDE :)
- Get into the <strong>root folder</strong> of the project -> <strong>feedCollector</strong>
- ``` pip install requirement.txt ``` to install dependencies

- ```python .\scripts\helloFeedParser-1.py ``` to parse rss flux they will be temp stored in rss folder
- ```python .\scripts\elasticSearchTest.py GEN ``` save rss flux in your local elasticsearch server ( may have some security error )
- ```python .\scripts\classifier.py ``` if you have our models it will just try to classify your new flux else if you delete bestModel and vectorizer it will train and create new model

# Launch Serverside

- Open two terminal and assure your elasticsearch server is running

- ``` python .\scripts\flaskAPI.py ``` start our api wich will augment user query and return predicted categorie and ordered docs
- ``` npm start ``` start an express server with ejs view on port 3000 of localhost

# Pictures worth a thousand words

<h1> Here you will have a desctiption of major scripts of our architecture, you may consider reading it if you want more details but it's okay not to read everything </h1>

# elasticSearchTest

to launch
```shell
python .\elasticSearchTest.py [param]
```
those parameters are
- GEN to index our JSON rss file ( need to use helloFeedParser script first ) 
- COUNT to count how much indexed documents we have
- SEARCH to test a query ( by default get all doc under an index)
- CLEAR to clear our indexed documents

# helloFeedParser-1
# classifier
# w2v
# server
# flaskAPI
# Error Section
