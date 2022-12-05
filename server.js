import express from 'express'
import fetch from 'node-fetch'


const app = express()
const port = 3000


app.set('view engine', 'ejs');

app.use(express.json());
app.use(express.urlencoded());
app.use(express.static('./public'));

let cat = "indefini"

app.get('/', (req, res) => {
    res.render('index',{categorie: cat})
})

app.post('/search', async (req, res) => {
    let fname = req.body.fname
    console.log(req.body)
    await fetch("http://localhost:5000/research", {method:'post', body: JSON.stringify(req.body),
    headers: {'Content-Type': 'application/json','Accept': 'application/json'}})
            .then( data => data.json() )
            .then( data => cat = data )
    console.log(cat)
    res.render('index', {categorie:cat})
})
app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})