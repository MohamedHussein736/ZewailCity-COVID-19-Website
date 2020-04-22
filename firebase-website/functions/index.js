//Importing dependencies
const express = require('express');
const functions = require('firebase-functions');

//Starting Express app
const app = express();


app.get('**',(request,response)=> {
    response.set('Cache-control','public,max-age=30,s-maxage=60');
});

