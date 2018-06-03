const express = require('express');
const router = express.Router();

const axios = require('axios');

router.get('/', function(req, res) {

	res.json({ key: "value" });
})

router.post('/predict', async function(req, res) {

	console.log('request:', req.body);

	let response = await axios.post('http://localhost:8000/api/predict/', {
		url: req.body.url
	});

	console.log('response:', response)

	res.json({ message: 'Done', predictions: response.data.predictions });	
});

module.exports = router;
