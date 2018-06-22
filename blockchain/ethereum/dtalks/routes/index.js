const express = require('express');
const router = express.Router();

router.get('/', async function(req, res, next) {

	let params = {
		account: 'Account',
		amount: 0
	};

	res.render('index', params);
});

module.exports = router;