const express = require('express');
const router = express.Router();

const os = require('os');

const contracts = require('../dapp/contracts');

router.get('/', async function(req, res, next) {

	let number = req.query.id || '0';
	let accounts = await contracts.getAccounts();
	console.log('session:', req.session);

	console.log('hexNumber:', Buffer.from(number).toString('hex'));

	let params = {
		account: accounts[parseInt(number)],
		amount: 0,
		IP_ADDRESS: os.networkInterfaces().en0[1].address
	};

	res.render('index', params);
});

module.exports = router;