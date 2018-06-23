const express = require('express');
const router = express.Router();

const os = require('os');

const contracts = require('../dapp/contracts');

router.get('/', async function(req, res, next) {

	let accounts = await contracts.getAccounts();

	let params = {
		account: accounts[0],
		amount: 0,
		IP_ADDRESS: os.networkInterfaces().en0[1].address
	};

	res.render('index', params);
});

module.exports = router;