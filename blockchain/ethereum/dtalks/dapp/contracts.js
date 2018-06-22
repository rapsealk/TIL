const Web3 = require('web3');
const web3 = new Web3(new Web3.providers.HttpProvider('http://127.0.0.1:8545'));

const tokenAbi = {};
const TokenContract = new web3.eth.Contract(tokenAbi, '');

exports.getAccounts = async () => {
    return await web3.eth.getAccounts();
};