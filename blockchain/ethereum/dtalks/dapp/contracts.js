const Web3 = require('web3');
const web3 = new Web3(new Web3.providers.HttpProvider('http://127.0.0.1:8545'));

const Address = require('./address');

console.log('web3.personal:', web3.personal);

/*
try {
    console.log('web3.personal:', web3.personal);
    // web3.eth.unlockAccount.unlockAccount(web3.eth.coinbase, 'PASSPHRASE');
}
catch (error) {
    console.error(error);
    process.exit(1);
}
*/

const tokenABI = require('../truffle/build/contracts/Token.json').abi;
const TokenContract = new web3.eth.Contract(tokenABI, Address.token);

const keepABI = require('../truffle/build/contracts/Keep.json').abi;
// const KeepContract = new web3.eth.Contract(keepABI, Address.keep);

exports.getAccounts = async () => {
    return await web3.eth.getAccounts();
};

exports.sendTransaction = async (from, to, value, gas) => {
    try {
        return await web3.eth.sendTransaction({
            from: from,
            to: to,
            value: web3.toWei(value, 'ether'),
            gas: gas || 100000
        }); // returns 32byte hex hash
    }
    catch (error) {
        console.log('error:', error);
        return error;
    }
};

exports.getTransaction = async (id) => {
    return await web3.eth.getTransaction(id);
};

exports.keepMessage = async (message) => {
    const KeepContract = new web3.eth.Contract(keepABI, Address.keep);
    // console.log('KeepContract:', KeepContract);
    // console.log('KeepContract.methods:', KeepContract.methods);
    // console.log('KeepContract.methods.keepMessage:', KeepContract.methods.keepMessage);
    console.log('KeepContract.keepMessage:', KeepContract.keepMessage);
    // return await KeepContract.methods.keepMessage(message).call();
    KeepContract.methods.keepMessage(message).call((error, tx) => {
        console.log('tx:', tx);
    });
    return true;
    /*
    return await KeepContract.methods.keepMessage(message).sendTransaction({
        from: web3.eth.getAccounts()[0],
        to: web3.eth.getAccounts()[1],
        gas: 100000,
        data: message
    });
    */
};

/*
exports.checkTransaction = (id, callback) => {
    if (web3.eth.getTransaction(id) !== null) {
        console.log('transaction has mined:', id);
    }
    setTimeout(() => {
        if (web3.eth.getTransaction(id) !== null) {
            clearTimeout(this);
        }
    }, 3000);
    checkTransaction(id, callback);
}
*/

exports.transferTo = async (to, amount) => {
    return await TokenContract.transfer(to, amount);
};

exports.transferEvent = callback => {
    TokenContract.Transfer().watch((error, res) => {
        if (error) return callback(error, null);
        return callback(null, res);
    });
};