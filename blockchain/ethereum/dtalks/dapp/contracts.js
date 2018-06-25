const Web3 = require('web3');
const web3 = new Web3(new Web3.providers.HttpProvider('http://127.0.0.1:8545'));

const Address = require('./address');

// console.log('web3.eth.personal:', web3.eth.personal);
(async function() {
    console.log('accounts:', await web3.eth.getAccounts());
})();

const tokenABI = require('../truffle/build/contracts/Token.json').abi;
const TokenContract = new web3.eth.Contract(tokenABI, Address.token);

const keepABI = require('../truffle/build/contracts/Keep.json').abi;
// const KeepContract = new web3.eth.Contract(keepABI, Address.keep);
const KeepContract = new web3.eth.Contract(keepABI);

exports.getAccounts = async () => {
    return await web3.eth.getAccounts();
};

exports.sendTransaction = async (from, to, value, gas, message, password) => {
    try {
        // await web3.eth.personal.unlockAccount(from, password, '1000');
        return await web3.eth.sendTransaction({
            from: from,
            to: to,
            value: web3.utils.toWei(value, 'ether'),
            gas: gas || 100000,
            data: web3.utils.stringToHex(message)
        }); // returns 32byte hex hash
    }
    catch (error) {
        console.log('error:', error);
        return error;
    }
};

exports.getTransaction = async (id) => {
    const transaction = await web3.eth.getTransaction(id);
    const input = transaction.input;
    console.log('Get Transaction: input >>', input);
    let message = web3.utils.hexToString(input);
    return message;
};

exports.keepMessage = async (message) => {
    /*
    KeepContract.methods.keepMessage(message).call((error, tx) => {
        console.log('tx:', tx);
    });
    */
    let result = await new web3.eth.Contract(keepABI, Address.keep, {
        from: Address.accounts.etherbase,
        gasPrice: '100000'
    }).methods.keepMessage(message).call();
    
    console.log('result:', result);
    return result;
    /*
    return await KeepContract.methods.keepMessage(message).sendTransaction({
        from: web3.eth.getAccounts()[0],
        to: web3.eth.getAccounts()[1],
        gas: 100000,
        data: message
    });
    */
};

exports.getHistory = async () => {
    const KeepContract = new web3.eth.Contract(keepABI, Address.keep, {
        from: await web3.eth.getAccounts()[0],
        gasPrice: '10000'
    });
    await KeepContract.methods.getHistory().call((error, tx) => {
        if (error) console.log(error);
        console.log('History tx:', tx);
    });
};

exports.transferTo = async (to, amount) => {
    return await TokenContract.transfer(to, amount);
};

exports.transferEvent = callback => {
    TokenContract.Transfer().watch((error, res) => {
        if (error) return callback(error, null);
        return callback(null, res);
    });
};

exports.web3 = web3;