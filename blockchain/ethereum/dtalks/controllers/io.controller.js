const Contract = require('../dapp/contracts');
const Address = require('../dapp/address');

exports.initSocket = function(io) {

    io.on('connect', function(socket) {

        console.log('socket::connect');

        socket.on('message', async function(data) {
            console.log('io::message:', data);

            const account = data.from;

            // contract with message
            // await Contract.sendTransaction(Address.accounts.etherbase, Address.accounts._02, )
            try {
                // const tx = await Contract.keepMessage(data.message);
                /*
                let receiver, password;
                if (data.from == Address.accounts.etherbase) {
                    receiver = Address.accounts._02;
                    password = 'PASSWORD';
                }
                else {
                    receiver = Address.accounts.etherbase;
                    password = '';
                }
                */

                /*
                let password;
                if (data.from == Address.accounts.etherbase) password = '';
                else password = '-';
                const signature = await Contract.web3.eth.personal.sign(data.message, data.from, password);
                */

                const tx = await Contract.sendTransaction(Address.accounts.etherbase, Address.accounts._02, '0', null, `${data.message}`);
                // const tx = await Contract.sendTransaction(data.from, receiver, '0', null, data.message, password);
                console.log('transaction result:', tx);
                const message = await Contract.getTransaction(tx.transactionHash);
                // const recovered = await Contract.web3.eth.personal.ecRecover()
                console.log('Retrieved message:', message);

                await Contract.getHistory();
            }
            catch (error) {
                console.error(error);
            }
            finally {
                socket.broadcast.emit('message', data);
            }

        });
    });

    io.on('connection', function(socket) {
        console.log('socket::connection');
    });

    io.on('disconnect', function() {
        console.log('socket::disconnect');
    });
};