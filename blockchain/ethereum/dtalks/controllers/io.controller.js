const Contract = require('../dapp/contracts');
const Address = require('../dapp/address');

exports.initSocket = function(io) {

    io.on('connect', function(socket) {

        console.log('socket::connect');

        socket.on('message', async function(data) {
            console.log('io::message:', data.message);

            // contract with message
            // await Contract.sendTransaction(Address.accounts.etherbase, Address.accounts._02, )
            try {
                const tx = await Contract.keepMessage(data.message);
                console.log('transaction result:', tx);
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