exports.initSocket = function(io) {

    io.on('connect', function(socket) {

        console.log('socket connected:', socket);

        socket.on('message', function(data) {
            console.log('io::message::data:', data);
            socket.broadcast.emit('message', data);
        });
    });

    io.on('connection', function(socket) {
        console.log('connection:', socket);
    });
};