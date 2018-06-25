pragma solidity ^0.4.18;

interface token {
    function transfer(address _to, uint256 _amount) external;
}

contract Keep {

    address public owner;
    bytes public message;

    token public chatToken;
    uint public price;

    mapping (address => bytes32[]) public history;

    event MessageStored(address indexed sender, bytes message, uint gas);

    constructor
    // function Keep
        (address addressOfToken) public {
        owner = msg.sender;
        // message = _message;
        chatToken = token(addressOfToken);
        price = 1 ether;
    }
    
    function keepMessage(string _message) public returns (uint) {
        message = bytes(_message);
        history[msg.sender].push(block.blockhash(block.number));
        emit MessageStored(msg.sender, message, tx.gasprice);
        return block.number;
    }

    function getHistory() public view returns (bytes32[]) {
        return history[msg.sender];
    }
}