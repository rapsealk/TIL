pragma solidity ^0.4.24; // ^0.4.18

interface token {
    function transfer(address _to, uint256 _amount) external;
}

contract Keep {

    address public owner;
    bytes public message;

    token public chatToken;
    uint public price;

    mapping (address => uint) public balanceOf;

    event MessageStored(address indexed sender, bytes message, uint gas);

    /*
    modifier hasEnoughCoin(address _sender, uint256 cost) {
        require(balanceOf[_sender] >= cost);
        _;
    }
    */

    constructor
    // function Keep
        (address addressOfToken) public {
        owner = msg.sender;
        // message = _message;
        chatToken = token(addressOfToken);
        price = 1 ether;
    }

    /*
    function () payable public {
        require(balanceOf[msg.sender] >= 0);
        // address receiver = msg.to;
        // uint256 amount = msg.value;
        // bytes storage data = msg.data;
        // uint256 gas = tx.gasprice;
        message = bytes(msg.data);

        emit MessageStored(msg.sender, msg.data, tx.gasprice);
    }
    */
    
    function keepMessage(string _message) public returns (uint) {
        message = bytes(_message);
        emit MessageStored(msg.sender, message, tx.gasprice);
        return block.number;
    }
}