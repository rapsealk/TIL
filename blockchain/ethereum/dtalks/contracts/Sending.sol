pragma solidity ^0.4.23;

contract Sending {

    address public owner;

    // bytes32 == uint256
    mapping (bytes32 => string) messages;
    mapping (bytes32 => address) ownerByHash;

    constructor() public {
        owner = msg.sender;
    }

    function sendMessage(string _message) public returns (bytes32) {
        require(bytes(_message).length > 0);

        bytes32 txHash = keccak256(abi.encodePacked(block.number, msg.sender, _message));

        ownerByHash[txHash] = owner;
        messages[txHash] = _message;

        return txHash;
    }
}