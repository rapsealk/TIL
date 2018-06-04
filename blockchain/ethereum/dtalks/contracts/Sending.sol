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

/*
        bytes memory b = new bytes(20);
        for (uint8 i = 0; i < 20; i++) b[i] = byte(uint8(uint(owner) / (2 ** (8 * (19 - i)))));
        string storage stringified = string(b);

        bytes32 txHash = keccak256(stringified + _message);
*/
        bytes32 txHash = keccak256(abi.encodePacked(block.timestamp, msg.sender, _message));
        // bytes32 txHash = keccak256(block.timestamp, msg.sender, _message);
        ownerByHash[txHash] = owner;
        messages[txHash] = _message;

        // return _message;
        return txHash;
    }
/*
    function addressToString(address _address) private returns (string) {
        bytes memory b = new bytes(20);
        for (uint i = 0; i < 20; i++) b[i] = byte(uint8(uint(_address) / (2 ** (8 * (19 - i)))));
        return string(b);
    }
*/
}