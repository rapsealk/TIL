pragma solidity ^0.4.23;

import "truffle/Assert.sol";
import "truffle/DeployedAddresses.sol";
import "../contracts/Sending.sol";

contract TestSending {

    Sending sending = Sending(DeployedAddresses.Sending());

    function testUserCanSendMessage() public {
        string memory message = "Test Message";
        bytes32 returnedHash = sending.sendMessage(message);

        bytes32 expected = keccak256(abi.encodePacked(block.number, this, message));

        Assert.equal(returnedHash, expected, "Transaction hash from sending message is not correct.");
    }
}