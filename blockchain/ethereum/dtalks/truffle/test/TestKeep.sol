pragma solidity ^0.4.23;

import "truffle/Assert.sol";
import "truffle/DeployedAddresses.sol";
import "../contracts/Keep.sol";

contract TestKeep {

    Keep keep = Keep(DeployedAddresses.Keep());
/*
    function testGetMessage() public {
        address sender;
        bytes message;
        bool flag;

        (sender, message, flag) = chat.getMsg();

        Assert.equal(sender, this, "msg.sender not matching.");
    }
*/
}