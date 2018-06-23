pragma solidity ^0.4.23;

interface token {
    function transfer(address receiver, uint amount) external;
}

contract Provision {

    address public beneficiary;
    uint public price;
    token public tokenReward;
    mapping (address => uint256) public balanceOf;

    constructor(
        address ifSuccessfulSendTo,         // 0xab7c4dde3fCbbA0dd47b7B98C8cb15B59103A600
        uint etherCostOfEachToken,
        address addressOfTokenUsedAsReward
    ) public {
        beneficiary = ifSuccessfulSendTo;
        price = etherCostOfEachToken;
        tokenReward = token(addressOfTokenUsedAsReward);
    }

    function () payable external {
        uint amount = msg.value;
        balanceOf[beneficiary] += amount;
        tokenReward.transfer(beneficiary, amount / price);
    }
}