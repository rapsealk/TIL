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
        address ifSuccessfulSendTo,
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