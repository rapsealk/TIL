pragma solidity ^0.4.23;

contract Token {

    string public constant name = "Decentralized Talk Storage Token";
    string public constant symbol = "DTSTs";

    uint8 public decimals = 18;

    mapping (address => uint256) public balanceOf;

    event Transfer(address indexed _from, address indexed _to, uint256 _amount);

    constructor(uint256 initialAmount) public {
        balanceOf[msg.sender] = initialAmount;
    }

    function transfer(address _to, uint256 _amount) public returns (bool success) {
        // TODO("modifier")
        require(balanceOf[msg.sender] >= _amount);
        if (balanceOf[_to] + _amount < balanceOf[_to]) revert();

        balanceOf[msg.sender] -= _amount;
        balanceOf[_to] += _amount;
        emit Transfer(msg.sender, _to, _amount);

        return true;
    }
}