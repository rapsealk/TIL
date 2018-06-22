var Token = artifacts.require("./Token.sol");
var Provision = artifacts.require("./Provision.sol");

module.exports = function(deployer) {
    deployer.deploy(Token);
    deployer.deploy(Provision);
};