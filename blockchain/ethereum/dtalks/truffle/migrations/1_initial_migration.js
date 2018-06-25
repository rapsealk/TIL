var Migrations = artifacts.require("./Migrations.sol");
var Token = artifacts.require("./Token.sol");
var Keep = artifacts.require("./Keep.sol");

module.exports = function(deployer) {
  deployer.deploy(Migrations);
  deployer.deploy(Token);
  deployer.deploy(Keep);
};
