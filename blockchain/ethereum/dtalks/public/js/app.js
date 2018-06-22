App = {
    web3Provider: null,
    contracts: {},

    init: function() {

        return App.initWeb3();
    },

    initWeb3: function() {
        // Is there an injected web3 instance?
        if (typeof web3 !== 'undefined') {
            App.web3Provider = web3.currentProvider;
        } else {
            // If no injected web3 instance is detected, fall back to Ganache.
            App.web3Provider = new Web3.providers.HttpProvider("http://localhost:7545");
        }
        web3 = new Web3(App.web3Provider);

        return App.initContract();
    },

    initContract: function() {
        $.getJSON('Source.json', function(data) {
            // Get the necessary contract artifact file and instantiate it with truffle-contract.
            var SourceArtifact = data;
            App.contracts.Source = TruffleContract(SourceArtifact);

            // Set the provider for our contract
            App.contracts.Source.setProvider(App.web3Provider);

            // Use our contract to retrieve and mark the ??
            // return App.markAdopted();
        });
        return App.bindEvents();
    },

    bindEvents: function() {
        $(document).on('click', '#btn-offer', App.handleOffer);
    },

    // markAdopted(): function(adopters, account)

    handleOffer: function(event) {
        event.preventDefault();

        // TODO("Generate hash")
        var hashId = 'HASH_STRING';

        var offerInstance;

        web3.eth.getAccounts(function(error, accounts) {
            if (error) {
                console.error(error);
            }

            var account = accounts[0];

            App.contracts.Source.deployed()
                .then(function(instance) {
                    offerInstance = instance;

                    // Execute offer as a transaction by sending account.
                    return offerInstance.offer(hashId, { from: account, gas: 21000 });
                })
                .then(function(result) {
                    // TODO("After")
                    console.log('offer result:', result);
                    return result;
                })
                .catch(function(error) {
                    console.error(error);
                });
        });
    }
};

$(function() {
    $(window).load(function() {
        App.init();
    });
});