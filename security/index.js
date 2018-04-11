const cryptoJS = require('crypto-js');
const crypto = require('crypto');
const eccrypto = require('eccrypto');   // https://www.npmjs.com/package/eccrypto

/**
 * Cryptograph
 * @author rapsealk
 * Symmetric-Key Algorithm: AES (Advanced Encryption Standard)
 * Asymmetric-Key Algorithm: ECC (Elliptic Curve Cryptography)
 * Hash Algorithm: SHA-256 (Secure Hash Algorithm)
 */

const studentNumber = '2015125005';
const studentName = 'KANGJUNGSUK';

const plainText = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

// Generate Hash Key
const hashKey = studentNumber.split('').map(number => (number | 0x30).toString(2).padStart(8, '0'));
studentName.split('').forEach(char => hashKey.push(char.toString(2).padStart(8, '0')));
console.log('hashKey:', hashKey.join(''));

// Hash plain text
// const hashedMessage = cryptoJS.SHA256(plainText).toString();
const hashedMessage = crypto.createHash('sha256').update(plainText).digest();
console.log('hashedMessage:', hashedMessage);

// Encrypt plain text with AES Algorithm
const aesKey = 'AES_KEY';
const cipherText = cryptoJS.AES.encrypt(plainText, aesKey).toString();
console.log('cipherText:', cipherText);

const decipherText = cryptoJS.AES.decrypt(cipherText, aesKey).toString(cryptoJS.enc.Utf8);
console.log('decipherText:', decipherText);

// Generate ECC key pair
const privateKeyA = crypto.randomBytes(32);
const publicKeyA = eccrypto.getPublic(privateKeyA);
const privateKeyB = crypto.randomBytes(32);
const publicKeyB = eccrypto.getPublic(privateKeyB);

(async function() {
    // A sends to B
    try {
        let signature = await eccrypto.sign(privateKeyA, hashedMessage);
        console.log('signature in DER format:', signature);
        eccrypto.encrypt(publicKeyB, aesKey)
            .then(encrypted => console.log('encrypted:', encrypted))
            .catch(error => console.log('error:', error));
        //let encryptedAESKey = await eccrypto.encrypt(publicKeyB, aesKey);
        //console.log('encryptedAESKey:', encryptedAESKey);
        // B received encrypted.
        let verification = await eccrypto.verify(publicKeyA, hashedMessage, signature);
        console.log('verification signature:', verification);
    }
    catch (error) {
        console.log('caught error:', error);
    }
})();
/*
eccrypto.sign(privateKeyA, hashedMessage)
    .then(async (signature) => {
        console.log('signature in DER format:', signature);
        eccrypto.verify(publicKeyA, hashedMessage, signature)
            .then(() => console.log('Signature is OK.'))
            .catch(() => console.log('Signature is BAD.'));
    })
    .catch(() => console.log('Signature is incorrect.'));
*/

/*
const ec = new EC('secp256k1');
const keyPairA = ec.genKeyPair();
const keyPairB = ec.genKeyPair();
console.log('keyPairA:', keyPairA, ', keyPairB:', keyPairB, ', equals: ', keyPairA == keyPairB);
console.log('keyPairA:', keyPairA.getPublic(), ', keyPairB:', keyPairB.getPublic(), ', equals: ', keyPairA == keyPairB);
console.log('keyPairA:', keyPairA.getPrivate(), ', keyPairB:', keyPairB.getPrivate(), ', equals: ', keyPairA == keyPairB);

const encryptedHashedMessage = keyPairA.sign(hashedMessage.split(''));
console.log('encryptedHashedMessage:', encryptedHashedMessage);
console.log('encryptedHashedMessage:', encryptedHashedMessage.toDER().map(d => d.toString(16)).join(''));

const encryptedHashKey = keyPairB.getPublic().sign(hashKey);
console.log('encryptedHashKey:', encryptedHashKey);
*/

/*
const cipherText = crypto.AES.encrypt(plainText, secretKey.join('')).toString();
console.log('cipherText:', cipherText);

let bytes = crypto.AES.decrypt(cipherText, secretKey.join(''));
let decipherText = bytes.toString(crypto.enc.Utf8);
console.log('decipherText:', decipherText);

let plainText2 = String.fromCharCode(plainText.charCodeAt(0) ^ 0x80) + plainText.slice(1);
console.log('plainText2:', plainText2);
let cipherText2 = crypto.AES.encrypt(plainText2, secretKey.join('')).toString();
console.log('cipherText2:', cipherText2);

console.log('cipherText1:', cipherText.split('').map(c => c.charCodeAt(0).toString(2).padStart(8, '0')).join(' '));
console.log('cipherText2:', cipherText2.split('').map(c => c.charCodeAt(0).toString(2).padStart(8, '0')).join(' '));
console.log('cipherText2:', cipherText.split('').map((c, i) => c == cipherText2.charAt(i)).join(' '));
*/