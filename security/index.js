const cryptoJS = require('crypto-js');
const crypto = require('crypto');
const ecurve = require('ecurve');
const BigInteger = require('bigi');
const EC = require('elliptic').ec;

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
const hashedMessage = cryptoJS.SHA256(plainText).toString();
// const hashedMessage = crypto.createHash('sha256').update(plainText).digest();
console.log('hashedMessage:', hashedMessage);

// Encrypt plain text with AES Algorithm
const aesKey = 'AES_KEY';
const cipherText = cryptoJS.AES.encrypt(plainText, aesKey).toString();
console.log('cipherText:', cipherText);

const decipherText = cryptoJS.AES.decrypt(cipherText, aesKey).toString(cryptoJS.enc.Utf8);
console.log('decipherText:', decipherText);

/* Generate ECC key pair
const ecparams = ecurve.getCurveByName('secp256k1');
const privateKeyA = crypto.randomBytes(32);
const privateKeyB = crypto.randomBytes(32);
console.log('privateKeyA:', privateKeyA.toString('hex'));
console.log('privateKeyB:', privateKeyB.toString('hex'));
const curvePointA = ecparams.G.multiply(BigInteger.fromBuffer(privateKeyA));
const pointX_A = curvePointA.affineX.toBuffer(32);
const pointY_A = curvePointA.affineY.toBuffer(32);
// console.log('A(x, y):', pointX_A, pointY_A);
const publicKeyA = Buffer.concat([new Buffer([0x04]), pointX_A, pointY_A]);
console.log('publicKeyA:', publicKeyA.toString('hex'));
const curvePointB = ecparams.G.multiply(BigInteger.fromBuffer(privateKeyB));
const pointX_B = curvePointB.affineX.toBuffer(32);
const pointY_B = curvePointB.affineY.toBuffer(32);
// console.log('B(x, y):', pointX_B, pointY_B);
const publicKeyB = Buffer.concat([new Buffer([0x04]), pointX_B, pointY_B]);
console.log('publicKeyB:', publicKeyB.toString('hex'));
*/

const ec = new EC('secp256k1');
const keyPairA = ec.genKeyPair();
const keyPairB = ec.genKeyPair();
console.log('keyPairA:', keyPairA, ', keyPairB:', keyPairB, ', equals: ', keyPairA == keyPairB);
console.log('keyPairA:', keyPairA.getPublic(), ', keyPairB:', keyPairB.getPublic(), ', equals: ', keyPairA == keyPairB);
console.log('keyPairA:', keyPairA.getPrivate(), ', keyPairB:', keyPairB.getPrivate(), ', equals: ', keyPairA == keyPairB);
const ehmSignature = keyPairA.sign(hashedMessage);
const derSign = ehmSignature.toDER();
console.log('encryptedHashedMessage:', derSign.map(c => c.toString(16)).join(''));
console.log('verification:', keyPairA.verify(hashedMessage, derSign));

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