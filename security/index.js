const cryptoJS = require('crypto-js');
const ecc = require('eccjs');

/**
 * Cryptograph
 * @author rapsealk
 * Symmetric-Key Algorithm: AES (Advanced Encryption Standard)
 * Asymmetric-Key Algorithm: ECC (Elliptic Curve Cryptography)
 * Hash Algorithm: SHA-256 (Secure Hash Algorithm)
 */

 // [Problem 1] Generate Hash Key
const studentNumber = '2015125005';
const studentName = 'KANGJUNGSUK'.slice(0, 6);
const hashKey = studentNumber.split('').map(number => (number | 0x30).toString(2).padStart(8, '0'));
studentName.split('').forEach((char, i) => hashKey.push(studentName.charCodeAt(i).toString(2).padStart(8, '0')));
console.log('[Problem 01] hashKey:', hashKey.join(''));

// [Problem 2] Get PlainText
const plainText = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
console.log('[Problem 02] Plain Text:', plainText);

// [Problem 3] Encrypt plain text with AES Algorithm
const aesKey = 'AES_KEY';
const cipherText = cryptoJS.AES.encrypt(plainText, aesKey).toString();
console.log('[Problem 03] Cipher Text:', cipherText);

// Flip the MSB of plain text
const fplainText = String.fromCharCode(plainText.charCodeAt(0) ^ 0x01) + plainText.slice(1);
console.log('fplainText:', fplainText);

// [Problem 05]
const fcipherText = cryptoJS.AES.encrypt(fplainText, aesKey).toString();
console.log('[Problem 05] fcipher text:', fcipherText);

const cipherTextInBits = cipherText.split('').map((t, i) => cipherText.charCodeAt(i).toString(2).padStart(8, '0')).join('');
const fcipherTextInBits = fcipherText.split('').map((t, i) => fcipherText.charCodeAt(i).toString(2).padStart(8, '0')).join('');
let bit_match_ratio = 0;
cipherTextInBits.split('').forEach((bit, index) => bit_match_ratio += (bit == fcipherTextInBits[index]));
console.log(`bit_match_ratio: ${bit_match_ratio - 256} / ${cipherTextInBits.length - 256}`);

// Hash plain text
const hashedMessage = cryptoJS.HmacSHA256(plainText, hashKey.join('')).toString();
console.log('Hashed Message:', hashedMessage);

// Avalanche Effect
console.log('plainText.:', plainText + '.');
console.log('hashedMessage.:', cryptoJS.HmacSHA256(plainText+'.', hashKey.join('')).toString());

// Generate ECC key pair
const keyPairA = ecc.generate(ecc.SIG_VER);
const keyPairB = ecc.generate(ecc.SIG_VER);
const keys = {
    public: {
        A: keyPairA.ver,
        B: keyPairB.ver
    },
    private: {
        A: keyPairA.sig,
        B: keyPairB.sig
    }
};

const signature = ecc.sign(keys.private.A, hashedMessage);
const encAESKey = ecc.encrypt(keys.public.B, aesKey);
console.log('Encrypted AES Key:', JSON.parse(encAESKey).tag);

// B receives Encrypted AES Key
const decAESKey = ecc.decrypt(keys.private.B, encAESKey);
console.log('Decrypted AES Key:', decAESKey);

// [Problem 4]
const decipherText = cryptoJS.AES.decrypt(cipherText, decAESKey).toString(cryptoJS.enc.Utf8);
console.log('[Problem 04] Decipher Text:', decipherText);

const hashedDecipherMessage = cryptoJS.HmacSHA256(decipherText, hashKey.join('')).toString();
console.log('Hashed Decipher Message:', hashedDecipherMessage);

const isPrivateA = ecc.verify(keys.public.A, signature, hashedDecipherMessage);
console.log('isPrivateA:', isPrivateA);