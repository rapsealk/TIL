const cryptoJS = require('crypto-js');
const ecc = require('eccjs');

/**
 * Cryptograph
 * @author rapsealk
 * Symmetric-Key Algorithm: AES (Advanced Encryption Standard)
 * Asymmetric-Key Algorithm: ECC (Elliptic Curve Cryptography)
 * Hash Algorithm: SHA-256 (Secure Hash Algorithm)
 */

const studentNumber = '2015125005';
const studentName = 'KANGJUNGSUK'.slice(0, 6);

// Get PlainText
const plainText = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
console.log('[Stage 01] Plain Text:', plainText);

// Generate Hash Key
const hashKey = studentNumber.split('').map(number => (number | 0x30).toString(2).padStart(8, '0'));
studentName.split('').forEach((char, i) => hashKey.push(studentName.charCodeAt(i).toString(2).padStart(8, '0')));
console.log('hashKey:', hashKey.join(''));

// Flip the MSB of Hash Key
let flippedHashKey = hashKey.join('');
flippedHashKey = (flippedHashKey.slice(0, 1) ^ 0x01) + flippedHashKey.slice(1);
console.log('flippedHashKey:', flippedHashKey);

// Hash plain text
const hashedMessage = cryptoJS.SHA256(plainText, hashKey.join('')).toString();
console.log('[Stage 02] Hashed Message:', hashedMessage);

console.log('HmacSHA256:', cryptoJS.HmacSHA256(plainText, hashKey.join('')).toString());
console.log('fHmacSHA256:', cryptoJS.HmacSHA256(plainText, flippedHashKey).toString());

// Hash plain text with flipped hash key
const fHashedMessage = cryptoJS.SHA256(plainText, flippedHashKey).toString();
cryptoJS.SHA256()
console.log('fHashed Message:', fHashedMessage);

console.log('hashed === fhashed:', hashedMessage === fHashedMessage);
process.exit(0);

// Avalanche Effect
console.log('plainText.:', plainText + '.');
console.log('hashedMessage.:', cryptoJS.SHA256(plainText+'.').toString());

// Encrypt plain text with AES Algorithm
const aesKey = 'AES_KEY';
const cipherText = cryptoJS.AES.encrypt(plainText, aesKey).toString();
console.log('[Stage 03] Cipher Text:', cipherText);

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
// console.log('signedHashedMessage:', signature);
const encAESKey = ecc.encrypt(keys.public.B, aesKey);
console.log('[Stage 04] Encrypted AES Key:', JSON.parse(encAESKey).tag);

// B receives Encrypted AES Key
const decAESKey = ecc.decrypt(keys.private.B, encAESKey);
console.log('[Stage 05] Decrypted AES Key:', decAESKey);

const decipherText = cryptoJS.AES.decrypt(cipherText, decAESKey).toString(cryptoJS.enc.Utf8);
console.log('[Stage 06] Decipher Text:', decipherText);

const hashedDecipherMessage = cryptoJS.SHA256(decipherText).toString();
console.log('[Stage 07] Hashed Decipher Message:', hashedDecipherMessage);

const isPrivateA = ecc.verify(keys.public.A, signature, hashedDecipherMessage);
console.log('[Stage 08] isPrivateA:', isPrivateA);