const crypto = require('crypto-js');

const studentNumber = '2015125005';
const studentName = 'KANGJUNGSUK';

const plainText = 'ABCDEFGHIJKLMNOP';

const secretKey = studentNumber.split('').map(number => (number | 0x30).toString(2).padStart(8, '0'));

studentName.split('').forEach(char => secretKey.push(char.toString(2).padStart(8, '0')));

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