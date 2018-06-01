import scipy.io.wavfile as wav

if __name__ == '__main__':
    rate, signal = wav.read('https://firebasestorage.googleapis.com/v0/b/kaubrain418.appspot.com/o/wav%2F1525765715519.wav?alt=media&token=5dd3f236-0a0f-4755-8080-327c8da81b9e')
    print(rate)
    print(signal)