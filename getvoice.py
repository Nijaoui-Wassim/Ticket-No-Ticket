import speech_recognition as sr

def get_voice():
    r = sr.Recognizer()
    r.energy_threshold=4000

    print("say something")

    with sr.Microphone() as source:                # use the default microphone as the audio source
        audio = r.listen(source)                   # listen for the first phrase and extract it into audio data

    try:
        print("You said " + r.recognize_google(audio))    # recognize speech using Google Speech Recognition
        return str(r.recognize_google(audio))
    except LookupError:                            # speech is unintelligible
        print("Could not understand audio")
    
if __name__ == '__main__':
    while True:
        try:
            get_voice()
        except:
            print("Could not understand audio")
