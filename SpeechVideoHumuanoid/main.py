import os
import random
import speech_recognition as sr
from gtts import gTTS
from responseml import resolveAIResponse

# Initialize the recognizer
recognizer = sr.Recognizer()
special_called = False
stranger = False

def temporaryFaceRec():
    names = ["Armstrong", "Bestman", "Stranger"]
    person = random.choice(names)

    return person

def main():
    global special_called
    global stranger

    while True:
        # Capture audio from the microphone
        with sr.Microphone() as source:
            print("Please speak something...")
            recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
            audio = recognizer.listen(source, timeout=None)  # Listen indefinitely until manually stopped

        print("Recognizing...")

        try:
            # Recognize the audio using Google Web Speech API
            text = recognizer.recognize_google(audio).lower()

            print("This is what im hearing...... ", text)

            if not special_called and ("hello special" or "hi special" or "hey special") in text:
                # Respond to "special" with a custom message
                
                # Call face recognition to determine who's speaking
                name = temporaryFaceRec()
                
                if name == "Armstrong":
                    response = "Hi Armstrong, how can I help you?"
                    special_called = True
                elif name == "Bestman":
                    response = "Hi Bestman, how can I help you?"
                    special_called = True
                else:
                    response = "I don't know who you are."
                    stranger = True

            elif special_called:
                response, info = resolveAIResponse(text)
                if info == "box":
                    response = response

            else:
                # If no recognized phrases match, instruct the user to call your name
                if stranger == True:
                    response = "I don't know who you are."
                else:    
                    response = "Hey! please call my name to talk with me."

            # Convert the response text to audio using gTTS
            tts = gTTS(response)

            # Play the generated audio (requires a program capable of playing audio files)
            tts.save("output.mp3")  # Save the generated audio to a file (optional)
            os.system("mpg123 -q output.mp3")  # Adjust the command based on your system's capabilities

        except sr.UnknownValueError:
            print("Sorry, I could not understand what you said.")
        except sr.RequestError as e:
            print(f"Sorry, an error occurred: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSpecial has been closed!")