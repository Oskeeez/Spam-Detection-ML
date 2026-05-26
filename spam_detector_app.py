import pickle # To unpack the model we trained earlier

import tkinter as tk # Lets us make a GUI for the app


# Grabbing the trained model 
with open("SpamModel.pkl", "rb") as file:
    model = pickle.load(file,)

window = tk.Tk()
greeting = 

window.mainloop()
