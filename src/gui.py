import tkinter as tk
import tkinter.ttk as ttk

##GUI Initializer
window = tk.Tk()
greeting = tk.Label(text="Please Upload an image file containing the text you wish to compare(Acceptable formats are JPG, PNG, etc.)"            
    text="Hello, Tkinter",
    width=10,
    height=5)

greeting.pack
window.mainloop()
##code for file upload button that will then send to the image identifier method
