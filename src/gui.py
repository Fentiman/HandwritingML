!apt-get install -y xvfb
import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfile
os.system("Xvfb :1 -screen 0 720x720x16 &")
os.environ['DISPLAY'] = ":1.0"

##GUI Initializer
window = tk.Tk()
window.geometry('200x100')
greeting = tk.Label(text="Please Upload an image file containing the text you wish to compare(Acceptable formats are JPG, PNG, etc.)",           
    width=10,
    height=5)

greeting.pack
def open_file():
    file = askopenfile(mode ='r', filetypes =[('Image Files', '*.png'),('Image Files', '*.JPG')])
 
btn = tk.Button(window, text ='Choose File', command = lambda:open_file())
btn.pack(pady = 10)

window.mainloop()
##code for file upload button that will then send to the image identifier method
