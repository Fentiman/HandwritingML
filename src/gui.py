import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import askopenfilename

##GUI Initializer
window = tk.Tk()
window.geometry('200x100')
greeting = tk.Label(text="Please Upload an image file containing the text you wish to compare(Acceptable formats are JPG, PNG, etc.)"            
    text="Hello, Tkinter",
    width=10,
    height=5)

greeting.pack
def open_file():
    file = askopenfile(mode ='r', filetypes =[('Python Files', '*.py')])
 
btn = Button(root, text ='Choose File', command = lambda:open_file())
btn.pack(side = TOP, pady = 10)

window.mainloop()
##code for file upload button that will then send to the image identifier method
