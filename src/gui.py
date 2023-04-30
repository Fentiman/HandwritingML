import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfile

def open_file():
        file = askopenfile(mode ='r', filetypes =[('Image Files', '*.png'),('Image Files', '*.JPG')])
        get_char(file)
 
LARGEFONT =("Verdana", 35)

class MainApp(tk.Tk):
     
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
         
        # __init__ function for class Tk
        tk.Tk.__init__(self, *args, **kwargs)
         
        # creating a container
        container = tk.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)
  
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        # initializing frames to an empty array
        self.frames = {} 
  
        # iterating through a tuple consisting
        # of the different page layouts
        for F in (StartPage, Results):
  
            frame = F(container, self)
  
            # initializing frame of that object from
            # startpage, page1, page2 respectively with
            # for loop
            self.frames[F] = frame
  
            frame.grid(row = 0, column = 0, sticky ="nsew")
  
        self.show_frame(StartPage)
  
    # to display the current frame passed as
    # parameter
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
 
  
# first window frame startpage
  
class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
         
        # label of frame Layout 2
        label = ttk.Label(self, text ="Handwriting Comparator", font = LARGEFONT)
    
        # putting the grid in its place by using
        # grid
        label.grid(row = 1, column = 2, padx = 10, pady = 10)
        
        greeting = tk.Label(self, text="Please Upload an image file containing the text you wish to compare(Acceptable formats are JPG, PNG, etc.)")
        greeting.grid(row = 2, column = 2, padx = 10, pady = 10)
  
        button1 = ttk.Button(self, text ="Open File",
        command = lambda : [open_file(), controller.show_frame(Results)])
     
        # putting the button in its place by
        # using grid
        button1.grid(row = 3, column = 2, padx = 10, pady = 10)
  
          
  
  
# second window frame page1
class Results(tk.Frame):
        
    def __init__(self, parent, controller):
         
        tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text ="Results", font = LARGEFONT)
        label.grid(row = 1, column = 2, padx = 10, pady = 10)
          
        # button to show frame 2 with text
        # layout2
        button1 = ttk.Button(self, text ="Return to Menu",
                            command = lambda : controller.show_frame(StartPage))
     
        # putting the button in its place
        # by using grid
        button1.grid(row = 3, column = 2, padx = 10, pady = 10)
        
        
  
  
# Driver Code
app = MainApp()
app.mainloop()
