import os
import time
import customtkinter as tkk
from tkinter import Tk, PhotoImage, Label
from CTkMessagebox_main.CTkMessagebox import CTkMessagebox # Library from https://github.com/Akascape/CTkMessagebox

s = []

root = tkk.CTk()
root.title("Class Lab")

# Takes the computer display width and height
WIDTH = root.winfo_screenwidth() 
HEIGHT = root.winfo_screenheight()

#Defines the window size with the width and height
root.geometry(f"{WIDTH}x{HEIGHT}")

#https://stackoverflow.com/questions/58758439/how-can-i-get-hex-or-rgb-color-code-of-the-window-background-color
def get_hex_colour(window):
    # Get the background color of the window
    bg_color = window.cget("bg")

    # Convert the color to hexadecimal format
    hex_colour = window.winfo_rgb(bg_color)
    hex_colour = "#{:02x}{:02x}{:02x}".format(hex_colour[0] // 256, hex_colour[1] // 256, hex_colour[2] // 256)

    return hex_colour

def set_mode():
    mode = tkk.get_appearance_mode() # get the current background colour
    if mode == "Light": # if is light set it to dark
        tkk.set_appearance_mode("Dark")
    elif mode == "Dark":# if is dark set it to light
        tkk.set_appearance_mode("White")

mode_button = tkk.CTkButton(root, text="Mode", command=set_mode) # Create a button to switch background colours
mode_button.place(x=600, y=5)# Place the button on screen


# Remove the all the on screen widgets of the window.
def remove_children():
    children = root.winfo_children()
    for child in children:
        child.destroy()

# Main area to access the diffrent sections of the window.
def plaza():
    from theory import page1_1
    from talk import chat

    root.bind('<Return>',lambda event: None) # the return key is set to no command
    root.bind('<BackSpace>',lambda event: None) # the backspace key is set to no command

    button1 = tkk.CTkButton(root, text="Theory", width=600, height=700, command=page1_1)
    button2 = tkk.CTkButton(root, text="Talk", width=600, height=700, command=chat)

    button1.grid(row = 0, column = 1, padx=50, pady  = 0)
    button2.grid(row = 0, column = 2, padx=0, pady  = 0)

#Main displays the diffrent messageboxes depending if text is True or False then it also re-directs to the plaza.
def main(text="True"):
    remove_children()
    s.append(text)
    plaza()
    if text == "True" and len(s) == 1:
        CTkMessagebox(title="Right Password", message="Welcome!", icon="check")
    elif text == "False":
        CTkMessagebox(title="Set", message="Entry text will become the password", icon="check")