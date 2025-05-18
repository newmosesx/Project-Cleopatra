from initialize import root, tkk, remove_children, CTkMessagebox, main, os

#the function is folder empty will look into the given directory and if there is no files or directories then
#it will return True.
def is_folder_empty(folder_path):
    return not any(os.listdir(folder_path))

#The checking function checks if a folder is empty if it is then newUser is ser as False else newUser is set as None.
#then return newUser.
def checking(password):
    folder_path = "A_Project/passwords"

    if is_folder_empty(folder_path):
        with open(f"A_Project/passwords/{password}","w") as bool:
            newUser = False
    else:
        newUser = None
    return newUser

#The search user goes to a directory and looks for a password and store it as text. If there is no such password instead of
#rasing an error display a messagebox with a message then send the password back to the checking function if the output of it
#is True then text is true else if is Flase then text is False, then send the text to main.
def search_password(password):
    new = True
    try:
        with open(f"A_Project/passwords/{password}","r") as current_password:
            text = current_password.read()
    except FileNotFoundError as FNFerror:
        CTkMessagebox(title="No User", message="This password is not the right one. Try Again.", icon="cancel")
        new = None
        new = checking(password)
    if new == True:
        text = "True"
    elif new == False:
        text = "False"
    main(text)


#Login defines the labels, entry, enter button and string variable for the login section.
#The string variable is used to store the entry's text. We bind the return key to the 
#search_user function while sending the entry's text too. We then just place the button.
#entry and labels on screen.
def joinIn():
    remove_children()

    comeL = tkk.CTkLabel(root, text="Come on In! We are waiting!", font=("Arial",50))
    passL = tkk.CTkLabel(root, text="Password")

    text = tkk.StringVar()
    entry = tkk.CTkEntry(root, textvariable=text, width=300, height=30)

    enter = tkk.CTkButton(root, text="Join", command=lambda: search_password(text.get()))
    root.bind("<Return>", lambda event: search_password(text.get()))

    comeL.grid(row = 0, column = 1, padx=0, pady  = 150)
    enter.grid(row = 2, column = 1, padx=0, pady  = 0)
    entry.grid(row = 1, column = 1, padx = 525, pady  = 50)
    
    passL.place(x=640, y=360)



#Welcome label text
title = tkk.CTkLabel(root, text="Welcome", font = ("Arial", 100))


#Starting button
start_button = tkk.CTkButton(root, text="Enter", command=joinIn, corner_radius=10, width=200, height=50)

#place label text and starting button on screen.
title.grid(row = 0, column = 0, padx=0, pady  = 100)
start_button.grid(row = 1, column = 0, padx=590, pady  = 150)


#loop the window.
root.mainloop()