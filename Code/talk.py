from initialize import tkk, root, remove_children, main

#We create a chat procedure where we define a textbox's shape, it's read and write status, border, text and 
#background colour, font style and size, the use of a scrollbar and its colour. We then place the textbox 
#on screen.
def chat():
    remove_children()

    textbox = tkk.CTkTextbox(root,
	width=600,
	height=300,
	corner_radius=50,
	border_width=20,
    state = tkk.DISABLED,
	border_color="#003660",
	border_spacing=10,
	fg_color="white",
	text_color="black",
	font=("Helvetica", 18),
	wrap="word", 
	activate_scrollbars = True,
	scrollbar_button_color="silver",
	scrollbar_button_hover_color="red",
	)

    textbox.place(x=400, y=150)

    def send_message():
        #Get the text in the entry
        message = entry.get()
        
        #if there is text continue with;
        if message:
            #We allow writing in the textbox
            textbox.configure(state=tkk.NORMAL)
            #We then insert the users message
            textbox.insert(tkk.END, "You: " + message +  "\n", "user")
            #We disable the wrinting in the textbox
            textbox.configure(state=tkk.DISABLED)
            #We delete the text in the entry
            entry.delete(0, tkk.END)

            
            response = "Thanks for your message!"

            #We allow writing in the textbox
            textbox.configure(state=tkk.NORMAL)
            #We then insert the models message
            textbox.insert(tkk.END, "Model: " + response + "\n", "bot")
            #We disable the wrinting in the textbox
            textbox.configure(state=tkk.DISABLED)

    # Create an Entry widget for typing messages
    entry = tkk.CTkEntry(root, width=200)
    
    # We bind the necessary keys to the corresponding function
    root.bind("<Return>", lambda event: send_message())
    root.bind('<Right>',lambda event: main())

    # We create a button for sending messages
    send_button = tkk.CTkButton(root, text="Send", command=send_message)
    arrow1 = tkk.CTkButton(root, text=">", width=50, height=40, command=main, font = ("Arial", 20))


    # Then we place the textbox, entry and send_button using grid
    textbox.grid(row=0, column=0, columnspan=2, padx=10, pady=10)
    entry.grid(row=1, column=0, padx=10, pady=5)
    send_button.grid(row=1, column=1, padx=10, pady=5)
    
    # We place the arrow button on the screen.
    arrow1.place(x=1311,y=5)
