from initialize import root, tkk, remove_children, get_hex_colour, PhotoImage, WIDTH, HEIGHT, main

#They are called pages because they all work within their own space (canvas). All the pages work the same.
#We start with the function remove children then we make a canvas we set the height, width and background
#colour. Then we pack it on the window, we create the arrow buttons for going back and forward, bind the
#backspace and return keys to a function or to nothing. Then we load the images and remove the label text
#that is default in the PhotoImage class using the customtkinter tkk.CTkLabel (This will output a warning,
#ignore it will work fine for the project) we then place the images on the window. We continue by defining
#another independent CTkLabel and write the corresponding text for each image. Then we just place the text
#and button on screen.
def page1_1():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=main, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page2_2, font = ("Arial", 20))
    
    root.bind('<Return>',lambda event:page2_2())
    root.bind('<BackSpace>',lambda event:main())

    img1 = PhotoImage(file="A_Project/images/ImportsIMG #1/Libraries.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    img2 = PhotoImage(file="A_Project/images/ImportsIMG #1/device_directory.png")
    label2 = tkk.CTkLabel(canvas, image=img2, text="")

    img3 = PhotoImage(file="A_Project/images/ImportsIMG #1/Global.png")
    label3 = tkk.CTkLabel(canvas, image=img3, text="")

    label1.place(x=50, y=50)
    label2.place(x=50, y=350)
    label3.place(x=50, y=550)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                We start by importing torch, nn, optim, F, random, re, os, incodedata, itertools and pandas.
                                Torch is a library widely used for deep learning mostly. NN is a module from torch which
                                provides tools to train neural networks. Optim is used to import various optimization
                                algorithms for training neural networks. F is used to get various activation functions or even
                                loss functions for deep learning. The random library is used to get random choices and
                                outcomes. The re library provides regular expression matching operations, for example;
                                looking for the word “hello” in a sentence. Importing os gives us a way to interact with the
                                operating system including files and folder management / manipulation. Unicodedata
                                provides access to Unicode standard character properties. Itertools is a set of fast memory
                                efficient tools part of the built-in of python. Lastly, pandas is a data manipulation library
                                normally used for working with pre structured data in python.
                                      """)
    text2 = tkk.CTkLabel(canvas, text="""
                                The device variable is used to perform operations using either the CPU or the GPU.
                                This is done to speed up training due to each device performing specific tasks best
                                than nay other device. For example: GPUs are best used for large parallel calculations.
                                The device_cpu variable is set to always use the CPU.
                                The save_dir variable has the path and the needed directories to save our files and models.
                                      """)
    text3 = tkk.CTkLabel(canvas, text="""
                                Lastly, we have a series of constants; pad, sos, eos, max_length, min_count.
                                Each of them serve their own purpose. Pad token is a value we will use to fill the data we feed
                                into our models so the mantain the same length. Sos token is the value that indicates the start
                                of the sentences we send into our models. While eos signifies the end of the sentence.
                                Each word is a token (a number that represents a word) computers don't understand words 
                                like humans do. Thus we need to convert these words into numbers, so previously this 'sentences'
                                we referred to where sentences of numbers also known as tokenized sentences. The maximum and minimum number
                                of tokens a tokenized sentence must have is defined by the MAX_LENGTH. MIN_COUNT is the minimum amount of 
                                instances in a pair we want a word to appear in a sentence for it to be used for training.
                                     """)

    text1.place(x=600, y=50)
    text2.place(x=630, y=350)
    text3.place(x=580, y=520)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page2_2():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page1_1, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page3_3, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page3_3())
    root.bind('<BackSpace>',lambda event:page1_1())

    img1 = PhotoImage(file="A_Project/images/LibraryIMG #2/Library_Init_Add.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                The class 'Library' has 3 functions which initializes the variables; name, trimmed, word2index,
                                word2count, index2word and num_words. Then we send each word in a sentence through a loop,
                                after spliting the sentence into words. Finally, if the word was not previously added to word2inedx,
                                then we can proceed to add it in the dictionaries words2index where the the word will be set with a
                                number, word2count where each word will be set as 1, index2word where the number will be assigned to
                                a word. Then we add 1 to the variable num_words. Else if the word is in word2index then we add 1 to the 
                                word's number in word2count.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page3_3():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page2_2, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page4_4, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page4_4())
    root.bind('<BackSpace>',lambda event:page2_2())

    img1 = PhotoImage(file="A_Project/images/LibraryIMG #2/Trim.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                The trim function first checks if the boolean variable 'trimmed' is True it will
                                then proceed to return nothing if trimmed is true else it will continue the
                                following operation. It will set trimmed as True, initialise the array "keep_words",
                                we continue by looping through each item in word2count which includes the word and
                                it's count(k and v). In the loop if the count is bigger or equal than the constant MIN_COUNT
                                then append the word to the keep_words, then print the total number of items in keep words followed
                                by a '/' then the total number of items in word2index followed by an '=' and the result of both
                                numbers divided. We then initlise like before word2index, word2count, index2word and num_words.
                                Finally, for every word in keep_words we send it to the addWord procedure.
                                      """)

    text1.place(x=500, y=550)
    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page4_4():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page3_3, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page5_5, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page5_5())
    root.bind('<BackSpace>',lambda event:page3_3())

    img1 = PhotoImage(file="A_Project/images/Norm-Read #3/Ascii_Normalize.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                The function unicodeToAscii simply returns a . Whereas, the function normalizeString simply
                                takes a string (s) sends the sentence to unicodeToAscii after converting the characters to 
                                lowercase and removing any whitespaces we will continue by using the returned data to look
                                and remove various patterns and symbols e.g. #, %, !, @, ., /, email addresses, tags, etc and
                                again removing any whitespaces. Finally, the final product of this function is returned.
                                      """)
    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page5_5():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page4_4, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page6_6, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page6_6())
    root.bind('<BackSpace>',lambda event:page4_4())

    img1 = PhotoImage(file="A_Project/images/Norm-Read #3/ReadVocs.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="")

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                In the readVocs function we define the array 'pairs' load and read the data on the datafile given path
                                we then take all the sentences in the "question" section of the dataset into the 'questions' variable,
                                then we take all the sentences in the "response" section of the dataset into the 'responses' variable,
                                we continue by looping each sentence in the questions and responses then send them to the normalizeString
                                function, then initialise both a pair array and add the returning setences into and append the pair in pairs.
                                Finally, we return the pairs array.
                                      """)
    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page6_6():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page5_5, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page7_7, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page7_7())
    root.bind('<BackSpace>',lambda event:page5_5())

    img1 = PhotoImage(file="A_Project/images/LoadFil-Trim #4/Filter_Prepare.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"
    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                The filterPair returns True if both parts of the pair are lower than the MAX_LENGTH else it returns False.
                                The filterPairs returns the pairs which have met the conditions. Using a for loop to send each pair to be
                                checked in the "filterPair". The loadPrepareData will initialise the "Library" class then we will send the
                                dataset's path in readVocs where we will recive the pairs then they are sent to the filterPairs and recive
                                the filtrated pairs. We then loop thorough each pair from the returned pairs, after the filterPairs function.
                                In the loop we send both items in the pair to the addSentence function. Outside of the loop we just return
                                the initilised Library instance (libra) and the filtrated pairs.
                                      """)
    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page7_7():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page6_6, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page8_8, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page8_8())
    root.bind('<BackSpace>',lambda event:page6_6())

    img1 = PhotoImage(file="A_Project/images/LoadFil-Trim #4/TrimRareWords.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"
    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                In the trimRareWords function we run the trim procedure in the Library, we continue by initialising
                                the array keep_pairs then for each pair in pairs we get their question sentence (input_sentence) and response 
                                sentence (output_sentence) then we initialise the following boolean variables keep_input and keep_output as True,
                                to then proceed to loop through each word in the input_sentence and output_sentence after spliting word by word
                                with in the loop if the word is not in the Library's word2index then keep_input and keep_output are set as false
                                and we break the loop. After the 2nd loop and within the main loop If the keep_input and keep_output are True then
                                we append the pair to the keep_pairs array.
                                      """)
    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page8_8():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page7_7, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page9_9, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page9_9())
    root.bind('<BackSpace>',lambda event:page7_7())

    img1 = PhotoImage(file="A_Project/images/InOu-ipadBin #5/IndexPaddingBinary.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="")


    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                In the indexesFromSentence return the index of each word in the sentence in their index representation.
                                The zeroPadding function will aim to make all the elements in the array (l) be of the same length as the
                                longest item in the array e.g. 
                                [[1,23,4],[2,4],[1,34,3],[1,2,34,92,6]...] --> [[1,23,4,0,0],[2,4,0,0,0],[1,34,3,0,0],[1,2,34,92,6]...]
                                The function binaryMatrix initialises an array (m) we then loop through each sequence in the sequences (l) and
                                we also keep count of the loop, we continue by appending an empty list into m then start another loop in
                                which for each token in the sequence if the token represents the PAD_token then for each count of the previous
                                loop in the array m append 0 otherwise append 1 outside of the loops return the array m.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page9_9():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page8_8, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page10_10, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page10_10())
    root.bind('<BackSpace>',lambda event:page8_8())

    img1 = PhotoImage(file="A_Project/images/InOu-ipadBin #5/Input_Output.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                In the inputVar function for each sentence in sentences (l) then send the sentence and library
                                to the indexesFromSentence function. With the returned data we loop it's contents and for each
                                sequence we find the number of tokens in them then after the loop we convert the array containing
                                all the lengths into a tensor. We continue by sending the returned data now to the zeroPadding
                                function returning a padList of all the sentences then this is stored in a long tensor. Finally,
                                we just return the long tensor and the tensor with the lengths.
                                The outputVar function has a similar procedure where we get the lengths we will instead get the
                                largest length by using the function 'max' after zeroPadding we will send the padList to the 
                                binaryMatrix returning a binary mask which will then be stored as a bool tensor. Finally we will
                                return the long tensor, mask and largest length.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page10_10():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page9_9, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page11_11, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page11_11())
    root.bind('<BackSpace>',lambda event:page9_9())

    img1 = PhotoImage(file="A_Project/images/Bacth2Train #6/BatchCreation.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="")


    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                The batch2TrainData function will first sort the batch of pairs in decending order from largest to smallest pair
                                then we initialise the arrays input_batch and output_batch the for each pair in the pair batch the first sentence
                                is appended to input_batch and the other is appended to output_batch. We continue by sending the input_batch and
                                the library to the inputVar function we do the same with the output_batch in the outputVar function. Finally,
                                we can return the data recived from the inputVar and outputVar function.
                                      """)
    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page11_11():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page10_10, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page12_12, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page12_12())
    root.bind('<BackSpace>',lambda event:page10_10())

    img1 = PhotoImage(file="A_Project/images/Encoder #7/Encoder_init.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                The Encoder class is initialised with the following n_layers, hidden_size, embedding and gru
                                the n_layers holds the number of layers that will be used. The hidden_size hold the number of
                                continuos vector representations. The embedding is an instance containing the Embedding class.
                                The gru is an instance of the GRU class which is initialized with hidden_size the n_layers and 
                                dropout with a bidirectional set as True.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page12_12():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page11_11, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page13_13, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page13_13())
    root.bind('<BackSpace>',lambda event:page11_11())

    img1 = PhotoImage(file="A_Project/images/Encoder #7/Encoder_forward.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                The forward of the Encoder class starts with initialising the continuos vector for the input_sequence
                                which is then stored as 'embedded' then we send this and input_lengths to nn.utilis.rnn.pad_packed_sequence
                                returning a packed data, is used to avoid uneccessary computations by hiding the pad tokens. The packed data
                                is then used in the gru computation for the reset gate and update gate thought. The reset gate determines
                                the number of the continuos vectors we will forget while the update gate determines the number of continuos
                                vector representations should be updated. We shall not dive any deeper than this as it would be too complicated
                                and confusing. In simple terms, they are essentialy designed to capture data and patterns along a sequence and
                                produce a prediction output with that information The nn.utilis.rnn.pad_packed_sequence will then show again the
                                padded values. We proceed by slicing the outputs twice and adding them together basically we are getting the first
                                hidden_size and the last hidden_size  resulting in hidden_size as after doing gru the hidden_size is updated so this
                                method makes sure the shape stays constant else it could rise errors. Finally, we can just return the output and hidden.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page13_13():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page12_12, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page14_14, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page14_14())
    root.bind('<BackSpace>',lambda event:page12_12())

    img1 = PhotoImage(file="A_Project/images/Attention #8/Attn_init.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"
    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                The attention init initialises the hidden_size and attn which is an instance of a linear layer which takes
                                an input size of vector representations and returns an output_size in our case is just the same size in both
                                cases.
                                      """)
    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page14_14():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page13_13, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page15_15, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page15_15())
    root.bind('<BackSpace>',lambda event:page13_13())

    img1 = PhotoImage(file="A_Project/images/Attention #8/attn_score.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                The general score function of the Attn class is uses the attn to apply linear transformation on the
                                continuos vector representation of the encoder's output. We then take the result of this to multiply
                                it by it's own hidden (memory containing information) then along their 2nd dimension we sum and return
                                the result.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page15_15():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page14_14, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page16_16, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page16_16())
    root.bind('<BackSpace>',lambda event:page14_14())

    img1 = PhotoImage(file="A_Project/images/Attention #8/attn_forward.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                In the forward function of the Attn class we send the hidden and encoder's output to the general_score
                                function then we transpose the returning data e.g. [x,y] --> [y,x] then return the data after applying softmax
                                on the data along the first dimension, then unsquezze along the 1 dimension (adding an extra dimension).
                                """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page16_16():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page15_15, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page17_17, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page17_17())
    root.bind('<BackSpace>',lambda event:page15_15())

    img1 = PhotoImage(file="A_Project/images/Decoder #9/decoder_init.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                In the decoder class we initialise everything seen in the encoder class but only there are only 4
                                new variants the Dropout Neural Network class the Attn instance which has nothing new of what
                                we've already explained is just not initialised in the encoder and the dropout which is just a value 
                                from 0 to 1 as it serves as a probability. The dropout class instance is used to randomly get some vectors
                                within a percentage defined by the dropout inwhich these values will then be set to 0 regardless of their 
                                importance. This is done to avoid the model trying to depend on certain features or and train all its
                                parameters to make sure is not only focusing all the changes in some areas aswell as it's an effective
                                solution to the problem of overfitting the model. Learning to well the dataset that when facing new data
                                it performs badly. And finally the output_size which is the total vocabulary_size (all the words the model
                                is supposed to know).
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page17_17():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page16_16, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page18_18, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page18_18())
    root.bind('<BackSpace>',lambda event:page16_16())

    img1 = PhotoImage(file="A_Project/images/Decoder #9/decoder_forward.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="")

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                In the decoder's forward function we will skip what we've seen in the encoder since its quite similar
                                the embedding_dropout is applying the dropout. The attn_weight.bmm is basically a batch matrix multiplication
                                which computes the weighted sum of the encoder output based on the weights recived after the attention function
                                the squeeze operations seen after are basically done to remove not needed dimensions.
                                torch.cat is used to concatenate the probabilities gotten from the gru (rnn_output) and the context archived
                                from the encoder's output and the attention weights then using the data to perform a tanh operation which is
                                used to transform numerical values into bounded range e.g. -1 to 1 allowing us to have our probabilities as
                                [0.233, -0.7832, 0.9233 ... ]. Then we finally perform a linear then softmax operation to then return the output
                                probabilities and hidden state.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page18_18():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page17_17, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page19_19, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page19_19())
    root.bind('<BackSpace>',lambda event:page17_17())

    img1 = PhotoImage(file="A_Project/images/Training #10/Trainfunc.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                In the train function we start by zeroing the optimizer's gradient using zero_grad. The gradients are
                                computed during the loss backpropagation in which the like the forward functions we've been seeing
                                in the encoder, decoder and attention we also have backward passes where the gradients are computed or
                                updated using optimization algorithms. Their aim is to help us find optimal solutions by adjusting the
                                models parameters (more specific; the weights and biases). The gradient is a vector that points to the
                                direction of the steepest increase of a function, they help for optimization, learning and understanding
                                how functions (e.g. loss function) change with respect to the inputs they are given. The input_variable
                                and target_variable are what we saw before the question and answer respectively then we also have the mask
                                and length these are being sent to a device (cpu, gpu, etc) for processing, diffrent devices are best for
                                diffrent things thus some can help speed up certain parts of training.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page19_19():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page18_18, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page20_20, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page20_20())
    root.bind('<BackSpace>',lambda event:page18_18())

    img1 = PhotoImage(file="A_Project/images/Training #10/En_out-.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                Still in the train function. We initialise loss and n_totals as equal to 0 then print_losses as an empty array
                                we then send the input variable and lengths to the encoder, returning the output and hidden. Then we create a
                                long tensor which containts a batch of items this is then also sent to the device used. We then slice the hidden
                                of the encoder to match the number of layers in the decoder to avoid mismatch errors. We continue by randomly
                                choosing a number between 0 and 1 in which if it's larger than the value in teacher_forcing_ratio then the boolean
                                use_teacher_forcing is set as True else is set as False.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page20_20():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page19_19, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page21_21, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page21_21())
    root.bind('<BackSpace>',lambda event:page19_19())

    img1 = PhotoImage(file="A_Project/images/Training #10/target-pair.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                Still in the training function. If use_teacher_forcing is true then for in range
                                of the maximum expected answer length it will loop sending the decoder input,
                                the decoder_hidden and encoder_outputs to be processed by the decoder then we
                                define the decoder_input again as the target_vaiable in the current time step of
                                the loop then flattens the target e.g. [[12,3,4],[32,3,54]] --> [[12,3,4,32,3,54]]
                                then we send the decoder_output, target_variable and mask, for these last 2, we get
                                the items from the current time step of the loop, so we don't send it all at once
                                to the maskNLLLoss function. This will return the mask_loss and nTotal then now we
                                add the loss variable with the mask_loss, append the item times the nTotal to the
                                print_losses and add nTotal to the n_totals.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page21_21():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page20_20, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page22_22, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page22_22())
    root.bind('<BackSpace>',lambda event:page20_20())

    img1 = PhotoImage(file="A_Project/images/Training #10/decoder_inp-target.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                if use_teacher_forcing is false then we do the same we did in the last page just with a few minor changes
                                after we get the decoder's outputs. Then from the decoder output we look for the most probable prediction
                                (topi),  we continue by creating a long tensor taking the top index from each item in the batch. Then this
                                tensor (decoder_input) is moved to the device. 
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page22_22():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page21_21, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page23_23, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page23_23())
    root.bind('<BackSpace>',lambda event:page21_21())

    img1 = PhotoImage(file="A_Project/images/Training #10/back, step,clip.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                Still in the train. We define the loss backward propagation which we have explained how it works previously, then we also have
                                gradient clipping, in which we do not allow the gradient be larger than a certain value thus able to control
                                the learning of the model to not have large gradients if not it could be very difficult for the model to 
                                pinpoint the optimal changes needed. We then continue by steping the optimizers which is basically just updating
                                the encoder and decoder parameters to get better at reducing loss. Then we finally return a sum of all the items
                                in print_losses divided by n_totals.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page23_23():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page22_22, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page24_24, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page24_24())
    root.bind('<BackSpace>',lambda event:page22_22())

    img1 = PhotoImage(file="A_Project/images/TrainIters #11/create_batch.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                In the trainIters function we start by printing "Creating the training batches..." then for in range
                                of batch_size we choose a pair at random then after the loop we send the Library instance and
                                batch of pairs into the batch2TrainData function then this is done for in range of n_iterations.
                                After this we continue by initilising some variables start_iteration as integer 1 then print_loss,
                                old and tries as 0. Next if loadFilename is True, meaning it has something, then from the checkpoint
                                load its 'time' data and represent it with the tries variable.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page24_24():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page23_23, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page25_25, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page25_25())
    root.bind('<BackSpace>',lambda event:page23_23())

    img1 = PhotoImage(file="A_Project/images/TrainIters #11/init_training.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                Still within the trainIters function.  We print "Initalizing Training..." then for in range of start_iteration
                                to n_iteration + 1 we loop through getting the training_batch from the training_batches by iterating through it
                                starting from iteration - 1 and the "iteration" will just continue increasing. Then from the training_batch we
                                unpack all the variables we return from the batch2TrainData. Next we send to the train function; The unpacked
                                variables, the encoder, decoder, optimizers, clip and batch_size to then recive the loss which is added to 
                                print_loss.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page25_25():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button1 = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page24_24, font = ("Arial", 20))
    button = tkk.CTkButton(canvas, text="<", width=50, height=40, command=page26_26, font = ("Arial", 20))
    root.bind('<Return>',lambda event:page26_26())
    root.bind('<BackSpace>',lambda event:page24_24())

    img1 = PhotoImage(file="A_Project/images/TrainIters #11/print_loss.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                if iteration is a multiple of print_every then print_loss divided by print_every is equal to print_loss_avg then
                                print the current iteration, the percentage left until the end of the iteration and the average loss and
                                set print_losses as 0.
                                      """)

    text1.place(x=500, y=500)

    button.place(x=5,y=5)
    button1.place(x=1311,y=5)

def page26_26():
    remove_children()
    canvas = tkk.CTkCanvas(root,
    width = WIDTH,
    height = HEIGHT,
    bg = get_hex_colour(root)
    );canvas.pack()

    button = tkk.CTkButton(canvas, text=">", width=50, height=40, command=page25_25, font = ("Arial", 20))
    root.bind('<BackSpace>',lambda event:page25_25())
    root.bind('<Return>',lambda event:None)

    img1 = PhotoImage(file="A_Project/images/TrainIters #11/save_model.png")
    label1 = tkk.CTkLabel(canvas, image=img1, text="") # text must be empty to remove the water mark "CTkLabel"

    label1.place(x=50, y=50)
    
    text1 = tkk.CTkLabel(canvas, text="""
                                if the iteration is within the mutiples of save_every then add save_every to tries and we define the directory
                                path and name if there are no similar directories then "makedirs" creates this directory then we save the model
                                using torch.save. Saving it's internal parameters, loss, optimizeer, library, embedding, encoder, decoder, trained time
                                then we specify the name of the checkpoint and where to save it (directory).
                                      """)

    text1.place(x=500, y=500)

    button.place(x=1311,y=5)