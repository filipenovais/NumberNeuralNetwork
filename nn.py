import numpy
import matplotlib.pyplot
import scipy.special
import scipy.ndimage
from skimage.transform import resize
from PIL import ImageTk, Image, ImageDraw
from tkinter import *
from skimage import color
from skimage import io
import PIL
from PIL import Image
from pyscreenshot import grab


# ************************** Neural Network **************************


class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        self.lr = learningrate

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):

        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))


        pass

    def query(self, inputs_list):

        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)


        return final_outputs


        pass


input_nodes = 784
hidden_nodes = 200
output_nodes = 10

learning_rate = 0.01


n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

print('Generating Neural Network...')
epochs = 50
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

        # rotated anticlockwise by 10 degrees
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        # rotated clockwise by 10 degrees
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)


        pass
    pass

print('FINISHED NN!!')

# ************************** TKINTER **************************


def save():
    
    global draw, image1

    im = grab(bbox=(490, 325, 810, 670))
    im.save("number.png")
    im.show()
    imgarray = numpy.array(image1)

    image1 = PIL.Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image1)

    pil_im = Image.open('number.png')
    pil_imgray = pil_im.convert('L')

    img = numpy.array(list(pil_imgray.getdata(band=0)), float)
    img.shape = (pil_imgray.size[1], pil_imgray.size[0])
    img = resize(img, (28,28))
    inputs = (numpy.asfarray(img)) - 255
    inputs = numpy.absolute(inputs)
    inputs = (inputs / 255.0 * 0.99) + 0.01
    inputs = inputs.ravel()
    #print(inputs)
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)

    predicted_label.config(text="The predicted number is: " + str(label))

    image_array = numpy.asfarray(inputs).reshape((28,28))
    matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')

    matplotlib.pyplot.show()

def cleanfunc():
    global cv, root

    cv.destroy()
    cv = Canvas(root, width=width, height=height, bg='white')
    cv.pack()
    cv.pack(expand=YES, fill=BOTH)
    cv.bind("<B1-Motion>", paint)

    predicted_label.config(text="The predicted number is:   ")


def paint(event):
    # python_green = "#476042"
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=40)
    draw.line([x1, y1, x2, y2],fill="black",width=40)



width = 350
height = 350
center = height//2
white = (255, 255, 255)



root = Tk()


button=Button(text="OK",command=save)
cleancanvas=Button(text="Clean",command=cleanfunc)
button.pack()
cleancanvas.pack()
predicted_label = Label(root, text="The predicted number is:   ")
predicted_label.pack()


cv = Canvas(root, width=width, height=height, bg='white')

# Gets the requested values of the height and widht.
# Gets both half the screen width/height and window width/height
positionRight = int(root.winfo_screenwidth()/2 - width/2)
positionDown = int(root.winfo_screenheight()/2 - height/2)

# Positions the window in the center of the page.
root.geometry("+{}+{}".format(positionRight, positionDown))


cv.pack()

image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)




root.mainloop()
