from tkinter import *
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image
from PIL.ImageTk import PhotoImage
import cv2 as cv
import random
import numpy as np
import matplotlib.pyplot as plt

image_path = ""
#---original_image functionality----------------------------

def original_image_functionality(root, original_image_f, original_image_l, func):
    
    if func=="load_image":
        # make file dialog box to allow user to choose image and store the path of image
        global image_path
        image_path = filedialog.askopenfilename(initialdir='/', title="Select a photo", filetypes=(("png files", "*.png"), ("jpg files", "*.jpg")))
        # if the user click on load and didn't choose an image
        # and closed the dialog box so return nothing
        if image_path == "":
            return 
        # destroy the white background image to replace it with chosen image
        original_image_f.destroy()
        original_image_f = ttk.LabelFrame(root, text="Original image")  
        original_image_f.grid(row=0, column=3)
        # make a label and put the chosen image in it using image_path
        original_image_l = ttk.Label(original_image_f)
        original_image_l.pack()
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (150, 150))
        image_original = PhotoImage(Image.fromarray(img))
        original_image_l.config(image=image_original)
        original_image_f.config(padding=(15, 15))

        root.mainloop()
    if func == 0:
        # display original image
        if image_path == "":
            return 
        original_image_f.destroy()
        original_image_f = ttk.LabelFrame(root, text="Original image")  
        original_image_f.grid(row=0, column=3)
        # make a label and put the chosen image in it using image_path
        original_image_l = ttk.Label(original_image_f)
        original_image_l.pack()
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (150, 150))
        image_original = PhotoImage(Image.fromarray(img))
        original_image_l.config(image=image_original)
        original_image_f.config(padding=(15, 15))
        root.mainloop()
    if func == 1:
        if image_path == "":
            return 
        # convert image to gray
        original_image_f.destroy()
        original_image_f = ttk.LabelFrame(root, text="The Image with gray scale")  
        original_image_f.grid(row=0, column=3)
        # make a label and put the chosen image in it using image_path
        original_image_l = ttk.Label(original_image_f)
        original_image_l.pack()
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (150, 150))
        img_gray = PhotoImage(Image.fromarray(img))
        original_image_l.config(image=img_gray)
        original_image_f.config(padding=(15, 15))
        root.mainloop()

def noise_functionality(root, noise_image_f, noise_image_l, func):
    
    if func==1:
        #salt and pepper noise
        if image_path == "":
            return 
        # destroy the white background image to replace it with chosen image
        noise_image_f.destroy()
        noise_image_f = ttk.LabelFrame(root, text="salt and pepper noise")  
        noise_image_f.grid(row=0, column=4)
        # make a label and put the noise image in it using image_path
        noise_image_l = ttk.Label(noise_image_f)
        noise_image_l.pack()
        # make noise with open cv
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        output = img.copy()
        p = 0.05
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                rdn = random.random()
                if rdn < p/2:
                    output[i][j] = [0, 0, 0]
                elif rdn < p:
                    output[i][j] = [255, 255, 255]
                else:
                    output[i][j] = img[i][j]

        output = cv.resize(output, (150, 150))
        output = PhotoImage(Image.fromarray(output))
        noise_image_l.config(image=output)
        noise_image_f.config(padding=(15, 15))
        root.mainloop()
    elif func == 2:
        # gaussian noise
        if image_path == "":
            return 
        # destroy the white background image to replace it with chosen image
        noise_image_f.destroy()
        noise_image_f = ttk.LabelFrame(root, text="adding gaussian filter")  
        noise_image_f.grid(row=0, column=4)
        # make a label and put the noise image in it using image_path
        noise_image_l = ttk.Label(noise_image_f)
        noise_image_l.pack()
        # make noise with open cv
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        output = cv.GaussianBlur(img, (5, 5), 0)
        output = cv.resize(output, (150, 150))
        output = PhotoImage(Image.fromarray(output))
        noise_image_l.config(image=output)
        noise_image_f.config(padding=(15, 15))

        root.mainloop()
        
    elif func == 3:
        # poisson noise
        if image_path == "":
            return 
        # destroy the white background image to replace it with chosen image
        noise_image_f.destroy()
        noise_image_f = ttk.LabelFrame(root, text="adding poisson filter")  
        noise_image_f.grid(row=0, column=4)
        # make a label and put the noise image in it using image_path
        noise_image_l = ttk.Label(noise_image_f)
        noise_image_l.pack()
        # make noise with open cv
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        noise = np.random.poisson(img)
        output = img + noise
        # make it unit so it could be read by Image.fromarray
        output = output.astype(np.uint8)
        output = cv.resize(output, (150, 150))
        output = PhotoImage(Image.fromarray(output))
        noise_image_l.config(image=output)
        noise_image_f.config(padding=(15, 15))

        root.mainloop()
        
#---result Frame------------------------------------------
result_img=''
def result_frame_functionality(root, result_f, result_l, func):
    global result_img
    if func == "brightness":
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Applying Brightness adjustment")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        # add constant to the value
        # clip values over 255 to avoid overflow
        h, s, v = cv.split(img)
        lim = 255 - 50
        v[v > lim] = 255
        v[v <= lim] += 50
        img = cv.merge((h, s, v))

        result_img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))

        root.mainloop()
    
    elif func == "contrast":
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Applying Contrast adjustment")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        # multiply by constant to the value
        # clip values over 255 to avoid overflow
        h, s, v = cv.split(img)
        lim = 255 / 1.5
        v[v > lim] = 255
        v[v <= lim] = np.clip(v[v<=lim]*1.5, 0, 255).astype(np.uint8)
        img = cv.merge((h, s, v))

        result_img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))

        root.mainloop()
    elif func == "histogram":
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="The Histogram of image")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        
        plt.hist(img.ravel(),256,[0,256])
        
        plt.savefig("images/result_img.png")
        result_img = cv.imread("images/result_img.png")
        

        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        plt.close()
        root.mainloop()
    elif func == "histogram_equalization":
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Histogram Equalization of image")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        
        img = cv.equalizeHist(img)
        plt.hist(img.ravel(),256,[0,256])
        plt.savefig("images/result_img.png")
        result_img = cv.imread("images/result_img.png")
        
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        plt.close()
        root.mainloop()
    elif func == "low_pass":
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Low pass filter of image")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        kernel = np.ones((5,5),np.float32)/25
        result_img = cv.filter2D(img,-1,kernel)
        
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == "high_pass":
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="High pass filter of image")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        result_img = cv.filter2D(img,-1,kernel)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == "median":
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Median filter of image")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        result_img = cv.medianBlur(img, 5)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == "average":
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Average filter of image")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        result_img = cv.blur(img, (5,5))
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == 1:
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Laplacian filter")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        result_img = cv.filter2D(img,-1,kernel)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == 2:
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Gaussian filter")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img_blur = cv.GaussianBlur(img, (5, 5), 0)
        result_img = img - img_blur
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == 3:
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Vertical Sobel filter")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        result_img = cv.filter2D(img,-1,kernel)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == 4:
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Horizontal Sobel filter")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        result_img = cv.filter2D(img,-1,kernel)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == 5:
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Vertical Prewitt filter")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        result_img = cv.filter2D(img,-1,kernel)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == 6:
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Horizontal Prewitt filter")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

        kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        result_img = cv.filter2D(img,-1,kernel)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == 7:
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="LoG filter")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img_blur = cv.GaussianBlur(img, (5, 5), 0)
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        result_img = cv.filter2D(img_blur,-1,kernel)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == 8:
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Canny edge detection")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        result_img = cv.Canny(img, 100, 200)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == 9:
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Zero Cross")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        LoG = cv.Laplacian(img, cv.CV_16S)
        LoG = cv.resize(LoG, (150, 150))
        minLoG = cv.morphologyEx(LoG, cv.MORPH_ERODE, np.ones((3,3)))
        maxLoG = cv.morphologyEx(LoG, cv.MORPH_DILATE, np.ones((3,3)))
        zeroCross = np.logical_or(np.logical_and(minLoG < 0,  LoG > 0), np.logical_and(maxLoG > 0, LoG < 0))

        result_img = zeroCross

        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    # thining, thicken and skeleton no idea 
    # about how to implement
    elif func == "line_hough":
        if image_path == "":
            return 
        
        # openCV
        img = cv.imread(image_path)
        img = cv.resize(img, (150, 150))
        img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        edges = cv.Canny(img_gray,50,150,apertureSize = 3)
        lines = cv.HoughLines(edges,1,np.pi/180,50)

        if lines is None:
            messagebox.showinfo("Hough Line", "There is no Hough Lines in this Image")
            root.mainloop()
            return
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Extract line segments based on Hough transform")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        for line in lines:
            rho,theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv.line(img_gray,(x1,y1),(x2,y2),(0,0,255),1)

        result_img = img_gray
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == "circle_hough":
        if image_path == "":
            return 
        
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (150, 150))
        img = cv.medianBlur(img,5)
        cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
        circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20, 
            param1=50,param2=30,minRadius=0,maxRadius=0)
        if circles is None:
            messagebox.showinfo("Circles", "There is no Hough Circles in this Image")
            root.mainloop()
            return
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),1)
            # draw the center of the circle
            cv.circle(cimg,(i[0],i[1]),2,(0,0,255),1)

        
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Extract circule segments based on Hough transform")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()

        result_img = cimg
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == "dilate":
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Dilation")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        kernel = np.ones((5,5),np.uint8)
        result_img = cv.dilate(img, kernel, iterations=1)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == "erode":
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Erosion")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        kernel = np.ones((5,5),np.uint8)
        result_img = cv.erode(img, kernel, iterations=1)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()
    elif func == "open":
        if image_path == "":
            return 
        # destroy the white background image to replace it with result image
        result_f.destroy()
        result_f = ttk.LabelFrame(root, text="Open")  
        result_f.grid(row=0, column=5)
        # make a label and put the result image in it using image_path
        result_l = ttk.Label(result_f)
        result_l.pack()
        # openCV
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        kernel = np.ones((5,5),np.uint8)
        result_img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        result_img = cv.resize(result_img, (150, 150))
        result_pil = PhotoImage(Image.fromarray(result_img))
        result_l.config(image=result_pil)
        result_f.config(padding=(15, 15))
        root.mainloop()

def save_img():
    global result_img
    if result_img=='':
        return
    file = filedialog.asksaveasfilename(defaultextension=".png")
    
    if file:
        cv.imwrite(str(file), result_img)
def exit(root):
    
    root.quit()
