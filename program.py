import functionality
from tkinter import *
from PIL import Image
from PIL.ImageTk import PhotoImage

root = Tk()
root.title("Image Processing Program")
# win_width = root.winfo_screenwidth()
# win_height = root.winfo_screenheight()
# root.geometry(f"{win_width}x{win_height}")

root.state('zoomed')

#---load image frame--------------------------------------------
load_image_f = LabelFrame(root, text="Load image")

load_image_f.grid(row=0, column=0, padx=10, pady=20)
#load image button

load_image_b = Button(load_image_f, text="Open..", width=20, command=lambda: functionality.original_image_functionality(root, original_image_f, original_image_l, "load_image"))
load_image_b.pack(padx=5, pady=10)


#---Convert Frame-------------------------------------------------
convert_f = LabelFrame(root, text="Convert")
convert_f.grid(row=0, column=1, padx=10, pady=20)
# radio buttons
convert_rb = IntVar()
convert_rb.set(0)

convert_rb1 = Radiobutton(convert_f, text="Default color", variable=convert_rb, value=0, command=lambda: functionality.original_image_functionality(root, original_image_f, original_image_l, convert_rb.get()))
convert_rb1.pack(padx=5, pady=10)

convert_rb2 = Radiobutton(convert_f, text="Gray color", variable=convert_rb, value=1, command=lambda: functionality.original_image_functionality(root, original_image_f, original_image_l, convert_rb.get()))
convert_rb2.pack(padx=5, pady=10)
#---Add noise Frame-----------------------------------------------
add_noise_f = LabelFrame(root, text="Add noise")
add_noise_f.grid(row=0, column=2, padx=10, pady=20)
# radio buttons
noise_rb = IntVar()
noise_rb.set(0)

noise_rb1 = Radiobutton(add_noise_f, text="Salt & Pepper noise", variable=noise_rb, value=1, command=lambda: functionality.noise_functionality(root, noise_image_f, noise_image_l, noise_rb.get()))
noise_rb1.pack(padx=5, pady=10)
noise_rb2 = Radiobutton(add_noise_f, text="Gaussian noise", variable=noise_rb, value=2, command=lambda: functionality.noise_functionality(root, noise_image_f, noise_image_l, noise_rb.get()))
noise_rb2.pack(padx=5, pady=10)
noise_rb3 = Radiobutton(add_noise_f, text="Poisson noise", variable=noise_rb, value=3, command=lambda: functionality.noise_functionality(root, noise_image_f, noise_image_l, noise_rb.get()))
noise_rb3.pack(padx=5, pady=10)
#---Original image Frame------------------------------------------
from tkinter import ttk
original_image_f = ttk.LabelFrame(root, text="Original image")  
original_image_f.grid(row=0, column=3)
original_image_l = ttk.Label(original_image_f)
original_image_l.pack()

image_original = PhotoImage(file = "images/white.jpg")
original_image_l.config(image=image_original)
original_image_f.config(padding=(15, 15))

#---noise Adding Frame------------------------------------
noise_image_f = ttk.LabelFrame(root, text="after noise adding")  
noise_image_f.grid(row=0, column=4)
noise_image_l = ttk.Label(noise_image_f)
noise_image_l.pack()

noise_original = PhotoImage(file = "images/white.jpg")
noise_image_l.config(image=noise_original)
noise_image_f.config(padding=(15, 15))

#---Point Transform Op's Frame------------------------------------
point_f = ttk.LabelFrame(root, text="Point Transform Op's")
point_f.grid(row=1, column=0, columnspan=3)

point_f.config(padding=(15, 15))
#buttons

brightness_b = ttk.Button(point_f, text="Brightness adjustment", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "brightness"))
brightness_b.grid(row=0, column=0)
contrast_b = ttk.Button(point_f, text="Contrast adjustment", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "contrast"))
contrast_b.grid(row=1, column=1)
histogram_b = ttk.Button(point_f, text="Histogram", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "histogram"))
histogram_b.grid(row=2, column=2)
hist_equal_b = ttk.Button(point_f, text="Histogram Equalization", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "histogram_equalization"))
hist_equal_b.grid(row=3, column=3)


#---Result Frame------------------------------
result_f = ttk.LabelFrame(root, text="Result")  
result_f.grid(row=0, column=5)
result_l = ttk.Label(result_f)
result_l.pack()

result_original = PhotoImage(file = "images/white.jpg")
result_l.config(image=result_original)
result_f.config(padding=(15, 15))


#---Local Transform Op's Frame------------------------------------
local_f = ttk.LabelFrame(root, text="Local Transform Op's")
local_f.grid(row=2, column=0, columnspan=3)

local_f.config(padding=(15, 15))
#buttons

lp_b = ttk.Button(local_f, text="Low pass filter", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "low_pass"))
lp_b.grid(row=0, column=0)
hp_b = ttk.Button(local_f, text="High pass filter", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "high_pass"))
hp_b.grid(row=1, column=0)
mf_b = ttk.Button(local_f, text="Median filtering (gray image)", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "median"))
mf_b.grid(row=2, column=0)
af_b = ttk.Button(local_f, text="Averaging filtering", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "average"))
af_b.grid(row=3, column=0)
# --- edge detection frame and radio buttons
edge_f = ttk.LabelFrame(local_f, text="Edge detection filters")
edge_f.grid(row=0, column=1, columnspan=3)

local_f.config(padding=(15, 15))
#radio buttons
edge_rb = IntVar()
edge_rb.set(0)

Radiobutton(edge_f, text="Laplacian filter", variable=edge_rb, value=1, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=0, column=0)
Radiobutton(edge_f, text="Gaussian filter", variable=edge_rb, value=2, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=0, column=1)
Radiobutton(edge_f, text="Vert. Sobel", variable=edge_rb, value=3, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=0, column=2)
Radiobutton(edge_f, text="Horiz. Sobel", variable=edge_rb, value=4, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=0, column=3)
Radiobutton(edge_f, text="Vert. Prewitt", variable=edge_rb, value=5, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=1, column=0)
Radiobutton(edge_f, text="Horiz. Prewitt", variable=edge_rb, value=6, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=1, column=1)
Radiobutton(edge_f, text="Lap of Gau(log)", variable=edge_rb, value=7, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=1, column=2)
Radiobutton(edge_f, text="Canny method", variable=edge_rb, value=8, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=1, column=3)
Radiobutton(edge_f, text="Zero cross", variable=edge_rb, value=9, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=2, column=0)
Radiobutton(edge_f, text="Thicken", variable=edge_rb, value=10, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=2, column=1)
Radiobutton(edge_f, text="Skeleton", variable=edge_rb, value=11, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=2, column=2)
Radiobutton(edge_f, text="Thining", variable=edge_rb, value=12, command=lambda: functionality.result_frame_functionality(root, original_image_f, original_image_l, edge_rb.get())).grid(row=2, column=3)


#---Global Transform Op's Frame--------------------------------------
global_f = ttk.LabelFrame(root, text="Global Transform Op's")
global_f.grid(row=1, column=3, columnspan=2)

global_f.config(padding=(15, 25))
#buttons

ttk.Button(global_f, text="Line detection using Hough Transform", width=40, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "line_hough")).grid(row=0, column=0)
ttk.Button(global_f, text="Circles detection using Hough Transform", width=40, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "circle_hough")).grid(row=1, column=0)

#---Morphological Op's Frame--------------------------------------
morph_f = ttk.LabelFrame(root, text="Morphological Op's")
morph_f.grid(row=1, column=5)

morph_f.config(padding=(15, 25))
#buttons

ttk.Button(morph_f, text="Dilation", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "dilate")).grid(row=0, column=0)
ttk.Button(morph_f, text="Erosion", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "erode")).grid(row=1, column=0)
ttk.Button(morph_f, text="Close", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "close")).grid(row=2, column=0)
ttk.Button(morph_f, text="Open", width=20, command=lambda: functionality.result_frame_functionality(root, result_f, result_l, "open")).grid(row=3, column=0)

#---Final Button save and exit -------------------
Button(root, text="Save Result image", width=20, command=functionality.save_img).grid(row=2, column=3)
Button(root, text="Exit", width=20, command=lambda: functionality.exit(root)).grid(row=2, column=4)
root.mainloop()
