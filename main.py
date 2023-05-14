import tkinter as tk
from tkinter import filedialog
import numpy as np
from tensorflow import keras
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

model = keras.models.load_model("model.h5")

def classify_image(image_path):
    image = keras.preprocessing.image.load_img(image_path, target_size=(300, 440))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0]

    classes = ['Adeno carcinoma', 'Large Cell Carcinoma', 'Normal', 'Squamous Cell Carcinoma']
    classes = [class_name.replace(" ","\n") for class_name in classes]
    class_names = np.array(classes)
    class_idx = np.argsort(prediction)[::-1]
    class_probs = prediction[class_idx]

    top_classes = class_names[class_idx]
    top_probs = class_probs

    result_label.config(text="Predicted Class: {}".format(top_classes[0].replace("\n"," ")))


    fig, ax = plt.subplots(num="CT Scan Probabilities")
    y_pos = np.arange(len(top_classes))
    ax.barh(y_pos, top_probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_classes)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title('Image Classification Results')
    for i, v in enumerate(top_probs):
        ax.text(v + 0.01, i, str(round(v, 2)), color='blue')

    plt.show()

def browse_button_clicked():
    plt.close("all")
    file_path = filedialog.askopenfilename(filetypes=[("Image File",'.jpg .png .jpeg')])
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, file_path)

    img = Image.open(file_path)
    img = img.resize((440, 300), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

    file_name_label.configure(text="Selected file: " + file_path.split("/")[-1])
    result_label.config(text="")
    classify_button['state'] = "normal"

def classify_button_clicked():
    image_path = file_path_entry.get()
    classify_image(image_path)

def clear_button_clicked():
    plt.close("all")
    classify_button['state'] = "disabled"
    file_name_label.config(text="")
    result_label.config(text="")
    file_path_entry.delete(0, tk.END)
    image_label.image = None

window = tk.Tk()
window.title("CT Scan Classification")
window.geometry("700x550")

top_frame = tk.Frame(window)
top_frame.pack(side=tk.TOP)

file_path_entry = tk.Entry(top_frame, width=50)

browse_button = tk.Button(top_frame, text="Browse", command=browse_button_clicked)

clear_button = tk.Button(top_frame,text="Clear",command=clear_button_clicked)

image_label = tk.Label(window)

classify_button = tk.Button(window, text="Classify", command=classify_button_clicked)
classify_button['state'] = 'disabled'
result_label = tk.Label(window, text="")

file_name_label = tk.Label(window)

clear_button.pack(side=tk.RIGHT, padx=5)
browse_button.pack(side=tk.RIGHT, padx=5)
file_path_entry.pack(side=tk.LEFT,pady=10)
image_label.pack(pady=20)
file_name_label.pack()
result_label.pack(pady=20)
classify_button.pack(pady=10)

window.mainloop()
