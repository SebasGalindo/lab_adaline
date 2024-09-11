import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
from tkinter import filedialog
from PIL import Image

import shutil
import os, sys
import json

# Variables Declaration
main_window = None
title_lbl, description_lbl, logo_UdeC = None, None, None
canvas = None

def GUI_creation():

    global main_window, title, description, logo_UdeC

    # Build the main window
    main_window = ctk.CTk()
    main_window.title("Inteligencia Artificial - Adaline")
    main_window.geometry("1200x800")
    main_window.resizable(False, False)
    icon_path = resource_path("Resources/brand_logo.ico")
    main_window.iconbitmap(icon_path)
    main_window.attributes('-topmost', True)
    main_window.lift()
    main_window.grid_columnconfigure(1, weight=0) 
    main_window.grid_columnconfigure(2, weight=1) 
    main_window.grid_rowconfigure(0, weight=1)  

    # Sidebar creation
    sidebar = ctk.CTkFrame(master=main_window, width=200, fg_color="#11371A", corner_radius=0)
    sidebar.grid(row=0, column=0, sticky="nsew")

    # Vertical separator
    ctk.CTkFrame(master=main_window, width=2, fg_color="#0B2310").grid(row=0, column=1, sticky="ns")

    # UdeC logo creation
    logo_path = resource_path("Resources/logo_UdeC.png")
    logo_UdeC = Image.open(logo_path)
    logo_UdeC = ctk.CTkImage(dark_image=logo_UdeC, size=(60, 90))

    # title creation
    title_lbl = ctk.CTkLabel(master=sidebar, text="  Adaline", image=logo_UdeC, font=("Arial", 18), compound="left", cursor="hand2")
    title_lbl.bind("<Button-1>", initial_frame)
    title_lbl.grid(row=0, column=0, pady=10, sticky="n")

    # horizontal separator
    ctk.CTkFrame(master=sidebar, height=2, fg_color="#0B2310").grid(row=1, column=0, sticky="ew", pady=5)

    # Authors section 
    authors_lbl = ctk.CTkLabel(master=sidebar, text="Autores: \nJohn Sebastián Galindo Hernández \nMiguel Ángel Moreno Beltrán", font=("Arial", 16))
    authors_lbl.grid(row=2, column=0, pady=10, padx=5, sticky="n")

    # horizontal separator
    ctk.CTkFrame(master=sidebar, height=2, fg_color="#0B2310").grid(row=3, column=0, sticky="ew", pady=5)

    # Buttons section
    train_btn = ctk.CTkButton(master=sidebar, text="Realizar nuevo entrenamiento", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", command=train_frame)
    train_btn.grid(row=4, column=0, pady=10, sticky="n")

    train_info_btn = ctk.CTkButton(master=sidebar, text="Ver Información de entrenamiento", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", command=show_train_info)
    train_info_btn.grid(row=5, column=0, pady=10, sticky="n")

    test_solution_btn = ctk.CTkButton(master=sidebar, text="Probar soluciones", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", command=test_solutions_frame)
    test_solution_btn.grid(row=6, column=0, pady=10, sticky="n")

    initial_frame()
    # Run the main window
    main_window.mainloop()

def show_train_info():
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()

    principal_frame = ctk.CTkScrollableFrame(master=main_window, corner_radius=0, fg_color="#1D3F23",scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    principal_frame.grid(row=0, column=2, sticky="nsew")
    principal_frame.grid_rowconfigure(0, weight=1)
    principal_frame.grid_columnconfigure(0, weight=1)

    # Create the section title
    title = ctk.CTkLabel(master=principal_frame, text="Información de entrenamiento", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=20, sticky="nsew")

def train_frame():
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()

    train_frame = ctk.CTkScrollableFrame(master=main_window, corner_radius=0, fg_color="#1D3F23",scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    train_frame.grid(row=0, column=2, sticky="nsew")
    train_frame.grid_rowconfigure(0, weight=1)
    train_frame.grid_rowconfigure(1, weight=1)
    train_frame.grid_rowconfigure(2, weight=1)
    train_frame.grid_rowconfigure(3, weight=1)
    train_frame.grid_rowconfigure(4, weight=1)
    train_frame.grid_rowconfigure(5, weight=1)
    train_frame.grid_rowconfigure(6, weight=1)
    train_frame.grid_rowconfigure(7, weight=1)
    train_frame.grid_rowconfigure(8, weight=1)
    train_frame.grid_rowconfigure(9, weight=1)
    train_frame.grid_rowconfigure(10, weight=1)
    train_frame.grid_rowconfigure(11, weight=1)

    train_frame.grid_columnconfigure(0, weight=1)
    train_frame.grid_columnconfigure(1, weight=1)
    train_frame.grid_columnconfigure(2, weight=1)
    train_frame.grid_columnconfigure(3, weight=1)
    train_frame.grid_columnconfigure(4, weight=1)
    train_frame.grid_columnconfigure(5, weight=1)
    train_frame.grid_columnconfigure(6, weight=1)
    train_frame.grid_columnconfigure(7, weight=1)
    train_frame.grid_columnconfigure(8, weight=1)
    train_frame.grid_columnconfigure(9, weight=1)
    train_frame.grid_columnconfigure(10, weight=1)
    train_frame.grid_columnconfigure(11, weight=1)

    # Create the section title
    title = ctk.CTkLabel(master=train_frame, text="Sección de Nuevo Entrenamiento", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=20, sticky="nsew", columnspan=12)

    # Explanation section
    explanation_txt = ("Esta sección permitirá realizar un nuevo entrenamiento del modelo Adaline, "
                       "para ello, se deberá cargar un archivo de datos en formato JSON, "
                       "el cual deberá contener los datos de entrenamiento (Tanto las Entradas como las Salidas). "
                       "Se Aconseja descargar la plantilla de ejemplo para cargar los datos correctamente.\n "
                       "Una vez tenga los datos en formato JSON, deberá cargarlos con el botón 'Cargar Datos'."
                       )
    explanation_lbl = ctk.CTkLabel(master=train_frame, text=explanation_txt, font=("Arial", 16), justify="center", wraplength=900)   
    explanation_lbl.grid(row=1, column=0, pady=5, sticky="nsew", columnspan=12)

    json_png_path = resource_path("Resources/json_explain.png")
    json_png = Image.open(json_png_path)
    json_png_img = ctk.CTkImage(dark_image=json_png, size=(800, 226))

    advice_lbl = ctk.CTkLabel(master=train_frame,text="", image=json_png_img, justify="center", wraplength=900)
    advice_lbl.grid(row=2, column=0, pady=5, sticky="nsew", columnspan=12)

    # create the button to download the template json
    download_btn = ctk.CTkButton(master=train_frame, text="Descargar Plantilla", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", command=download_json)
    download_btn.grid(row=3, column=0, pady=10, sticky="n", columnspan=6)

    # create the button to load the json file
    load_btn = ctk.CTkButton(master=train_frame, text="Cargar Datos", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", command=load_json)
    load_btn.grid(row=3, column=6, pady=10, sticky="n", columnspan=6)

    # create labels with random text for test the scroll funtion
    for i in range(4, 12):
        ctk.CTkLabel(master=train_frame, text=f"Label {i}", font=("Arial", 16)).grid(row=i, column=0, pady=10, sticky="nsew", columnspan=12)
        

def test_solutions_frame():
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()

    test_frame = ctk.CTkScrollableFrame(master=main_window, corner_radius=0, fg_color="#1D3F23",scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    test_frame.grid(row=0, column=2, sticky="nsew")
    test_frame.grid_rowconfigure(0, weight=1)
    test_frame.grid_columnconfigure(0, weight=1)

    # Create the section title
    title = ctk.CTkLabel(master=test_frame, text="Sección de Pruebas", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=20, sticky="nsew")    

def initial_frame(event=None):
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()

    principal_frame = ctk.CTkScrollableFrame(master=main_window, corner_radius=0, fg_color="#1D3F23",scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    principal_frame.grid(row=0, column=2, sticky="nsew")
    principal_frame.grid_rowconfigure(0, weight=1)
    principal_frame.grid_columnconfigure(0, weight=1)

    # Create the section title
    title = ctk.CTkLabel(master=principal_frame, text="Sección de Explicación", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=20, sticky="nsew")

def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller
    params:
        relative_path: relative path to the resource
    return: absolute path to the resource
    """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# Function to download the template JSON file
def download_json():
    file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], initialfile="plantilla_Datos.json")
    json_template_path = resource_path("Data/Template.json")
    if file:
        shutil.copy(json_template_path, file)  # Copiar el archivo a la nueva ubicación
        print(f"JSON guardado en: {file}")

# Funtion to load the JSON file
def load_json():
    archivo = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if archivo:
        with open(archivo, 'r') as file:
            contenido = json.load(file)
            print(f"Contenido del JSON cargado:\n{contenido}")

if __name__ == "__main__":
    # Acceso a los archivos de pesos
    #ruta_pesos_json = resource_path('Data/pesos.json')
    #ruta_graficas_json = resource_path('Data/graficas.json')
    #pesos_json = {}
    #graficas_json = {}
    # Verificando si el archivo de pesos existe
    #if os.path.isfile(ruta_pesos_json) or os.path.isfile(ruta_graficas_json):
        # Si existe, se cargan los pesos
    #    with open(ruta_pesos_json, 'r') as file:
    #        pesos_json = json.load(file)
    #    with open(ruta_graficas_json, 'r') as file:
    #        graficas_json = json.load(file)
    #else:
        # Si no existe, se crean los pesos
        # pesos_json, graficas_json = creacion_pesos()
        # Se guardan los pesos
        # with open(ruta_pesos_json, 'w') as file:
        #    json.dump(pesos_json, file)
        # with open(ruta_graficas_json, 'w') as file:
        #    json.dump(graficas_json, file)

    # Creacion del GUI
    GUI_creation()
