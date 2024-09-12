import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
from datetime import datetime
from PIL import Image

import shutil
import webbrowser
import os, sys
import json
from entrenamiento import train_adaline, adaline_aplication

# train and graph variables declaration
weights_path, graph_data_path, last_train_path, last_train_date = None, None, None, None
data_json, graph_data, train_json, inputs_json, date, test_results = None, None, None, None, None, None
theta = 0
weights_json, weights = None, None

# GUI Variables Declaration
main_window = None
title_lbl, description_lbl, logo_UdeC = None, None, None
canvas = None
status2_lbl, train_status2_lbl, last_training2_lbl, status_weights2_lbl, status_test_data2_lbl, test_status2_lbl  = None, None, None, None, None, None
results_lbl = None
precision_input, theta_input, alpha_input = None, None, None
download_weights_btn, load_test_btn, load_weights_btn = None, None, None
last_train_check = None

def GUI_creation():

    global main_window, title, description, logo_UdeC, last_training2_lbl, download_weights_btn

    # Build the main window
    main_window = ctk.CTk()
    main_window.title("Inteligencia Artificial - Adaline")
    main_window.geometry("1200x700")
    main_window.resizable(False, True)
    icon_path = resource_path("Resources/brand_logo.ico")
    main_window.iconbitmap(icon_path)
    main_window.lift()
    main_window.grid_columnconfigure(1, weight=0) 
    main_window.grid_columnconfigure(2, weight=1) 
    main_window.grid_rowconfigure(0, weight=1)  

    # Sidebar creation
    sidebar = ctk.CTkFrame(master=main_window, width=200, fg_color="#11371A", corner_radius=0)
    sidebar.grid(row=0, column=0, sticky="nsew")
    sidebar.grid_rowconfigure(8, minsize=160)

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

    # horizontal separator
    ctk.CTkFrame(master=sidebar, height=2, fg_color="#0B2310").grid(row=7, column=0, sticky="ew", pady=5)

    # Button for download the resultant weights
    download_weights_btn = ctk.CTkButton(master=sidebar,command= download_weitghs , text="Descargar Pesos", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", state="disabled")
    download_weights_btn.grid(row=9, column=0, pady=5)
    
    # Label for know the last date of the training
    last_training_lbl = ctk.CTkLabel(master=sidebar, text="Último entrenamiento: ", font=("Arial", 16, "bold"), text_color="#fbe122", anchor="w")
    last_training_lbl.grid(row=10, column=0, pady=5)

    # Label for the last training date
    last_training2_lbl = ctk.CTkLabel(master=sidebar, text="No hay entrenamientos ", font=("Arial", 16, "bold"), text_color="#fb2323", anchor="w")
    last_training2_lbl.grid(row=11, column=0, pady=2)

    if weights_json:
        download_weights_btn.configure(state="normal") 

    if date:
        last_training2_lbl.configure(text = date, text_color="#fff")
    elif weights_json:
        last_training2_lbl.configure(text = "Desconocido", text_color="#fff")

    initial_frame()
    # Run the main window
    main_window.mainloop()

def show_train_info():
    global graph_data, canvas   
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()

    graph_frame = ctk.CTkScrollableFrame(master=main_window, corner_radius=0, fg_color="#1D3F23",scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    graph_frame.grid(row=0, column=2, sticky="nsew")
    for i in range(12): 
        graph_frame.grid_rowconfigure(i, weight=1)
        graph_frame.grid_columnconfigure(i, weight=1)

    # Create the section title
    title = ctk.CTkLabel(master=graph_frame, text="Información de entrenamiento", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=20, sticky="nsew", columnspan=12)
    
    if graph_data is None:
        return

    # Graph of the Errors by epoch
    figure = plt.figure(figsize=(10, 10))
    ax = figure.add_subplot(211)
    ax.plot(graph_data["epochs"], graph_data["errors"], marker="o", color="red")
    ax.set_title("Errores por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("Error")

    # Graph of the Weights by epoch
    ax2 = figure.add_subplot(212)
    for i in range(len(graph_data["weights"][0])):
        ax2.plot(graph_data["epochs"], [w[i] for w in graph_data["weights"]], marker="o", label=f"Peso X_{i}")

    ax2.set_title("Pesos por época")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Peso")
    ax2.legend()

    figure.subplots_adjust(hspace=0.8)
    canvas = FigureCanvasTkAgg(figure, master=graph_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=1, column=0, pady=10,padx=20, sticky="nsew", columnspan=12, rowspan=10)

def train_frame():
    global status2_lbl,precision_input, theta_input, alpha_input, train_status2_lbl, download_weights_btn, weitghs_json, data_json, canvas, date
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()

    train_frame = ctk.CTkScrollableFrame(master=main_window, corner_radius=0, fg_color="#1D3F23",scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    train_frame.grid(row=0, column=2, sticky="nsew")
    for i in range(12):  
        train_frame.grid_rowconfigure(i, weight=1)
        train_frame.grid_columnconfigure(i, weight=1)

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

    # Training data section

    # create the button to download the template json
    download_btn = ctk.CTkButton(master=train_frame, text="Descargar Plantilla", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", command=download_json)
    download_btn.grid(row=3, column=0, pady=10, sticky="n", columnspan=6)

    # create the button to load the json file
    load_btn = ctk.CTkButton(master=train_frame, text="Cargar Datos", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", command=load_train_data)
    load_btn.grid(row=3, column=6, pady=10, sticky="n", columnspan=6)

    # Input label for the alpha value
    alpha_lbl = ctk.CTkLabel(master=train_frame, text="Valor de α:", font=("Arial", 16), text_color="#fbe122")
    alpha_lbl.grid(row=4, column=0, pady=10, sticky="nsew", columnspan=2, padx=10)
    alpha_input = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=10)
    alpha_input.grid(row=4, column=2, pady=10, sticky="nsew", columnspan=1, padx=2)

    # Input label for theta value
    theta_lbl = ctk.CTkLabel(master=train_frame, text="Valor del umbral (θ):", font=("Arial", 16), text_color="#fbe122")
    theta_lbl.grid(row=4, column=3, pady=10, sticky="nsew", columnspan=3, padx=10)
    theta_input = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=10)
    theta_input.grid(row=4, column=6, pady=10, sticky="nsew", columnspan=1, padx=2)

    # Input label for the precision value
    precision_lbl = ctk.CTkLabel(master=train_frame, text="Valor de la precisión:", font=("Arial", 16), text_color="#fbe122")
    precision_lbl.grid(row=4, column = 7, pady=10, sticky="nsew", columnspan=3, padx=10)
    precision_input = ctk.CTkEntry(master=train_frame, font=("Arial", 16), width=10)
    precision_input.grid(row=4, column=10, pady=10, sticky="nsew", columnspan=1, padx=2)

    # label for the status of the load data (JSON)
    status_lbl = ctk.CTkLabel(master=train_frame, text="Estado de los datos: ", font=("Arial", 16, "bold"), text_color="#fbe122", anchor="w")
    status_lbl.grid(row=5, column=1, pady=10, sticky="nsew", columnspan=3, padx=10)
    status2_lbl = ctk.CTkLabel(master=train_frame, text="No Cargados ", font=("Arial", 16, "bold"), text_color="#fb2323", anchor="w")
    status2_lbl.grid(row=5, column=4, pady=10, sticky="nsew", columnspan=8, padx=2)

    # button for start the training
    train_btn = ctk.CTkButton(master=train_frame, text="Iniciar Entrenamiento", command=start_training, fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010")
    train_btn.grid(row=6, column=0, pady=10, sticky="n", columnspan=4)

    # label for training status
    train_status_lbl = ctk.CTkLabel(master=train_frame, text="Estado del entrenamiento: ", font=("Arial", 16, "bold"), text_color="#fbe122", anchor="w")
    train_status_lbl.grid(row=6, column=4, pady=10, sticky="nsew", columnspan=3, padx=10)

    # label for the training status
    train_status2_lbl = ctk.CTkLabel(master=train_frame, text="No Iniciado ", font=("Arial", 16, "bold"), text_color="#fb2323", anchor="w")
    train_status2_lbl.grid(row=6, column=7, pady=10, sticky="nsew", columnspan=5, padx=2)

def test_solutions_frame():
    global last_train_check, load_test_btn, load_weights_btn, status_test_data2_lbl, status_weights2_lbl, test_status2_lbl, results_lbl

    test_frame = ctk.CTkScrollableFrame(master=main_window, corner_radius=0, fg_color="#1D3F23",scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    test_frame.grid(row=0, column=2, sticky="nsew")
    for i in range(12):  
        test_frame.grid_rowconfigure(i, weight=1)
        test_frame.grid_columnconfigure(i, weight=1) 

    # Create the section title
    title = ctk.CTkLabel(master=test_frame, text="Sección de Pruebas", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=20, sticky="nsew", columnspan=12)    

    # Explanation section
    explanation_txt = ("Esta sección permitirá realizar pruebas con el modelo Adaline, "
                        "Se tienen 2 opciones para realizar las pruebas."
                        "\n1. Marcar la casilla de verificación 'Usar último entrenamiento' para cargar los datos del último entrenamiento realizado."
                        "esto permite no cargar los datos manualmente."
                        "\n2. Cargar los datos manualmente, para ello, se deberá cargar dos archivos en formato JSON, "
                        "uno con las entradas y otro con los pesos y el bias entregados por el entrenamiento."
                        )
    explanation_lbl = ctk.CTkLabel(master=test_frame, text=explanation_txt, font=("Arial", 16), justify="center", wraplength=900)
    explanation_lbl.grid(row=1, column=0, pady=5, sticky="nsew", columnspan=12)

    # Image for the explanation of the json_input_data (json_test.png)
    json_png_path = resource_path("Resources/json_test.png")
    json_png = Image.open(json_png_path)
    json_png_img = ctk.CTkImage(dark_image=json_png, size=(750, 176))

    # Label for the image
    inputs_lbl = ctk.CTkLabel(master=test_frame,text="", image=json_png_img, justify="center", wraplength=900)
    inputs_lbl.grid(row=2, column=0, pady=5, sticky="nsew", columnspan=12)

    # Image for the explanation of the weights json (json_weights.png)
    json_png_path = resource_path("Resources/json_weights.png")
    json_png = Image.open(json_png_path)
    json_png_img = ctk.CTkImage(dark_image=json_png, size=(750, 241))

    # Label for the image
    weights_lbl = ctk.CTkLabel(master=test_frame,text="", image=json_png_img, justify="center", wraplength=900)
    weights_lbl.grid(row=3, column=0, pady=5, sticky="nsew", columnspan=12)
    
    # Checkbutton for use the last training
    last_train_check = ctk.CTkCheckBox(master=test_frame,command= change_state_btn , text="Usar último entrenamiento", font=("Arial", 16), text_color="#fbe122", border_color="#fbe122", fg_color="#d66913", hover_color="#d66913")
    last_train_check.grid(row=4, column=1, pady=10, padx = 30, sticky="nsew", columnspan=11)

    # Button for load the test data
    load_test_btn = ctk.CTkButton(master=test_frame, text="Cargar Entradas", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", command=load_test_data, )
    load_test_btn.grid(row=5, column=0, pady=10, sticky="n", columnspan=6)

    # Button for load the weights
    load_weights_btn = ctk.CTkButton(master=test_frame, text="Cargar Pesos", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", command=load_weights)
    load_weights_btn.grid(row=5, column=6, pady=10, sticky="n", columnspan=6)

    # label for the status of the test data (JSON)
    status_test_data_lbl = ctk.CTkLabel(master=test_frame, text="Estado de los datos: ", font=("Arial", 16, "bold"), text_color="#fbe122", anchor="w")
    status_test_data_lbl.grid(row=6, column=1, pady=10, sticky="nsew", columnspan=3, padx=10)
    status_test_data2_lbl = ctk.CTkLabel(master=test_frame, text="No Cargados ", font=("Arial", 16, "bold"), text_color="#fb2323", anchor="w")
    status_test_data2_lbl.grid(row=6, column=4, pady=10, sticky="nsew", columnspan=8, padx=2)

    # label for the status of the weights (JSON)
    status_weights_lbl = ctk.CTkLabel(master=test_frame, text="Estado de los pesos: ", font=("Arial", 16, "bold"), text_color="#fbe122", anchor="w")
    status_weights_lbl.grid(row=7, column=1, pady=10, sticky="nsew", columnspan=3, padx=10)
    status_weights2_lbl = ctk.CTkLabel(master=test_frame, text="No Cargados ", font=("Arial", 16, "bold"), text_color="#fb2323", anchor="w")
    status_weights2_lbl.grid(row=7, column=4, pady=10, sticky="nsew", columnspan=8, padx=2)

    # Button for start the test
    test_btn = ctk.CTkButton(master=test_frame, text="Probar Soluciones", fg_color="#fbe122", width=180, height=40, font=("Arial", 13, "bold"), hover_color="#E2B12F", text_color="#0F1010", command=start_test)
    test_btn.grid(row=8, column=0, pady=10, sticky="n", columnspan=12)

    # Label for the status of the test
    test_status_lbl = ctk.CTkLabel(master=test_frame, text="Estado de la prueba: ", font=("Arial", 16, "bold"), text_color="#fbe122", anchor="w")
    test_status_lbl.grid(row=9, column=0, pady=10, sticky="nsew", columnspan=3, padx=10)    

    # Label for the status of the test
    test_status2_lbl = ctk.CTkLabel(master=test_frame, text="No Iniciada ", font=("Arial", 16, "bold"), text_color="#fb2323", anchor="w")
    test_status2_lbl.grid(row=9, column=4, pady=10, sticky="nsew", columnspan=8, padx=2)

    # Horizontal separator
    ctk.CTkFrame(master=test_frame, height=2, fg_color="#0B2310").grid(row=10, column=0, sticky="ew", pady=5, columnspan=12)

    # List of the test results in the way of [Entradas: Entries| Resultado: Outputs]
    test_results_lbl = ctk.CTkLabel(master=test_frame, text="Resultados de las pruebas", font=("Arial", 16, "bold"), text_color="#fbe122", anchor="center")
    test_results_lbl.grid(row=11, column=0, pady=10, sticky="nsew", columnspan=12)

    results_lbl = ctk.CTkLabel(master=test_frame, text="", font=("Arial", 16), justify="center", wraplength=900)
    results_lbl.grid(row=12, column=0, pady=5, sticky="nsew", columnspan=12)



def initial_frame(event=None):
    if canvas is not None:
        canvas.get_tk_widget().grid_forget()

    principal_frame = ctk.CTkScrollableFrame(master=main_window, corner_radius=0, fg_color="#1D3F23",scrollbar_button_color="#112f16", scrollbar_button_hover_color="#446249")
    principal_frame.grid(row=0, column=2, sticky="nsew")
    for i in range(12):  # Configura filas 0 a 4
        principal_frame.grid_rowconfigure(i, weight=1)
    principal_frame.grid_columnconfigure(0, weight=1)
    principal_frame.grid_columnconfigure(1, weight=1)

    # Create the section title
    title = ctk.CTkLabel(master=principal_frame, text="Sección de Explicación", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title.grid(row=0, column=0, pady=20, sticky="nsew", columnspan=2)

    # App Explanation section
    explanation_txt = ("Esta aplicación permite realizar un entrenamiento de un modelo Adaline, "
                        "Por defecto esta cargado el ultimo entrenamiento realizado."
                        "El programa cuenta con 3 secciones principales: Nuevo Entrenamiento, Información de Entrenamiento y Pruebas.")
    explanation_lbl = ctk.CTkLabel(master=principal_frame, text=explanation_txt, font=("Arial", 16), justify="left", wraplength=900)
    explanation_lbl.grid(row=1, column=0, pady=5, sticky="nsew", columnspan=2)

    # Training adaline explanation section
    # Create the section title
    title2 = ctk.CTkLabel(master=principal_frame, text="Sección de entrenamiento", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title2.grid(row=2, column=0, pady=20, sticky="nsew", columnspan=2)
    # explanation text
    explanation2_txt = ("La sección de entrenamiento permite realizar un nuevo entrenamiento del modelo Adaline, "
                        "para ello, se deberá cargar un archivo de datos en formato JSON, "
                        "el cual deberá contener los datos de entrenamiento (Tanto las Entradas como las Salidas). "
                        "Se Aconseja descargar la plantilla de ejemplo para cargar los datos correctamente.\n "
                        "Una vez tenga los datos en formato JSON, deberá cargarlos con el botón 'Cargar Datos'."
                        "Una vez cargados los datos, deberá ingresar el valor de α, θ y la precisión deseada para el entrenamiento."
                        "el valor de α y la precisión deben ser mayores a 0."
                        "Finalmente, deberá presionar el botón 'Iniciar Entrenamiento' para comenzar el proceso."
                        )
    explanation2_lbl = ctk.CTkLabel(master=principal_frame, text=explanation2_txt, font=("Arial", 16), justify="left", wraplength=900)
    explanation2_lbl.grid(row=3, column=0, pady=5, sticky="nsew", padx=5, columnspan=2)

    # Graph explanation section
    # Create the section title
    title3 = ctk.CTkLabel(master=principal_frame, text="Sección de Información de Entrenamiento", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title3.grid(row=4, column=0, pady=20, sticky="nsew", columnspan=2)
    # explanation text
    explanation3_txt = ("La sección de Información de Entrenamiento permite visualizar la información del último entrenamiento realizado, "
                        "en esta sección se mostrará un gráfico con los errores por época y los pesos por época."
                        )
    explanation3_lbl = ctk.CTkLabel(master=principal_frame, text=explanation3_txt, font=("Arial", 16), justify="left", wraplength=900)
    explanation3_lbl.grid(row=5, column=0, pady=5, sticky="nsew", columnspan=2)

    # Test explanation section
    # Create the section title
    title4 = ctk.CTkLabel(master=principal_frame, text="Sección de Pruebas", font=("Arial", 20, "bold"), text_color="#fbe122", anchor="center", justify="center")
    title4.grid(row=6, column=0, pady=20, sticky="nsew", columnspan=2)
    # explanation text
    explanation4_txt = ("La sección de Pruebas permite realizar pruebas con el modelo entrenado, "
                        "para ello, se deberá cargar dos archivos en formato JSON, "
                        "uno con las entradas y otro con los pesos y el bias usado en el entrenamiento."
                        "La opcion por defecto es usar el boton que permite cargar el último entrenamiento realizado y que está alojaddo en la app."
                        "Una vez cargados los datos, se deberá presionar el botón 'Probar Soluciones' para comenzar el proceso."
                        "Finalmente, se mostrara una la lista de entradas y las salidas obtenidas."
                        )
    explanation4_lbl = ctk.CTkLabel(master=principal_frame, text=explanation4_txt, font=("Arial", 16), justify="left", wraplength=900)
    explanation4_lbl.grid(row=7, column=0, pady=5, sticky="nsew", padx=5, columnspan=2)
    
    # Github logo 
    github_path = resource_path("Resources/github_PNG.png")
    logo_github_img = Image.open(github_path)
    logo_github = ctk.CTkImage(dark_image=logo_github_img, size=(216, 80))

    # link to the Github Project
    github_link = ctk.CTkLabel(master=principal_frame, text="Codigo del proyecto", font=("Arial", 16, "bold"), text_color="#fbe122", cursor="hand2", image=logo_github, compound="right")
    github_link.bind("<Button-1>", open_github)
    github_link.grid(row=8, column=0, pady=20, sticky="nsew")

    # Documentation logo
    doc_path = resource_path("Resources/doc_logo.png")
    logo_doc_img = Image.open(doc_path)
    logo_doc = ctk.CTkImage(dark_image=logo_doc_img, size=(80, 80))

    # link to the Documentation
    doc_link = ctk.CTkLabel(master=principal_frame, text="IEEE del proyecto", font=("Arial", 16, "bold"), text_color="#fbe122", cursor="hand2", image=logo_doc, compound="right")
    doc_link.bind("<Button-1>", open_documentation)
    doc_link.grid(row=8, column=1, pady=20, sticky="nsew")

def open_github(event):
    webbrowser.open("https://github.com/SebasGalindo/lab_adaline")    

def open_documentation(event):
    webbrowser.open("https://youtube.com")

def change_state_btn():
    global last_train_check, load_test_btn, load_weights_btn, status_test_data2_lbl, status_weights2_lbl, inputs_json, weights_json
    state = last_train_check.get()
    if state == 0:
        load_test_btn.configure(state="normal")
        load_weights_btn.configure(state="normal")
        status_test_data2_lbl.configure(text="No cargado", text_color="#d62c2c")
        status_weights2_lbl.configure(text="No cargado", text_color="#d62c2c")
    else:
        load_test_btn.configure(state="disabled")
        load_weights_btn.configure(state="disabled")
        if inputs_json:
            status_test_data2_lbl.configure(text="Cargadas Las Entradas Por Defecto", text_color="#b58e12")
        if weights_json:
            status_weights2_lbl.configure(text="Cargadas Los Pesos Por Defecto", text_color="#b58e12")

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
    data_json = {}
    file = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if file:
        file_name = os.path.basename(file)
        with open(file, 'r') as file:
            content = json.load(file)
            data_json = content
            print(f"Contenido del JSON cargado:\n{content}")
        return data_json, file_name
    else:
        return None, None

def load_train_data():
    global status2_lbl, data_json
    data, file_name = load_json()
    if data:
            status2_lbl.configure(text=f"Cargado correctamente {file_name}", text_color="#45b51f")
            data_json = data
    elif data_json and not data:
        status2_lbl.configure(text="Anterior Sigue Cargado", text_color="#b58e12")
    else:
        status2_lbl.configure(text="No cargado", text_color="#d62c2c")

def load_test_data():
    global inputs_json, status_test_data2_lbl
    data, file_name = load_json()
    if data:
        status_test_data2_lbl.configure(text=f"Cargado correctamente {file_name}", text_color="#45b51f")
        inputs_json = data
    elif inputs_json and not data:
        status_test_data2_lbl.configure(text="Entradas Anteriores Siguen Cargadas", text_color="#b58e12")
    else:
        status_test_data2_lbl.configure(text="No cargado", text_color="#d62c2c")

def load_weights():
    global weights_json, status_weights2_lbl
    data, file_name = load_json()
    if data:
        status_weights2_lbl.configure(text=f"Cargado correctamente {file_name}", text_color="#45b51f")
        weights_json = data
    elif weights_json and not data:
        status_weights2_lbl.configure(text="Pesos Anteriores Siguen Cargados", text_color="#b58e12")
    else:
        status_weights2_lbl.configure(text="No cargado", text_color="#d62c2c")

def download_weitghs():
    global weights_json
    file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")], initialfile="pesos.json")
    if file_path:
        with open(file_path, 'w') as file:
            json.dump(weights_json, file, indent=4)  # Guardar el diccionario en formato JSON
        print(f"Diccionario guardado en: {file_path}")
 
def start_training():   
    global precision_input, theta_input, alpha_input, download_weights_btn, data_json, theta, weights_json, last_training2_lbl, graph_data

    if data_json is None:
        train_status2_lbl.configure(text="Datos no cargados", text_color="#d62c2c")
        return
    
    try:
        if not alpha_input.get():
            train_status2_lbl.configure(text="Falta el valor de α", text_color="#d62c2c")
            return
        if not theta_input.get():
            train_status2_lbl.configure(text="Falta el valor de θ", text_color="#d62c2c")
            return
        if not precision_input.get():
            train_status2_lbl.configure(text="Falta el valor de la precisión", text_color="#d62c2c")
            return
        if not data_json:
            train_status2_lbl.configure(text="Datos JSON no cargados", text_color="#d62c2c")
            return
        
        alpha = float(alpha_input.get())
        theta = float(theta_input.get())
        precision = float(precision_input.get())

        if alpha <= 0:
            train_status2_lbl.configure(text="El valor de α debe ser mayor a 0", text_color="#d62c2c")
            return
        
        if precision <= 0:
            train_status2_lbl.configure(text="El valor de la precisión debe ser mayor a 0", text_color="#d62c2c")
            return

        train_status2_lbl.configure(text="Entrenamiento Empezado ", text_color="#45b51f")
        weights, graph_data, theta = train_adaline(data_json, alpha, theta, precision)

        weights_json = {
            "weights": weights,
            "theta": theta
        }

        train_status2_lbl.configure(text="Entrenamiento Finalizado ", text_color="#45b51f")
        download_weights_btn.configure(state="normal")
        # get the actual date in format dd/mm/yyyy hh:mm:ss
        date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        last_training2_lbl.configure(text = date, text_color="#fff")
        store_data(weights_json, data_json, graph_data, date)

    except Exception as e:
        print("error", e)
        return

def start_test():
    global weights_json, inputs_json, test_status2_lbl, test_results, results_lbl, status_test_data2_lbl, status_weights2_lbl, last_train_check

    results_lbl.configure(text="")

    status_test_data_txt = status_test_data2_lbl.cget("text")
    status_weights_txt = status_weights2_lbl.cget("text")

    if status_test_data_txt == "No cargado" or status_weights_txt == "No cargado":
        test_status2_lbl.configure(text="Datos no cargados", text_color="#d62c2c")
        return

    if not weights_json:
        test_status2_lbl.configure(text="Pesos no cargados", text_color="#d62c2c")
        return
    if not inputs_json:
        test_status2_lbl.configure(text="Entradas no cargadas", text_color="#d62c2c")
        return

    if not "theta" in weights_json:
        test_status2_lbl.configure(text="valor Theta no encontrado", text_color="#d62c2c")
        return
    
    if not "weights" in weights_json:
        test_status2_lbl.configure(text="pesos no encontrados", text_color="#d62c2c")
        return
    
    theta = weights_json["theta"]
    weights = weights_json["weights"]
    inputs = inputs_json["entradas"]

    if len(weights) != (len(inputs[0]) + 1):
        test_status2_lbl.configure(text="No es igual # entradas y # pesos ", text_color="#d62c2c")
        return

    test_status2_lbl.configure(text="Prueba Iniciada", text_color="#45b51f")

    test_results = adaline_aplication(inputs_json, weights, theta)

    text_results = ""

    for i in range(len(inputs)):
        text_results += f" {inputs[i]}     Resultado: {round(test_results[i],4)}\n"
    
    results_lbl.configure(text=text_results)

def first_train():
    global weights_json, train_json, date, graph_data
    template_path = resource_path('Data/Template.json')

    if not train_json:
        if os.path.isfile(template_path):
            with open(template_path, 'r') as file:
                train_json = json.load(file)
        else:
            return
        
        weights, graph_data, theta = train_adaline(train_json)

        weights_json = {
            "weights": weights,
            "theta": theta
        }
        date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        store_data(weights_json, train_json, graph_data, date)

def store_data(weights_json, train_json, graph_data, date):
    global weights_path, graph_data_path, last_train_path, last_train_date

    # Store the weights in a JSON file
    with open(weights_path, 'w') as file:
        json.dump(weights_json, file, indent=4)  

    # Store the graph data in a JSON file
    with open(graph_data_path, 'w') as file:
        json.dump(graph_data, file, indent=4)

    # Store the last train data in a JSON file
    with open(last_train_path, 'w') as file:
        json.dump(train_json, file, indent=4)  

    # Store the last train date in a JSON file
    with open(last_train_date, 'w') as file:
        json.dump({"date": date}, file, indent=4)

if __name__ == "__main__":
    # Paths of the JSON files
    weights_path = resource_path('Data/weights.json')
    last_train_path = resource_path('Data/last_train_data.json')
    graph_data_path = resource_path('Data/graph_data.json')
    last_train_date = resource_path('Data/last_train_date.json')

    # Files verification and load
    if os.path.isfile(last_train_path) and os.path.isfile(weights_path) and os.path.isfile(graph_data_path):
        with open(last_train_path, 'r') as file:
            train_json = json.load(file)
            inputs_json = train_json
        with open(weights_path, 'r') as file:
            weights_json = json.load(file)
        with open(graph_data_path, 'r') as file:
            graph_data = json.load(file)
    else:
        first_train()

    if os.path.isfile(last_train_date):
        with open(last_train_date, 'r') as file:
            date = json.load(file)["date"]
    
    print("weights_json", weights_json)
    # GUI creation
    GUI_creation()
