# Lista de nombres de archivos de texto a fusionar
text_files = ["1.txt", "2.txt", "3.txt", "4.txt"]

# Nombre del archivo de salida donde se guardarán los archivos fusionados
output_file = "merged_dataset.txt"

# Abre el archivo de salida en modo escritura
with open(output_file, "w", encoding="utf-8") as outfile:
    # Itera a través de la lista de archivos de texto
    for text_file in text_files:
        # Abre el archivo de texto actual en modo lectura
        with open(text_file, "r", encoding="utf-8") as infile:
            # Lee el contenido del archivo de texto y lo guarda en el archivo de salida
            content = infile.read()
            outfile.write(content)
            
            # Opcionalmente, agrega una nueva línea entre los archivos para facilitar la lectura
            outfile.write("\n")

print("Los archivos de texto se han fusionado correctamente en", output_file)
