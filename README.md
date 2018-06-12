# bigDataProyect2

# News Clusterer
Clusterizador de noticias en base a los temas que tratan utilizando el algoritmo LDA (Latent Driclet Allocation)

#### Notas:
Este programa está hecho para funcionar especícamente con el dataset "All the news" previsto por Kaggle (https://www.kaggle.com/snapcrack/all-the-news/data)
Encontraré subsets de datos en la carpeta **data**

### Ejecución:
Este programa debe ser ejecutado en el framework de computación Spark en la versión 2.3 utilizando Python en su versión 3.6 mediante el siguiente comando, y estando ubicado en la carpeta **src**:

spark-submit newsLDA.py \<path> [options]

Donde \<path> represneta la dirección del archivo que se quiere procesar

Las opciones que acepta el programa son las siguientes:
| comando | explicacion |
| --------- | --- |
| -k=n | Especifica el número de clusters en los que se clasificarón las noticias. Default = 10. |
| -p=\<path> | Persiste datos de procesamiento en el directorio dado. Agiliza ejecuciones que usan las mismas bases. |
| -r | Reemplaza los datos almacenados para una nueva ejecución. -p es requerido. |
| -a | Imprime todos los resultados de la ejecución: Noticias con su relación en cada cluster y los datos de los temas. |
| -c=n | Imprime las noticias relacionadas con el cluster dado. |
