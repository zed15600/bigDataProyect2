# bigDataProyect2

# News Clusterer
Clusterizador de noticias en base a los t�picos que tratan utilizando el algoritmo LDA (Latent Driclet Allocation)

#### Notas:
Este programa est� hecho para funcionar espec�ficamente con el dataset "All the news" previsto por Kaggle (https://www.kaggle.com/snapcrack/all-the-news/data)
Encontrar� subsets de datos en la carpeta **data**

### Ejecuci�n:
Este programa debe ser ejecutado en el framework de computaci�n Spark en la versi�n 2.3 utilizando Python en su versi�n 3.6 mediante el siguiente comando, y estando ubicado en la carpeta **src**:
<rawtext>
spark-submit newsLDA.py \<path> [options]
</rawtext>
Donde \<path> represneta la direcci�n del archivo que se quiere procesar

Las opciones que acepta el programa son las siguientes:
| comando | explicacion |
| --------- | --- |
| -k=n | Especifica el n�mero de clusters en los que se clasificar�n las noticias. |
| -p=\<path> | Persiste datos de procesamiento en el directorio dado. Agiliza ejecuciones que usan las mismas bases. |
| -r | Reemplaza los datos almacenados para una nueva ejecuci�n. -p es requerido. |
| -a | Imprime todos los resultados de la ejecuci�n: Noticias con su relaci�n en cada cluster y los datos de los t�picos. |
| -c=n | Imprime las noticias relacionadas con el cluster dado. |
