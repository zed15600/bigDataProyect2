from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.clustering import LDA, LDAModel, LocalLDAModel

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import re, sys, os, shutil

class LdaClusterer:

    #Global vars
    #Init spark context and session
    sc = None
    spark = None

    #Control vars
    pesistSteps = False
    recompute = False
    printOnly = None
    printAll = None

    kTopics = 10
    titles = []
    vocabulary = None
    topics = None
    topicIndices = None
    predictions = None
    stepsPath = "/home/ec2-user/ldaNewsCluster/"

    def __init__(self):
        self.sc = SparkContext.getOrCreate()
        self.spark = SparkSession.builder.appName("News Topic Clustering - Python").getOrCreate()


    #Remove undesired columns, remove stopwords and punctuation
    def removeCols(self, inputData):
        print("Extracting and cleaning data")
        filteredSet = []
        for i,line in enumerate(inputData):
            #Split line into columns
            splitted = re.split(r'(".*?")*,', (line+","))
            #Verify if row doesn't fit adequate syntax
            if(len(splitted[4])<5 or len(splitted[19])<100):
                continue
            self.titles.append(splitted[4])
            #New content to lowercase
            splitted[19] = splitted[19].lower()
            #Tokenizer for remove punctuation
            tokenizer = RegexpTokenizer(r'\w+')
            #Remove punctuation from new content
            tokens = tokenizer.tokenize(splitted[19])
            #Remove stopwords from new content
            filtered = [w for w in tokens if not w in stopwords.words('english')]
            #Save tittle and content columns
            filteredSet.append([i, filtered])
        return filteredSet

    #Transform new data into dataframe
    def transformToDF(self, filteredData):
        print("Transforming to DF")
        rdd = self.sc.parallelize(filteredData)
        df = self.spark.createDataFrame(rdd, ["label", "words"])
        return df

    def transformDataToFeaturesVector(self, dataDF):
        print("Term frecuency and vocabulary extraction")
        #Count Term Frecuency, transform data into features vector
        vector = CountVectorizer(inputCol="words", outputCol="vector")
        model = vector.fit(dataDF)
        self.vocabulary = model.vocabulary
        result = model.transform(dataDF)
        return result

    def ponderData(self, featuresVector):
        print("Features pondering")
        #Pondering data due it's frecuency
        idf = IDF(inputCol="vector", outputCol="features")
        idfModel = idf.fit(featuresVector)
        rescaledData = idfModel.transform(featuresVector)
        return rescaledData

    def getCorpus(self, data):
        corpus = None
        filteredSet = self.removeCols(data)
        dataDF = self.transformToDF(filteredSet)
        featuresVector = self.transformDataToFeaturesVector(dataDF)
        if(self.persistSteps and not self.recompute and os.path.isdir(self.stepsPath+"corpus")):
            print("Corpus exist, loading")
            corpus = self.spark.read.load(self.stepsPath+"corpus")
        else:
            ponderedData = self.ponderData(featuresVector)
            #Fit dataframe to ML
            print("Computing Corpus")
            corpus = ponderedData.select("label", "features").rdd.map(lambda x: [x[0], Vectors.dense(x[1])]).cache()
            corpus = self.spark.createDataFrame(corpus, ["label", "features"])
            if(self.persistSteps):
                print("Saving Corpus")
                if(os.path.isdir(self.stepsPath+"corpus")):
                    shutil.rmtree(self.stepsPath+"corpus")
                corpus.select("label", "features").write.save(self.stepsPath+"corpus")

        return corpus

    def modelData(self, corp):
        print("Data modeling")
        #Cluster the data into n topics using LDA
        ldaModel = None

        if (self.persistSteps and not self.recompute and os.path.isdir(self.stepsPath+"ldaModel")):
            print("Model exist, loading")
            ldaModel = LocalLDAModel.load(self.stepsPath+"ldaModel")
        else:
            print("Creating Model")
            lda = LDA(k=self.kTopics, maxIter=100, optimizer='online')
            ldaModel = lda.fit(corp)
            if(self.persistSteps):
                print("Saving model")
                if(os.path.isdir(self.stepsPath+"ldaModel")):
                    shutil.rmtree(self.stepsPath+"ldaModel")
                ldaModel.save(self.stepsPath+"ldaModel")

        print("Extracting Topics")
        self.topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 5)

        if(self.persistSteps and not self.recompute and os.path.isdir(self.stepsPath+"predictions")):
            print("Predictions exist, loading")
            self.predictions = self.spark.read.load(self.stepsPath+"predictions")
        else:
            print("Predicting Data")
            self.predictions = ldaModel.transform(corp)
            self.topics = ldaModel.topicsMatrix()
            if(self.persistSteps):
                print("Saving predictions data")
                if(os.path.isdir(self.stepsPath+"predictions")):
                    shutil.rmtree(self.stepsPath+"predictions")
                self.predictions.select("label", "features", "topicDistribution").write.save(self.stepsPath+"predictions")


    def main(self, argv):

        if(len(argv) < 2 or "-h" in argv or not os.path.isfile(argv[1])):
            self.printHelp()
            return
        for i,arg in enumerate(argv):
            if(i in [0, 1]):
                continue
            if(arg[:3] == "-p="):
                if(arg[len(arg)-1:] != "/"):
                    self.stepsPath = arg+"/"
                else:
                    self.stepsPath = arg[3:]
                if(not os.path.isdir(self.stepsPath)):
                    print("\tPersis path must be a directory")
                    return
                self.persistSteps = True
            elif(arg == "-r"):
                self.recompute = True
            elif(arg[:3] == "-c="):
                try:
                    self.printOnly = int(arg[3:])
                except ValueError:
                    print("Cluster value must be an integer")
                    self.printHelp()
                    return
            elif(arg == "-a"):
                self.printAll = True
            elif(arg[:0] == "-k="):
                try:
                    self.kTopics = int(arg[3:])
                except ValueError:
                    print("K value must be an integer")
                    self.printHelp()
                    return
            else:
                print("\tUnrecognized Argument: " + arg)
                self.printHelp()
                return
        #Open news file
        dataset = self.sc.textFile(argv[1]).collect()

        corpus = self.getCorpus(dataset)
        self.modelData(corpus)
        if(self.printAll):
            self.printPredictions()
            self.printTopics(5)
        if(self.printOnly != None):
            self.printSingle()

    def printHelp(self):
        print("\tNews clusterizer by topics usin the LDA algorithm.\n")
        print("\tUsage:")
        print("\t\tspark-submit newsLDA.py <input file path> [options].\n")
        print("\tOptions:")
        print("\t\t-p=<path>\t|\tPersist computational steps in given path.")
        print("\t\t-r\t\t|\tRecompute of program steps. -p required for this argument.")
        print("\t\t-c=n\t\t|\tPrint only news that belong to given cluster.")
        print("\t\t-a\t\t|\tPrint all the results including topics keywords.")

    def printSingle(self):
        print("News in the cluster " + str(self.printOnly))
        for prediction in self.predictions.select("label", "topicDistribution").collect():
            if(int(prediction["topicDistribution"][self.printOnly]*100)>0):
                print(self.titles[prediction["label"]])

    def printPredictions(self):
        for prediction in self.predictions.select("label", "topicDistribution").collect():
            print("Topic Distribution for " + self.titles[prediction["label"]] + "\nTopics", end=" ")
            for i,topic in enumerate(prediction["topicDistribution"]):
                perc = int(topic*100)
                if(perc > 0):
                    print(str(i) + ": " + str(perc), end=" ")
            print()

    def printTopics(self, wordNumbers):
        def topic_render(vocabulary, topic):
            terms = topic["termIndices"]
            result = []
            for i in range(wordNumbers):
                term = vocabulary[terms[i]]
                result.append(term)
            return result

        vocabulary = self.sc.parallelize(self.vocabulary).collect()
        topics_final = self.topicIndices.rdd.map(lambda topic: topic_render(vocabulary, topic)).collect()

        for topic in range(len(topics_final)):
            print("Topic " + str(topic) + ":")
            for term in topics_final[topic]:
                print(term)
            print()

if __name__ == "__main__":
    clusterer = LdaClusterer()
    clusterer.main(sys.argv)
