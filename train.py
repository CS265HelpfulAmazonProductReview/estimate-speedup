import pandas as pd
import random

from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF 
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import rand
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from nltk.stem.lancaster import LancasterStemmer

# local mode

spark = SparkSession \
    .builder \
    .master("local[2]") \
    .appName("IsItHelpfull") \
    .getOrCreate()

# cluster mode
"""
spark = SparkSession \
    .builder \
    .appName("IsItHelpfull") \
    .config("spark.executor.instances", 4) \
    .config("spark.executor.cores", 1) \
    .getOrCreate()
"""
data_df = spark \
    .read.parquet("output/reviews_preprocessed.parquet")

# tokenize words
tokenizer = \
    RegexTokenizer(inputCol="reviewText", outputCol="wordsRaw", pattern="\\W")
data_df_tokenized = tokenizer.transform(data_df)

# remove stop words
remover = StopWordsRemover(inputCol="wordsRaw", outputCol="words")
data_df_filtered = remover.transform(data_df_tokenized)

# skip stemming, haven't figured out how to bootstrap EMR with nltk installed
# # stemming
# stemmer = LancasterStemmer()
# stemmer_udf = udf(
#     lambda tokens: [stemmer.stem(token) for token in tokens], 
#     ArrayType(StringType())
# )
# data_df_stemmed = data_df_filtered.withColumn("wordsStemmed", stemmer_udf("words"))

# # hashing term frequency 
# hashing_term_freq = \
#     HashingTF(inputCol="wordsStemmed", outputCol="featuresRaw", numFeatures=5000)
# data_df_tf = hashing_term_freq.transform(data_df_stemmed)

# hashing term frequency 
hashing_term_freq = \
    HashingTF(inputCol="words", outputCol="featuresRaw", numFeatures=5000)
data_df_tf = hashing_term_freq.transform(data_df_filtered)

# inverse document frequency
inv_doc_freq = IDF(inputCol="featuresRaw", outputCol="features", minDocFreq=5)
inv_doc_freq_fitted = inv_doc_freq.fit(data_df_tf)
data_df_tfidf = inv_doc_freq_fitted.transform(data_df_tf)

# encode classes
indexer = StringIndexer(inputCol="category", outputCol="label")
indexer_fitted = indexer.fit(data_df_tfidf)
data_prepared_df = indexer_fitted.transform(data_df_tfidf)
data_prepared_df = data_prepared_df.orderBy(rand())
data_prepared_df.select("label").show()
train, test = data_prepared_df.randomSplit([0.9, 0.1], seed=205);

# replicate the data into multiple copies
replication_df = spark.createDataFrame(pd.DataFrame(list(range(1,10)),columns=['replication_id']))
replicated_train = train.crossJoin(replication_df)

# parameter tunning
outSchema = StructType([StructField('replication_id', IntegerType(), True), 
    StructField('regParam', DoubleType(), True),
    StructField('elasticNetParam', DoubleType(), True),
    StructField('AUC', DoubleType(), True),
    StructField('APC', DoubleType(), True)])

@F.pandas_udf(outSchema, F.PandasUDFType.GROUPED_MAP)
def random_tune(traindf):
    regParam = random.random()
    elasticNetParam = random.random()
    log_reg = LogisticRegression(featuresCol="features", labelCol="label", predictionCol="prediction",
        maxIter=500, regParam=regParam, elasticNetParam=elasticNetParam)
    trainset, validationset = traindf.randomSplit([0.8, 0.2], seed=205);
    log_reg_fitted = log_reg.fit(trainset)
    prediction = log_reg_fitted.transform(validationset)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='probability', labelCol = 'label')
    AUC = evaluator.evaluate(prediction, {evaluator.metricName: "areaUnderROC"})
    AUP = evaluator.evaluate(prediction, {evaluator.metricName: "areaUnderPR"})
    result = pd.DataFrame({'replication_id', replication_id,
                            'regParam', regParam,
                            'elasticNetParam', elasticNetParam,
                            'AUC', AUC,
                            'APC', APC},
                            index=[0])
    return result

results = replicated_train.groupby("replication_id").apply(random_tune)
results_df = spark.createDataFrame(results)
results_df.show()
results.sort(F.desc("AUC")).show()
# train
"""
log_reg = LogisticRegression(
    featuresCol="features", labelCol="label", predictionCol="prediction",
    maxIter=100, regParam=0.3, elasticNetParam=0
)
# log_reg_fitted = log_reg.fit(data_prepared_df)
log_reg_fitted = log_reg.fit(train)
# log_reg_fitted.save("output/reviews_model.model")
log_reg_fitted.transform(test).select("features", "label", "prediction").show()
"""

"""
# parameter tunning
log_reg = LogisticRegression(maxIter = 500)
paramGrid = ParamGridBuilder()\
	.addGrid(log_reg.regParam, [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.0])\
	.addGrid(log_reg.elasticNetParam, [1.0, 0.5, 0.0])\
	.build()
tvs = TrainValidationSplit(estimator = log_reg,
							estimatorParamMaps=paramGrid,
							evaluator=BinaryClassificationEvaluator(),
                            trainRatio=0.8,
                            parallelism=8)
log_reg_fitted = tvs.fit(train)
# log_reg_fitted.transform(test).select("features", "label", "prediction").show()

# metrics
#prediction = test.rdd.map(lambda lp: (float(log_reg_fitted.predict(lp.features)), lp.label))
#metrics = BinaryClassificationMetrics(prediction)
prediction = log_reg_fitted.transform(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol='probability', labelCol = 'label')
AUC = evaluator.evaluate(prediction, {evaluator.metricName: "areaUnderROC"})
AUP = evaluator.evaluate(prediction, {evaluator.metricName: "areaUnderPR"})
print("Area under ROC = {}".format(AUC))
print("Area under PR = {}".format(AUP))
"""