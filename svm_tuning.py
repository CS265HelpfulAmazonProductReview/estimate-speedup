import random
import time

from multiprocessing.pool import ThreadPool
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, rand
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF 
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import rand

# local mode
spark = SparkSession \
    .builder \
    .appName("IsItHelpfull") \
    .master("local[1]") \
    .getOrCreate()


# cluster mode
"""
spark = SparkSession \
    .builder \
    .appName("IsItHelpfull") \
    .config("spark.executor.instances", 1) \
    .config("spark.executor.cores", 1) \
    .getOrCreate()
"""

def category_review(votes):
    score = votes[0]/votes[1]
    return "good" if score >= 0.8 else "else"

category_review_udf = udf(category_review, StringType())

review_df = spark \
    .read.json("clothing.json") \
    .where(col("helpful")[1] >= 5) \
    .withColumn("category", category_review_udf("helpful")) \
    .select("reviewText", "category") \
    .cache()

# up-sample the else category, so that #else = #good
review_df_good = review_df.where(col("category") == "good").cache()
review_df_else = review_df.where(col("category") == "else").cache()
n_good = review_df_good.count()
n_else = review_df_else.count()
fraction = float(n_good)/n_else
review_df_else_upsampled = \
    review_df_else.sample(withReplacement=True, fraction=fraction)
review_df_preprocessed = review_df_good.unionAll(review_df_else_upsampled)

review_df_preprocessed.show()

print(n_good)
print(n_else)

# tokenize words
tokenizer = \
    RegexTokenizer(inputCol="reviewText", outputCol="wordsRaw", pattern="\\W")
data_df_tokenized = tokenizer.transform(review_df_preprocessed)

# remove stop words
remover = StopWordsRemover(inputCol="wordsRaw", outputCol="words")
data_df_filtered = remover.transform(data_df_tokenized)

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

pool = ThreadPool(1)

start = time.time()

def random_tune(df, parameters):
    trainset, validationset = df.randomSplit([0.8, 0.2], seed=205)
    lsvc = LinearSVC(maxIter=100, regParam=parameters[0], threshold=parameters[1])
    lsvc_fitted = lsvc.fit(trainset)
    prediction = lsvc_fitted.transform(validationset)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol = 'label')
    AUC = evaluator.evaluate(prediction, {evaluator.metricName: "areaUnderROC"})
    AUP = evaluator.evaluate(prediction, {evaluator.metricName: "areaUnderPR"})
    print("reg: {} threshold: {} AUC {} AUP {}".format(parameters[0], parameters[1], AUC, AUP))

parameters = [[random.random()/10, (-0.2+0.4*random.random())] for i in range(9)]
parameters.append([0.0, 0.0])

pool.map(lambda parameters: random_tune(data_prepared_df, parameters), parameters)

end = time.time()
print("Tuning time is {} seconds in total".format(end - start))
