from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, rand
from pyspark.sql.types import *
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer


# local mode

spark = SparkSession \
    .builder \
    .appName("IsItHelpfull") \
    .master("local[4]") \
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

# read json file

def category_review(votes):
    score = votes[0]/votes[1]
    return "good" if score >= 0.8 else "else"

category_review_udf = udf(category_review, StringType())

review_df = spark \
    .read.json("data/reviews.json") \
    .where(col("helpful")[1] >= 5) \
    .withColumn("category", category_review_udf("helpful")) \
    .select("reviewText", "category") \
    .cache()

print(review_df.rdd)

# up-sample the else category, so that #else = #good
review_df_good = review_df.where(col("category") == "good").cache()
review_df_else = review_df.where(col("category") == "else").cache()
n_good = review_df_good.count()
n_else = review_df_else.count()
fraction = float(n_good)/n_else
review_df_else_upsampled = \
    review_df_else.sample(withReplacement=True, fraction=fraction)
review_df_preprocessed = review_df_good.unionAll(review_df_else_upsampled)


# tokenize words
tokenizer = \
    RegexTokenizer(inputCol="reviewText", outputCol="wordsRaw", pattern="\\W")
review_df_tokenized = tokenizer.transform(review_df_preprocessed)

# remove stop words
remover = StopWordsRemover(inputCol="wordsRaw", outputCol="words")
review_df_filtered = remover.transform(review_df_tokenized)

# hashing term frequency 
hashing_term_freq = \
    HashingTF(inputCol="words", outputCol="featuresRaw", numFeatures=5000)
review_df_tf = hashing_term_freq.transform(review_df_filtered)

# inverse document frequency
inv_doc_freq = IDF(inputCol="featuresRaw", outputCol="features", minDocFreq=5)
inv_doc_freq_fitted = inv_doc_freq.fit(review_df_tf)
review_df_tfidf = inv_doc_freq_fitted.transform(data_df_tf)

# encode classes
indexer = StringIndexer(inputCol="category", outputCol="label")
indexer_fitted = indexer.fit(review_df_tfidf)
review_prepared_df = indexer_fitted.transform(review_df_tfidf)

"""
review_prepared_df = review_prepared_df.orderBy(rand())
train, test = data_prepared_df.randomSplit([0.9, 0.1], seed=205);
"""
