// Databricks notebook source
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
//Test set file path: /FileStore/tables/Test_u94Q5KV.csv
//Trainning set file path: FileStore/tables/Train_UWu5bXk.csv


val training_data_raw = (spark.read.option("header","true")
                          .option("inferSchema","true")
                          .format("csv")
                          .load("Train_UWu5bXk.csv"))



val test_data_raw = (spark.read.option("header","true")
                          .option("inferSchema","true")
                          .format("csv")
                          .load("Test_u94Q5KV.csv"))
test_data_raw.printSchema

// COMMAND ----------

training_data_raw.printSchema

//Begin transformation of training data into vector form for model training

val training_data1 = (training_data_raw.select(training_data_raw("Item_Outlet_Sales").as("label"),
                             $"Item_Visibility",
                             $"Item_Type",
                             $"Item_MRP",
                             $"Outlet_Identifier",
                             $"Outlet_Location_Type",
                             $"Outlet_Type"))

val itemTypeIndexer = (new StringIndexer()
                           .setInputCol("Item_Type")
                           .setOutputCol("itemTypeIndex"))
val outletIdentifierIndexer = (new StringIndexer()
                                   .setInputCol("Outlet_Identifier")
                                   .setOutputCol("outletIdentifierIndex"))
val outletLocationTypeIndexer = (new StringIndexer()
                                   .setInputCol("Outlet_Location_Type")
                                   .setOutputCol("outletLocationTypeIndex"))
val outletTypeIndexer = (new StringIndexer()
                                   .setInputCol("Outlet_Type")
                                   .setOutputCol("outletTypeIndex"))

val itemTypeEncoder = (new OneHotEncoder()
                           .setInputCol("itemTypeIndex")
                           .setOutputCol("itemTypeVec"))
val outletIdentifierEncoder = (new OneHotEncoder()
                                   .setInputCol("outletIdentifierIndex")
                                   .setOutputCol("outletIdentifierVec"))
val outletLocationTypeEncoder = (new OneHotEncoder()
                                     .setInputCol("outletLocationTypeIndex")
                                     .setOutputCol("outletLocationTypeVec"))
val outletTypeEncoder = (new OneHotEncoder()
                             .setInputCol("outletTypeIndex")
                             .setOutputCol("outletTypeVec"))


val feature_names = (Array("Item_Visibility","itemTypeVec","Item_MRP",
                      "outletIdentifierVec","outletLocationTypeVec","outletTypeVec"))
val assembler = new VectorAssembler().setInputCols(feature_names).setOutputCol("features")


val rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features").setMaxDepth(5).setNumTrees(5)
val pipeStages = (Array(itemTypeIndexer,
                        outletIdentifierIndexer,
                        outletLocationTypeIndexer,
                        outletTypeIndexer,
                        itemTypeEncoder,
                        outletIdentifierEncoder,
                        outletLocationTypeEncoder,
                        outletTypeEncoder,
                        assembler,
                        rf))


val rf_pipeline = new Pipeline().setStages(pipeStages)
val rf_model = rf_pipeline.fit(training_data1)


val test_data1 = (test_data_raw
                     .select($"Item_Visibility",
                             $"Item_Type",
                             $"Item_MRP",
                             $"Outlet_Identifier",
                             $"Outlet_Location_Type",
                             $"Outlet_Type"))
test_data1.printSchema


val dataItemIndexer = itemTypeIndexer.fit(test_data1).transform(test_data1)
val dataItemTypeEncoder = itemTypeEncoder.transform(dataItemIndexer)
val dataOutletIdentifierIndexer=outletIdentifierIndexer.fit(dataItemTypeEncoder).transform(dataItemTypeEncoder)
val dataOutletIdentifierEncoder = outletIdentifierEncoder.transform(dataOutletIdentifierIndexer)
val dataOutletLocationTypeIndexer = outletLocationTypeIndexer.fit(dataOutletIdentifierEncoder).transform(dataOutletIdentifierEncoder)
val dataOutletLocationTypeEncoder = outletLocationTypeEncoder.transform(dataOutletLocationTypeIndexer)
val dataOutletTypeIndexer = outletTypeIndexer.fit(dataOutletLocationTypeEncoder).transform(dataOutletLocationTypeEncoder)
val interimTestData = outletTypeEncoder.transform(dataOutletTypeIndexer)


val finalTestData = assembler.transform(interimTestData).select($"features")


val predictions = rf_model.transform(test_data1)

//predictions.select("prediction").show(5)


predictions.select("prediction").write.format("csv").save("big_mart_3_sales_predictions.csv")

// COMMAND ----------
