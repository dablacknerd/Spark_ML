import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.RegressionEvaluator


val raw_data = (spark.read.option("header","true")
                          .option("inferSchema","true")
                          .format("csv")
                          .load("FileStore/tables/Train_UWu5bXk.csv"))


val data1 = (raw_data.select(raw_data("Item_Outlet_Sales").as("label"),
                             $"Item_Visibility",
                             $"Item_Type",
                             $"Item_MRP",
                             $"Outlet_Identifier",
                             $"Outlet_Location_Type",
                             $"Outlet_Type"))
 //data1.show()


//Create objects to index string features
val itemTypeIndex =new StringIndexer().setInputCol("Item_Type").setOutputCol("itemTypeIndex")
val outletIdentifierIndex = new StringIndexer().setInputCol("Outlet_Identifier").setOutputCol("outletIdentifierIndex")
val outletLocationTypeIndex = new StringIndexer().setInputCol("Outlet_Location_Type").setOutputCol("outletLocationTypeIndex")
val outletTypeIndex = new StringIndexer().setInputCol("Outlet_Type").setOutputCol("outletTypeIndex")

//Create objects to perform one hot encoding on indexed features
val itemTypeEncoder = new OneHotEncoder().setInputCol("itemTypeIndex").setOutputCol("itemTypeVec")
val outletIdentifierEncoder = new OneHotEncoder().setInputCol("outletIdentifierIndex").setOutputCol("outletIdentifierVec")
val outletLocationTypeEncoder = new OneHotEncoder().setInputCol("outletLocationTypeIndex").setOutputCol("outletLocationTypeVec")
val outletTypeEncoder = new OneHotEncoder().setInputCol("outletTypeIndex").setOutputCol("outletTypeVec")


//Peform conversion of categorial features into indexes and one hot encode them
val dataItemIndexer = itemTypeIndex.fit(data1).transform(data1)
val dataItemTypeEncoder = itemTypeEncoder.transform(dataItemIndexer)
val dataOutletIdentifierIndexer=outletIdentifierIndex.fit(dataItemTypeEncoder).transform(dataItemTypeEncoder)
val dataOutletIdentifierEncoder = outletIdentifierEncoder.transform(dataOutletIdentifierIndexer)
val dataOutletLocationTypeIndexer = outletLocationTypeIndex.fit(dataOutletIdentifierEncoder).transform(dataOutletIdentifierEncoder)
val dataOutletLocationTypeEncoder = outletLocationTypeEncoder.transform(dataOutletLocationTypeIndexer)
val dataOutletTypeIndexer = outletTypeIndex.fit(dataOutletLocationTypeEncoder).transform(dataOutletLocationTypeEncoder)
val interimData = outletTypeEncoder.transform(dataOutletTypeIndexer)


val feature_names = (Array("Item_Visibility","itemTypeVec","Item_MRP",
                      "outletIdentifierVec","outletLocationTypeVec","outletTypeVec"))
val assembler = new VectorAssembler().setInputCols(feature_names).setOutputCol("features")


val finalData = assembler.transform(interimData).select($"label",$"features")
val Array(training2,test2) = finalData.randomSplit(Array(0.8,0.2),seed=12345)


//when setElasticNetParam = 0.0, the penalty is L2 meaning ridge regression
//when setElasticNetParam = 1.0, the penalty is L1 meaning lasso regression

val regular_lr = new LinearRegression()
val ridge_lr = new LinearRegression().setRegParam(10.0).setElasticNetParam(0.0)
val lasso_lr = new LinearRegression().setRegParam(13.0).setElasticNetParam(1.0)


val lrModel = regular_lr.fit(training2)
val ridgeModel = ridge_lr.fit(training2)
val lassoModel = lasso_lr.fit(training2)


val predictions = lrModel.transform(test2)
val ridgePredictions = ridgeModel.transform(test2)
val lassoPredictions = lassoModel.transform(test2)


val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("r2")


val regularR2 = evaluator.evaluate(predictions)
val ridgeR2 = evaluator.evaluate(ridgePredictions)
val lassoR2 = evaluator.evaluate(lassoPredictions)

println("Regular Linear Regression R2 on test data = " + regularR2)
println("Ridge Regression R2 on test data = " + ridgeR2)
println("Lasso Regression R2 on test data = " + lassoR2)
