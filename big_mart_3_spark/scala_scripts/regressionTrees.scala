import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}

val raw_data = (spark.read.option("header","true")
                          .option("inferSchema","true")
                          .format("csv")
                          .load("FileStore/tables/Train_UWu5bXk.csv"))


/*
Item_Visibility,Item_Type,Item_MRP,
Outlet_Identifier,Outlet_Location_Type,
Outlet_Type,Item_Outlet_Sales
*/


val data1 = (raw_data.select(raw_data("Item_Outlet_Sales").as("label"),
                             $"Item_Visibility",
                             $"Item_Type",
                             $"Item_MRP",
                             $"Outlet_Identifier",
                             $"Outlet_Location_Type",
                             $"Outlet_Type"))
 data1.show()


//Item_Type,Outlet_Identifier,Outlet_Location_Type,Outlet_Type

val itemTypeIndex =(new StringIndexer()
                          .setInputCol("Item_Type")
                          .setOutputCol("itemTypeIndex"))
val outletIdentifierIndex = (new StringIndexer()
                                 .setInputCol("Outlet_Identifier")
                                 .setOutputCol("outletIdentifierIndex"))
val outletLocationTypeIndex = (new StringIndexer()
                              .setInputCol("Outlet_Location_Type")
                              .setOutputCol("outletLocationTypeIndex"))
val outletTypeIndex = (new StringIndexer()
                      .setInputCol("Outlet_Type")
                      .setOutputCol("outletTypeIndex"))


val itemTypeEncoder = new OneHotEncoder().setInputCol("itemTypeIndex").setOutputCol("itemTypeVec")
val outletIdentifierEncoder = new OneHotEncoder().setInputCol("outletIdentifierIndex").setOutputCol("outletIdentifierVec")
val outletLocationTypeEncoder = new OneHotEncoder().setInputCol("outletLocationTypeIndex").setOutputCol("outletLocationTypeVec")
val outletTypeEncoder = new OneHotEncoder().setInputCol("outletTypeIndex").setOutputCol("outletTypeVec")


val dataItemIndexer = itemTypeIndex.fit(data1).transform(data1)
val dataItemTypeEncoder = itemTypeEncoder.transform(dataItemIndexer)
val dataOutletIdentifierIndexer=outletIdentifierIndex.fit(dataItemTypeEncoder).transform(dataItemTypeEncoder)
val dataOutletIdentifierEncoder = outletIdentifierEncoder.transform(dataOutletIdentifierIndexer)
val dataOutletLocationTypeIndexer = outletLocationTypeIndex.fit(dataOutletIdentifierEncoder).transform(dataOutletIdentifierEncoder)
val dataOutletLocationTypeEncoder = outletLocationTypeEncoder.transform(dataOutletLocationTypeIndexer)
val dataOutletTypeIndexer = outletTypeIndex.fit(dataOutletLocationTypeEncoder).transform(dataOutletLocationTypeEncoder)
val interimData = outletTypeEncoder.transform(dataOutletTypeIndexer)


//interimData.printSchema

val feature_names = (Array("Item_Visibility","itemTypeVec","Item_MRP",
                      "outletIdentifierVec","outletLocationTypeVec","outletTypeVec"))
val assembler = new VectorAssembler().setInputCols(feature_names).setOutputCol("features")


val finalData = assembler.transform(interimData).select($"label",$"features")


val Array(training2,test2) = finalData.randomSplit(Array(0.8,0.2),seed=12345)


val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("features")


val dtModel = dt.fit(training2)


dt.extractParamMap


val predictions = dtModel.transform(test2)


val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("r2")


val r2 = evaluator.evaluate(predictions)
println("R2 on test data = " + r2)


val rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features")

val rfModel = rf.fit(training2)


rfModel.extractParamMap


val rf_predictions = rfModel.transform(test2)


val rf_r2 = evaluator.evaluate(rf_predictions)
println("Random Forest R2 on test data = " + rf_r2)


val rf_5 = new RandomForestRegressor()
               .setLabelCol("label")
               .setFeaturesCol("features")
               .setMaxDepth(5)
               .setNumTrees(5)
val rf_10 = new RandomForestRegressor()
               .setLabelCol("label")
               .setFeaturesCol("features")
               .setMaxDepth(5)
               .setNumTrees(10)
val rf_15 = new RandomForestRegressor()
               .setLabelCol("label")
               .setFeaturesCol("features")
               .setMaxDepth(5)
               .setNumTrees(15)


val rf5Model = rf_5.fit(training2)
val rfPreds5 = rf5Model.transform(test2)
val rf5R2 = evaluator.evaluate(rfPreds5)

val rf10Model = rf_10.fit(training2)
val rfPreds10 = rf10Model.transform(test2)
val rf10R2 = evaluator.evaluate(rfPreds10)

val rf15Model = rf_15.fit(training2)
val rfPreds15 = rf15Model.transform(test2)
val rf15R2 = evaluator.evaluate(rfPreds15)


println("Random Forest R2(5 Trees) on test data = " + rf5R2)
println("Random Forest R2(10 Trees) on test data = " + rf10R2)
println("Random Forest R2(15 Trees) on test data = " + rf15R2)


val rf_25 = new RandomForestRegressor()
               .setLabelCol("label")
               .setFeaturesCol("features")
               .setMaxDepth(5)
               .setNumTrees(25)
val rf_30 = new RandomForestRegressor()
               .setLabelCol("label")
               .setFeaturesCol("features")
               .setMaxDepth(5)
               .setNumTrees(30)
val rf_35 = new RandomForestRegressor()
               .setLabelCol("label")
               .setFeaturesCol("features")
               .setMaxDepth(5)
               .setNumTrees(35)

val rf25Model = rf_25.fit(training2)
val rfPreds25 = rf25Model.transform(test2)
val rf25R2 = evaluator.evaluate(rfPreds25)

val rf30Model = rf_30.fit(training2)
val rfPreds30 = rf30Model.transform(test2)
val rf30R2 = evaluator.evaluate(rfPreds30)

val rf35Model = rf_35.fit(training2)
val rfPreds35 = rf35Model.transform(test2)
val rf35R2 = evaluator.evaluate(rfPreds35)

println("Random Forest R2(25 Trees) on test data = " + rf25R2)
println("Random Forest R2(30 Trees) on test data = " + rf30R2)
println("Random Forest R2(35 Trees) on test data = " + rf35R2)


val rf_200 = new RandomForestRegressor()
               .setLabelCol("label")
               .setFeaturesCol("features")
               .setMaxDepth(5)
               .setNumTrees(200)
val rf200Model = rf_200.fit(training2)
val rfPreds200 = rf200Model.transform(test2)
val rf200R2 = evaluator.evaluate(rfPreds200)

println("Random Forest R2(200 Trees) on test data = " + rf200R2)
//Random Forest R2(100 Trees) on test data = 0.5951664321564244


val gbt = new GBTRegressor()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setMaxIter(10)

val gbtModel = gbt.fit(training2)
val gbtPreds = gbtModel.transform(test2)
val gbtR2 = evaluator.evaluate(gbtPreds)

println("Random Forest R2(gbt Trees) on test data = " + gbtR2)

gbtModel.extractParamMap
