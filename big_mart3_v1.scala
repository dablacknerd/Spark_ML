//FileStore/tables/Train_UWu5bXk.csv
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline

val raw_data = (spark.read.option("header","true")
                          .option("inferSchema","true")
                          .format("csv")
                          .load("FileStore/tables/Train_UWu5bXk.csv"))
raw_data.printSchema()

/*
Item_Visibility,Item_Type,Item_MRP,
Outlet_Identifier,Outlet_Location_Type,
Outlet_Type,Item_Outlet_Sales
*/

val data1 = (raw_data.select(raw_data("Item_Outlet_Sales").as("label"),
                                  $"Item_Visibility",$"Item_Type",
                                  $"Item_MRP",$"Outlet_Identifier",
                                  $"Outlet_Location_Type",$"Outlet_Type"))
data1.printSchema

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

val Array(training,test) = data1.randomSplit(Array(0.8,0.2),seed = 12345)

val lr_regular = new LinearRegression()
val pipeStages = (Array(itemTypeIndexer,
                        outletIdentifierIndexer,
                        outletLocationTypeIndexer,
                        outletTypeIndexer,
                        itemTypeEncoder,
                        outletIdentifierEncoder,
                        outletLocationTypeEncoder,
                        outletTypeEncoder,
                        assembler,
                        lr_regular))

val dataItemTypeIndexer = itemTypeIndexer.fit(data1).transform(data1)
val dataItemTypeVec = itemTypeEncoder.transform(dataItemTypeIndexer)

val dataOutletIdentifierIndexer = outletIdentifierIndexer.fit(dataItemTypeVec).transform(dataItemTypeVec)
val dataOutletIdentifierVec = outletIdentifierEncoder.transform(dataOutletIdentifierIndexer)

val dataoutletLocationTypeIndexer = (outletLocationTypeIndexer.fit(dataOutletIdentifierVec)
                                                                                      .transform(dataOutletIdentifierVec))
val datadataoutletLocationTypeVec = outletLocationTypeEncoder.transform(dataoutletLocationTypeIndexer)

val dataoutletTypeIndexer = (outletTypeIndexer.fit(datadataoutletLocationTypeVec)
                                                                      .transform(datadataoutletLocationTypeVec))
val dataInterim = outletTypeEncoder.transform(dataoutletTypeIndexer)
dataInterim.printSchema

val finalData = assembler.transform(dataInterim).select($"label",$"features")
val Array(training2,test2) = finalData.randomSplit(Array(0.8,0.2),seed=12345)
val lr1 = new LinearRegression()
val lr1Model = lr1.fit(training2)
val summary1 = lr1Model.summary

val mse = summary1.meanSquaredError
val rmse = summary1.rootMeanSquaredError
val r2 = summary1.r2

println(s"MSE: $mse")
println(s"RMSE: $rmse")
println(s"R2: $r2")

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder,TrainValidationSplit}

val ridgeLr = new LinearRegression()
val paramGrid = (new ParamGridBuilder().addGrid(ridgeLr.regParam,Array(11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,10.0))
                                       .addGrid(ridgeLr.fitIntercept)
                                       .addGrid(ridgeLr.elasticNetParam,Array(0.0))
                                       .build())
val evaluator = new RegressionEvaluator().setMetricName("r2")

val trainvalsplit = (new TrainValidationSplit()
                         .setEstimator(ridgeLr)
                         .setEvaluator(evaluator)
                         .setEstimatorParamMaps(paramGrid)
                         .setTrainRatio(0.8))

val model = trainvalsplit.fit(finalData)
model.bestModel.extractParamMap

//ridge regression with best tuning parameter equals 10
val lr2 = (new LinearRegression()
               .setRegParam(10.0)
               .setElasticNetParam(0.0))

val lr2Model = lr2.fit(training2)
val summary2 = lr2Model.summary

val ridgeMse = summary2.meanSquaredError
val ridgeRmse = summary2.rootMeanSquaredError
val ridgeR2 = summary2.r2

println(s"MSE: $ridgeMse")
println(s"RMSE: $ridgeRmse")
println(s"R2: $ridgeR2")

val lassoLr = new LinearRegression()
val paramGridLasso = (new ParamGridBuilder()
                          .addGrid(lassoLr.regParam,Array(11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0))
                          .addGrid(lassoLr.fitIntercept)
                          .addGrid(lassoLr.elasticNetParam,Array(1.0))
                          .build())
val lassoEvaluator = new RegressionEvaluator().setMetricName("r2")

val trainvalsplit_lasso = (new TrainValidationSplit()
                         .setEstimator(lassoLr)
                         .setEvaluator(lassoEvaluator)
                         .setEstimatorParamMaps(paramGridLasso)
                         .setTrainRatio(0.8))

val modelLasso = trainvalsplit_lasso.fit(finalData)
modelLasso.bestModel.extractParamMap

//Lasso regression with best tuning parameter equals 13
val lr3 = (new LinearRegression()
               .setRegParam(13.0)
               .setElasticNetParam(1.0))

val lr3Model = lr3.fit(training2)
val summary3 = lr3Model.summary

val lassoMse = summary3.meanSquaredError
val lassoRmse = summary3.rootMeanSquaredError
val lassoR2 = summary3.r2

println(s"MSE: $lassoMse")
println(s"RMSE: $lassoRmse")
println(s"R2: $lassoR2")

//Plain Linear regression:
//MSE: 1268121.7729232428
//RMSE: 1126.109130112727
//R2: 0.5600496100156415

//Rigde Linear Regression with tuning parameter = 11
//MSE: 1268224.0570058532
//RMSE: 1126.1545440150978
//R2: 0.5600141245260029

//Lasso Linear Regression with tuning parameter = 13
//MSE: 1270801.0473687379
//RMSE: 1127.2981182317026
//R2: 0.5591200874237744

//Based on R2, Regular or Rigde regression would perform best.
