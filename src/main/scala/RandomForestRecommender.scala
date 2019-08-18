/**
  * Author:IceS
  * Date:2019-08-18 07:52:37
  * Description:
  * NONE
  */

import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.sql.SparkSession
import org.slf4j.{Logger, LoggerFactory}


case class RandomForestParams(
                     k: Int = 5,
                     maxIterations: Int = 20,
                     numClass: Int = 10,
                     numTrees: Int = 5,
                     featureSubsetStrategy: String = "auto",
                     impurity: String = "gini",
                     maxDepth: Int = 5,
                     maxBins: Int = 100) extends Params {
  override def getName(): String = this.getClass.getSimpleName.replace("Params", "")

  override def toString: String = {
    s"${this.getClass.getSimpleName}:{k:$k,maxIterations:$maxIterations,numClass:$numClass,numTrees:$numTrees,featureSubsetStrategy:$featureSubsetStrategy,impurity:$impurity,maxDepth:$maxDepth,maxBins:$maxBins}\r\n"
  }
}

class RandomForestRecommender(ap: RandomForestParams) extends Recommender {

  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)
  override def getParams: Params = ap

  private var userId2Index: BiMap[Int, Long] = _
  private var itemId2Index: BiMap[Int, Long] = _

  override def prepare(data: Seq[Rating]): PrepairedData = {
    //生成全部用户ID
    val userIDSet: Seq[Int] = data.map(_.user).distinct
    userId2Index = BiMap.toIndex(userIDSet)

    //生成全部物品ID
    val itemIDSet = data.map(_.item).distinct
    itemId2Index = BiMap.toIndex(itemIDSet)

    new PrepairedData(data)
  }



  override def train(trainingData: TrainingData): Unit = {
    //所有物品的大小
    val userGroup = trainingData.ratings.groupBy(_.user)
    userHasItem = userGroup

    //1.对用户聚类。直接用稀疏矩阵尝试
    val userVector: Map[Int, linalg.Vector] = userGroup.map(r => {
      //r._1//userId
      val itemSeq = r._2.map(r2 => {
        (itemId2Index(r2.item).toInt, r2.rating)
      })
      (r._1, Vectors.sparse(itemId2Index.size, itemSeq))
    })

    val itemGroup = trainingData.ratings.groupBy(_.item)
    val itemVector: Map[Int, linalg.Vector] = itemGroup.map(r => {
      //r._1//itemId
      val userSeq = r._2.map(r2 => {
        (userId2Index(r2.user).toInt, r2.rating)
      })
      (r._1, Vectors.sparse(userId2Index.size, userSeq))
    })

    //2.根据聚类的中心，生成新向量。
    val sparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()

    val bkmUser = new BisectingKMeans().setK(ap.k).setMaxIterations(ap.maxIterations)
    val rddUserVector = sparkSession.sparkContext.parallelize(userVector.values.toSeq)
    val userModel = bkmUser.run(rddUserVector)
    //调试信息
    userModel.clusterCenters.foreach(println)
    //生成新特征向量

    newUserVector = userVector.map(r => {
      //r._1//userID
      val distance: Array[Double] = userModel.clusterCenters.map(c => {
        Correlation.getDistance(r._2, c)
      })
      val sum = distance.sum
      val fdistance = distance.map(r2 => {
        r2 / sum
      })
      (r._1, fdistance)
    })

    val bkmItem = new BisectingKMeans().setK(ap.k).setMaxIterations(ap.maxIterations)
    val rddItemVector = sparkSession.sparkContext.parallelize(itemVector.values.toSeq)
    val itemModel = bkmItem.run(rddItemVector)
    //调试信息
    itemModel.clusterCenters.foreach(println)

    newItemVector = itemVector.map(r => {
      //r._1//userID
      val distance: Array[Double] = itemModel.clusterCenters.map(c => {
        Correlation.getDistance(r._2, c)
      })
      val sum = distance.sum
      val fdistance = distance.map(r2 => {
        r2 / sum
      })
      (r._1, fdistance)
    })

    //建立随机森林模型
    //3.2 处理处理数据格式
    val dt= trainingData.ratings.map(r => {
      val userV=newUserVector(r.user)
      val itemV=newItemVector(r.item)
      val arr = new Array[Double](userV.length + itemV.length)

      userV.indices.foreach(idx => {
        arr(idx) = userV(idx)
      })

      itemV.indices.foreach(idx => {
        arr(idx + userV.length) = itemV(idx)
      })
      LabeledPoint(r.rating,Vectors.dense(arr))
    })

    //3.3 准备模型参数

    //设定输入数据格式
    val categoricalFeaturesInfo = Map[Int, Int]()


     model = RandomForest.trainClassifier(
      sparkSession.sparkContext.parallelize(dt),
      ap.numClass,
      categoricalFeaturesInfo,
      ap.numTrees,
      ap.featureSubsetStrategy.toLowerCase(),
      ap.impurity.toLowerCase(),
      ap.maxDepth,
      ap.maxBins)


    //sparkSession.close()



  }

  private var model: RandomForestModel = _
  //训练集中用户所拥有item
  private var userHasItem: Map[Int, Seq[Rating]] = _

  private var newItemVector: Map[Int, Array[Double]] = _
  private var newUserVector: Map[Int, Array[Double]] = _

  override def predict(query: Query): PredictedResult = {

    //判断有没有用户向量
    if(!newUserVector.contains(query.user)){
      //该用户没有最相似的用户列表
      logger.warn(s"该用户:${query.user}没有相似用户列表，无法生成推荐！")
      return PredictedResult(Array.empty)
    }

    //生成候选物品列表
    val userHad=userHasItem(query.user).map(_.item).toSet
    val items= itemId2Index.toMap.keys.filter(r=>{
      userHad.nonEmpty && !userHad.contains(r)
    })

    //根据当前用户的ID和物品列表生成预测集
    val userV =newUserVector(query.user)//用户特征向量
    val result= items.filter(r=>{
      newItemVector.contains(r)
    }).map(r=>{

      val itemV =newItemVector(r)
      val arr = new Array[Double](userV.length + itemV.length)
      userV.indices.foreach(idx => {
        arr(idx) = userV(idx)
      })

      itemV.indices.foreach(idx => {
        arr(idx + userV.length) = itemV(idx)
      })

      ItemScore(r, model.predict(Vectors.dense(arr)))
    }).toArray.sortBy(_.score).reverse.take(query.num)


    PredictedResult(result)

  }
}


