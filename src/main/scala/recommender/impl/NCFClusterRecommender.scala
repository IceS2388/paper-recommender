package recommender.impl

/**
  * Author:IceS
  * Date:2019-08-18 16:29:57
  * Description:
  * 结合NCF和Cluster集群算法
  */

import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.misc.ElementWiseMultiplicationLayer
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}
import recommender._
import recommender.tools.{Correlation, NearestUserAccumulator}

import scala.collection.mutable
import scala.util.Random


case class NCFClusterParams(
                             method: String = "Cosine",
                             k: Int = 4,
                             maxIterations: Int = 20,
                             numNearestUsers: Int = 60,
                             numUserLikeMovies: Int = 100) extends Params {
  override def getName(): String = {
    this.getClass.getSimpleName.replace("Params", "") + s"_$method"
  }

  override def toString: String = s"聚类参数{method:$method,k:$k,maxIterations:$maxIterations,numNearestUsers:$numNearestUsers,numUserLikeMovies:$numUserLikeMovies}\r\n"
}


class NCFClusterRecommender(ap: NCFClusterParams) extends Recommender {
  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def getParams: Params = ap


  //训练集中用户所拥有item
  private var userHasItem: Map[Int, Seq[Rating]] = _

  //每个用户所有的物品
  private var allUserItemSet: Map[Int, Set[Int]] = _

  //用户的评分向量
  private var afterClusterRDD: Array[(Int, (Int, linalg.Vector))] = _

  override def prepare(data: Seq[Rating]): PrepairedData = {

    allUserItemSet = data.groupBy(_.user).map(r => {
      val userId = r._1
      //
      val itemSet = r._2.map(_.item).toSet
      (userId, itemSet)
    })
    //数据分割前
    new PrepairedData(data)
  }


  private var newItemVector: collection.Map[Int, linalg.Vector] = _
  private var newUserVector: collection.Map[Int, linalg.Vector] = _

  override def train(data: TrainingData): Unit = {

    require(data.ratings.nonEmpty, "训练数据不能为空！")

    //1.获取训练集中每个用户的观看列表
    userHasItem = data.ratings.groupBy(_.user)

    //2.生成用户喜欢的电影
    userLikedMap = userLikedItems()

    //实现新的统计方法
    val userVectors = userHasItem.map(r => {
      val uid = r._1
      var c1 = 0D //1.0
      var c2 = 0D //2.0
      var c3 = 0D //3.0
      var c4 = 0D //4.0
      var c5 = 0D //5.0

      r._2.foreach(r2 => {
        if (r2.rating == 1.0)
          c1 += 1
        else if (r2.rating == 2.0)
          c2 += 1
        else if (r2.rating == 3.0)
          c3 += 1
        else if (r2.rating == 4.0)
          c4 += 1
        else
          c5 += 1
      })

      (uid, Vectors.dense(c1, c2, c3, c4, c5))
    }).toSeq

    //2.生成物品的统计列表
    val itemVectors = data.ratings.groupBy(_.item).map(r => {
      val iid = r._1

      var c1 = 0D //1.0
      var c2 = 0D //2.0
      var c3 = 0D //3.0
      var c4 = 0D //4.0
      var c5 = 0D //5.0

      r._2.foreach(r2 => {
        if (r2.rating == 1.0)
          c1 += 1
        else if (r2.rating == 2.0)
          c2 += 1
        else if (r2.rating == 3.0)
          c3 += 1
        else if (r2.rating == 4.0)
          c4 += 1
        else
          c5 += 1
      })
      (iid, Vectors.dense(c1, c2, c3, c4, c5))
    }).toSeq

    newUserVector = userVectors.toMap
    newItemVector = itemVectors.toMap


    val sparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()


    /** -------------------对用户评分向量进行聚类--------------------- **/
    logger.info("正在对用户评分向量进行聚类，需要些时间...")
    //3.准备聚类

    val userVectorsRDD = sparkSession.sparkContext.parallelize(userVectors)
    //val itemVectorsRDD = sparkSession.sparkContext.parallelize(itemVectors)

    val bkm = new BisectingKMeans().setK(ap.k).setMaxIterations(ap.maxIterations)
    val model = bkm.run(userVectorsRDD.map(_._2))

    //model.clusterCenters.foreach(println)

    //4.聚类用户评分向量(族ID,评分向量),根据聚类计算用户之间的相似度使用
    afterClusterRDD = userVectorsRDD.map(r => {
      (model.predict(r._2), r)
    }).collect()

    /** -------------------生成共享数据--------------------- **/

    sparkSession.close()

    //3.根据用户评分向量生成用户最邻近用户的列表
    logger.info("计算用户邻近的相似用户中....")
    nearestUser = userNearestTopN()


    /** -------------------神经网络--------------------- **/
    val userVS = newUserVector.head._2.size
    val itemVS = newItemVector.head._2.size
    val unitMax = Math.max(userVS, itemVS)
    val unionSize = userVS + itemVS


    //1.配置网络结构
    val computationGraphConf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .activation(Activation.RELU)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new org.nd4j.linalg.learning.config.AdaDelta())
      .l2(1e-3)
      .graphBuilder()
      .addInputs("user_input", "item_input")

      .addLayer("userLayer", new DenseLayer.Builder().nIn(userVS).nOut(userVS).activation(Activation.IDENTITY).build(), "user_input")
      .addLayer("itemLayer", new DenseLayer.Builder().nIn(itemVS).nOut(itemVS).activation(Activation.IDENTITY).build(), "item_input")
      .addLayer("GML", new ElementWiseMultiplicationLayer.Builder().nIn(unionSize).nOut(unionSize).build(), "userLayer", "itemLayer")

      .addVertex("input", new MergeVertex(), "userLayer", "itemLayer")
      .addLayer("MLP4", new DenseLayer.Builder().nIn(unionSize).nOut(4 * unionSize).build(), "input")
      .addLayer("MLP2", new DenseLayer.Builder().nIn(4 * unionSize).nOut(2 * unionSize).build(), "MLP4")
      .addLayer("MLP1", new DenseLayer.Builder().activation(Activation.SIGMOID).nIn(2 * unionSize).nOut(unionSize).build(), "MLP2")
      .addVertex("ncf", new MergeVertex(), "GML", "MLP1")
      .addLayer("out", new OutputLayer.Builder().nIn(unionSize + unionSize).activation(Activation.SOFTMAX).nOut(2).build(), "ncf")
      .setOutputs("out")
      .build()


    ncfModel = new ComputationGraph(computationGraphConf)
    ncfModel.init()

    val allItemsSet = data.ratings.map(_.item).distinct.toSet
    val size = data.ratings.size

    //2.正样本数据
    logger.info("正样本数据")
    val positiveData: Seq[(Int, INDArray, INDArray, INDArray)] = data.ratings.map(r => {
      //构建特征数据
      val userV: linalg.Vector = newUserVector(r.user)
      val itemV: linalg.Vector = newItemVector(r.item)

      val userArray = new Array[Float](userV.size)
      for (idx <- 0 until userV.size) {
        userArray(idx) = userV(idx).toFloat
      }
      val itemArray = new Array[Float](itemV.size)
      for (idx <- 0 until itemV.size) {
        itemArray(idx) = itemV(idx).toFloat
      }

      //生成标签
      val la = new Array[Float](2)
      la(0) = 0
      la(1) = 1

      (Random.nextInt(size),
        Nd4j.create(userArray).reshape(1, userArray.length),
        Nd4j.create(itemArray).reshape(1, itemArray.length),
        Nd4j.create(la).reshape(1, 2))
    })

    //3.负样本数据
    logger.info("负样本数据")
    val negativeData: Seq[(Int, INDArray, INDArray, INDArray)] = userHasItem.flatMap(r => {
      val userHadSet = r._2.map(_.item).distinct.toSet
      //TODO 这里有bug，测试集的数据可能在里面
      val negativeSet = allItemsSet.diff(userHadSet)
      //保证负样本的数量，和正样本数量一致
      val nSet = Random.shuffle(negativeSet).take(100)

      val userV = newUserVector(r._1)

      nSet.map(itemID => {
        val itemV = newItemVector(itemID)

        val userArray = new Array[Float](userV.size)
        for (idx <- 0 until userV.size) {
          userArray(idx) = userV(idx).toFloat
        }
        val itemArray = new Array[Float](itemV.size)
        for (idx <- 0 until itemV.size) {
          itemArray(idx) = itemV(idx).toFloat
        }

        //生成标签
        val la = new Array[Float](2)
        la(0) = 1
        la(1) = 0
        (Random.nextInt(size),
          Nd4j.create(userArray).reshape(1, userArray.length),
          Nd4j.create(itemArray).reshape(1, itemArray.length),
          Nd4j.create(la).reshape(1, 2))
      })
    }).toSeq

    //4.数据打散
    logger.info("数据打散")
    val trainTempData = new Array[(Int, INDArray, INDArray, INDArray)](positiveData.size + negativeData.size)
    var idx = 0
    positiveData.foreach(r => {
      trainTempData(idx) = r
      idx += 1
    })

    negativeData.foreach(r => {
      trainTempData(idx) = r
      idx += 1
    })

    val finalData = trainTempData.sortBy(_._1).map(r => {
      (r._2, r._3, r._4)
    })

    //训练模型
    logger.info("开始训练模型...")
    var logIdx = 0
    finalData.foreach(r => {
      if (logIdx % 100 == 0) {
        logger.info(s"index:$logIdx")
      }

      ncfModel.fit(Array(r._1, r._2), Array(r._3))
      logIdx += 1
    })
    logger.info("训练模型完成")


  }


  def userLikedItems(): Map[Int, Seq[Rating]] = {


    val groupRDD = userHasItem
    //1.计算用户的平均分
    val userMean = groupRDD.map(r => {
      val userLikes: Seq[Rating] = r._2.toList.sortBy(_.rating).reverse.take(ap.numUserLikeMovies)
      (r._1, userLikes)
    })
    userMean
  }

  def userNearestTopN(): mutable.Map[String, Double] = {
    //afterClusterRDD: RDD[(Int, (Int, linalg.Vector))]
    //                簇Index  Uid    评分向量

    //计算所有用户之间的相似度
    val userNearestAccumulator = new NearestUserAccumulator

    //1.考虑从族中进行计算相似度
    for (idx <- 0 until ap.k) {

      val curUsersVectors: Array[(Int, linalg.Vector)] = afterClusterRDD.filter(_._1 == idx).map(_._2)
      logger.info(s"簇$idx 中用户数量为：${curUsersVectors.length}")

      val uids: Array[(Int, linalg.Vector)] = curUsersVectors.sortBy(_._1)

      for {
        (u1, v1) <- uids
        (u2, v2) <- uids
        if u1 < u2
      } {

        val score = if (ap.method.toLowerCase() == "cosine") {
          Correlation.getCosine(v1, v2)
        } else if (ap.method.toLowerCase() == "improvedpearson") {
          Correlation.getImprovedPearson(v1, v2)
        } else if (ap.method.toLowerCase() == "pearson") {
          Correlation.getPearson(v1, v2)
        } else {
          throw new Exception("没有找到对应的方法。")
        }
        //logger.info(s"score:$score")
        if (score > 0) {
          userNearestAccumulator.add(u1, u2, score)
        } //end  if (score > 0) {
      }
      logger.info(s"累加器数据条数：${userNearestAccumulator.value.size}条记录.")
    }

    userNearestAccumulator.value
  }

  //用户看过电影的前TopN
  private var userLikedMap: Map[Int, Seq[Rating]] = _
  //与用户相似度最高的前n个用户
  private var nearestUser: mutable.Map[String, Double] = _

  private var ncfModel: ComputationGraph = _

  override def predict(query: Query): PredictedResult = {
    //1. 查看用户是否有相似度用户
    val userNearestRDD = nearestUser.filter(r => {
      r._1.indexOf(s",${query.user},") > -1
    })
    if (userNearestRDD.isEmpty) {
      //该用户没有最相似的用户列表
      logger.warn(s"该用户:${query.user}没有相似用户列表，无法生成推荐！")
      return PredictedResult(Array.empty)
    }

    //2. 获取推荐列表
    //用户相似度的Map
    val userNearestMap = userNearestRDD.filter(r => {
      //筛选当前用户的相似度列表
      r._1.indexOf(s",${query.user},") > -1
    }).map(r => {
      val uid = r._1.replace(s",${query.user},", "").replace(",", "")
      (uid.toInt, r._2)
    }).toSeq.sortBy(_._2).reverse.take(ap.numNearestUsers).toMap

    logger.info(s"${query.user}的相似用户列表的长度为：${userNearestMap.size}")

    //用户的已经观看列表
    val currentUserSawSet = userHasItem(query.user).map(_.item)
    logger.info(s"已经观看的列表长度为:${currentUserSawSet.size}")

    var logN = 0
    //筛选相近用户
    val result = userLikedMap.filter(r => userNearestMap.contains(r._1)).
      //生成用户的候选列表
      flatMap(_._2).
      //过滤已经看过的
      filter(r => {
      currentUserSawSet.nonEmpty && !currentUserSawSet.contains(r.item)
    }).
      map(r => {
        //r.rating
        //r.item
        //userNearestMap(r.user)
        (r.item, r.rating * userNearestMap(r.user))
      }).groupBy(_._1).map(r => {

      //新增NCF筛选
      val userV = newUserVector(query.user)
      val itemV = newItemVector(r._1)
      //生成特征向量
      val userArray = new Array[Float](userV.size)
      for (idx <- 0 until userV.size) {
        userArray(idx) = userV(idx).toFloat
      }
      val itemArray = new Array[Float](itemV.size)
      for (idx <- 0 until itemV.size) {
        itemArray(idx) = itemV(idx).toFloat
      }


      val vU = Nd4j.create(userArray).reshape(1, userArray.length)

      val vI = Nd4j.create(itemArray).reshape(1, itemArray.length)

      val start = System.currentTimeMillis()
      val vs: Array[INDArray] = ncfModel.output(vU, vI)
      logger.info("计算所需时间：" + (System.currentTimeMillis() - start))

      val sc = vs(0).getFloat(1)
      if (logN < 2) {
        logger.info(s"itemID:${r._1},正样例率:$sc,负样例率:${vs(0).getFloat(0)}")
        logN += 1
      }

      val itemid = r._1
      val scores = r._2.map(_._2).sum * sc
      (itemid, scores)
    })
    logger.info(s"生成的推荐列表的长度:${result.size}")
    val sum: Double = result.values.sum
    if (sum == 0) return PredictedResult(Array.empty)

    val weight = 1.0
    val returnResult = result.map(r => {
      ItemScore(r._1, r._2 / sum * weight)
    }).toArray.sortBy(_.score).reverse.take(query.num)

    //排序，返回结果
    PredictedResult(returnResult)
  }


}
