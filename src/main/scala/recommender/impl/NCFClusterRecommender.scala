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
import org.deeplearning4j.nn.conf.layers.{DenseLayer, EmbeddingLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}
import recommender._
import recommender.tools.{Correlation, NearestUserAccumulator}

import scala.collection.mutable
import scala.util.Random


case class NCFClusterParams(
                             oneHot: Boolean = false,
                             method: String = "Cosine",
                             k: Int = 4,
                             maxIterations: Int = 20,
                             numNearestUsers: Int = 60,
                             numUserLikeMovies: Int = 100) extends Params {
  override def getName(): String = {
    this.getClass.getSimpleName.replace("Params", "") + s"_$method"
  }

  override def toString: String = s"聚类参数{oneHot:$oneHot,method:$method,k:$k,maxIterations:$maxIterations,numNearestUsers:$numNearestUsers,numUserLikeMovies:$numUserLikeMovies}\r\n"
}


class NCFClusterRecommender(ap: NCFClusterParams) extends Recommender {
  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def getParams: Params = ap


  //训练集中用户所拥有item
  private var userHasItem: Map[Int, Seq[Rating]] = _

  //每个用户所有的物品=训练集中的物品+测试集中的物品
  private var allUserItemSet: Map[Int, Set[Int]] = _

  //用户的评分向量
  private var afterClusterRDD: Array[(Int, (Int, linalg.Vector))] = _

  //使用单独数字输入时使用。
  private var dimUserID: Int = _
  private var dimItemID: Int = _


  override def prepare(data: Seq[Rating]): PreparedData = {

    //val hitFile = Paths.get("spark-warehouse", s"hitRecord_${new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss").format(new Date)}.txt").toFile()
    //fw = new FileWriter(hitFile)

    //用于生成负样例测试数据
    allUserItemSet = data.groupBy(_.user).map(r => {
      val userId = r._1
      //
      val itemSet = r._2.map(_.item).toSet
      (userId, itemSet)
    })
    if (ap.oneHot) {
      //获取最大userID和itemID
      dimUserID = data.map(_.user).max + 1
      dimItemID = data.map(_.item).max + 1
    }

    //数据分割前
    new PreparedData(data)
  }

  private var newItemVector: collection.Map[Int, linalg.Vector] = _
  private var newUserVector: collection.Map[Int, linalg.Vector] = _

  private def initData(data: TrainingData): Unit = {
    //1.获取训练集中每个用户的观看列表
    userHasItem = data.ratings.groupBy(_.user)
    //2.生成用户喜欢的电影
    userLikedMap = userLikedItems()


    //3.实现新的统计方法
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
    newUserVector = userVectors.toMap

    if (!ap.oneHot) {
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


      newItemVector = itemVectors.toMap
    }

  }

  private def initNCF(data: TrainingData): Unit = {
    /** -------------------神经网络--------------------- **/
    val userVS = newUserVector.head._2.size
    val itemVS = newItemVector.head._2.size
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

      .addLayer("out", new OutputLayer.Builder(LossFunction.XENT)
        .nIn(unionSize + unionSize).activation(Activation.SIGMOID)
        .nOut(1).build(), "ncf")
      .setOutputs("out")
      .build()


    ncfModel = new ComputationGraph(computationGraphConf)
    ncfModel.init()


    val size = data.ratings.size

    //2.正样本数据
    logger.info("正样本数据")
    val positiveData = data.ratings.map(r => {
      //构建特征数据
      val userV = newUserVector(r.user)
      val itemV = newItemVector(r.item)

      (Random.nextInt(size), userV, itemV, 1F)
    })

    //3.负样本数据
    logger.info("负样本数据")

    val trainingItemSet = data.ratings.map(_.item).distinct.toSet
    val negativeData = userHasItem.flatMap(r => {
      //当前用户拥有的物品
      val userHadSet = r._2.map(_.item).distinct.toSet
      //未分割前的所有物品
      val userHadAllSet = allUserItemSet(r._1)

      //用户测试集中的物品
      val userTestSet = userHadAllSet.diff(userHadSet)
      val negativeSet = trainingItemSet.diff(userHadSet).diff(userTestSet)

      //保证负样本的数量，和正样本数量一致
      val nSet = Random.shuffle(negativeSet).take(userHadSet.size)

      val userV = newUserVector(r._1)
      nSet.map(itemID => {
        val itemV = newItemVector(itemID)
        (Random.nextInt(size), userV, itemV, 0F)
      })
    }).toSeq

    //4.数据打散
    logger.info("数据打散")
    val trainTempData = new Array[(Int, linalg.Vector, linalg.Vector, Float)](positiveData.size + negativeData.size)
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
    val userInEmbedding: INDArray = Nd4j.create(finalData.length, userVS)
    val itemInEmbedding: INDArray = Nd4j.create(finalData.length, itemVS)
    val outLabels: INDArray = Nd4j.create(finalData.length, 1)

    finalData.zipWithIndex.foreach({ case ((userIDVector, itemIDVector, label), index) =>
      userInEmbedding.putRow(index, Nd4j.create(userIDVector.toArray))
      itemInEmbedding.putRow(index, Nd4j.create(itemIDVector.toArray))

      outLabels.putScalar(Array[Int](index, 0), label)
    })

    ncfModel.setInputs(userInEmbedding, itemInEmbedding)
    ncfModel.setLabels(outLabels)

    ncfModel.fit()

    logger.info("训练模型完成")
  }

  override def train(data: TrainingData): Unit = {

    require(Set("cosine", "improvedpearson", "pearson").contains(ap.method.toLowerCase()))
    require(data.ratings.nonEmpty, "训练数据不能为空！")

    initData(data)

    //3.初始化网络
    if (ap.oneHot) {
      initNCFOneHot(data)
    } else {
      initNCF(data)
    }

    clusterUsers()

  }


  private def clusterUsers(): Unit = {

    val sparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()

    logger.info("正在对用户评分向量进行聚类，需要些时间...")

    val userVectorsRDD = sparkSession.sparkContext.parallelize(newUserVector.toSeq)

    val bkm = new BisectingKMeans().setK(ap.k).setMaxIterations(ap.maxIterations)
    val model = bkm.run(userVectorsRDD.map(_._2))

    //1.聚类用户评分向量(族ID,评分向量),根据聚类计算用户之间的相似度使用
    afterClusterRDD = userVectorsRDD.map(r => {
      (model.predict(r._2), r)
    }).collect()

    sparkSession.close()

    //2.根据用户评分向量生成用户最邻近用户的列表
    logger.info("计算用户邻近的相似用户中....")
    nearestUser = userNearestTopN()
  }

  private def initNCFOneHot(data: TrainingData): Unit = {
    /** -------------------神经网络--------------------- **/
    val userVS = 10
    val itemVS = 10
    val mfDim = 10
    val mlpDim = 32
    val unionSize = userVS + itemVS


    //1.配置网络结构
    val computationGraphConf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .activation(Activation.RELU)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new org.nd4j.linalg.learning.config.AdaDelta())
      .l2(1e-3)
      .graphBuilder()
      .addInputs("userGMFInput", "itemGMFInput", "userMLPInput", "itemMLPInput")
      // GMF
      .addLayer("userGMFLayer", new EmbeddingLayer.Builder()
      .nIn(dimUserID)
      .nOut(mfDim)
      .activation(Activation.IDENTITY)
      .build(), "userGMFInput")

      .addLayer("itemGMFLayer", new EmbeddingLayer.Builder()
        .nIn(dimItemID)
        .nOut(mfDim)
        .activation(Activation.IDENTITY)
        .build(), "itemGMFInput")

      .addLayer("GML", new ElementWiseMultiplicationLayer.Builder()
        .nIn(unionSize)
        .nOut(unionSize)
        .activation(Activation.IDENTITY)
        .build(), "userGMFLayer", "itemGMFLayer")

      //MLP
      .addLayer("userMLPLayer", new EmbeddingLayer.Builder()
      .nIn(dimUserID)
      .nOut(mlpDim)
      .activation(Activation.IDENTITY)
      .build(), "userMLPInput")

      .addLayer("itemMLPLayer", new EmbeddingLayer.Builder()
        .nIn(dimItemID)
        .nOut(mlpDim)
        .activation(Activation.IDENTITY)
        .build(), "itemMLPInput")
      .addVertex("merge", new MergeVertex(), "userMLPLayer", "itemMLPLayer")

      .addLayer("MLP4", new DenseLayer.Builder().nIn(2 * mlpDim).nOut(4 * unionSize).build(), "merge")
      .addLayer("MLP2", new DenseLayer.Builder().nIn(4 * unionSize).nOut(2 * unionSize).build(), "MLP4")
      .addLayer("MLP1", new DenseLayer.Builder().nIn(2 * unionSize).nOut(unionSize).build(), "MLP2")
      .addVertex("ncf", new MergeVertex(), "GML", "MLP1")
      .addLayer("out", new OutputLayer.Builder(LossFunction.XENT).nIn(unionSize + unionSize).activation(Activation.SIGMOID).nOut(1).build(), "ncf")
      .setOutputs("out")
      .build()


    ncfModel = new ComputationGraph(computationGraphConf)
    ncfModel.init()


    val size = data.ratings.size

    //2.正样本数据
    logger.info("正样本数据")
    val positiveData = data.ratings.map(r => {
      //构建特征数据
      val userV = r.user
      val itemV = r.item

      (Random.nextInt(size),
        userV,
        itemV,
        1)
    })

    //3.负样本数据
    logger.info("负样本数据")
    val trainingItemSet = data.ratings.map(_.item).distinct.toSet
    val negativeData = userHasItem.flatMap(r => {
      //当前用户拥有的物品
      val userHadSet = r._2.map(_.item).distinct.toSet
      //未分割前的所有物品
      val userHadAllSet = allUserItemSet(r._1)
      //用户测试集中的物品
      val userTestSet = userHadAllSet.diff(userHadSet)

      val negativeSet = trainingItemSet.diff(userHadSet).diff(userTestSet)
      //保证负样本的数量，和正样本数量一致 4=kFold-1
      val nSet = Random.shuffle(negativeSet).take(userHadSet.size)

      val userV = r._1
      nSet.map(itemID => {
        val itemV = itemID
        (Random.nextInt(size), userV, itemV, 0)
      })
    }).toSeq

    //4.数据打散
    logger.info("数据打散")
    val trainTempData = new Array[(Int, Int, Int, Int)](positiveData.size + negativeData.size)
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

    //构建数据

    logger.info("开始训练模型...")

    val userInEmbedding: INDArray = Nd4j.create(finalData.length, 1)
    val itemInEmbedding: INDArray = Nd4j.create(finalData.length, 1)
    val outLabels: INDArray = Nd4j.create(finalData.length, 1)


    finalData.zipWithIndex.foreach({ case ((userID, itemID, label), index) =>
      userInEmbedding.putScalar(Array[Int](index, 0), userID)
      itemInEmbedding.putScalar(Array[Int](index, 0), itemID)
      outLabels.putScalar(Array[Int](index, 0), label)
    })

    ncfModel.setInputs(userInEmbedding, itemInEmbedding, userInEmbedding, itemInEmbedding)
    ncfModel.setLabels(outLabels)

    ncfModel.fit()

    logger.info("训练模型完成")
  }

  def userLikedItems(): Map[Int, Seq[Rating]] = {

    val userMean = userHasItem.map(r => {
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


  //private var fw: FileWriter = _

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

    //2.用户的已经观看列表
    val currentUserSawSet = userHasItem(query.user).map(_.item)


    //3. 获取推荐列表
    //用户相似度的Map
    val userNearestMap = userNearestRDD.filter(r => {
      //筛选当前用户的相似度列表
      r._1.indexOf(s",${query.user},") > -1
    }).map(r => {
      val uid = r._1.replace(s",${query.user},", "").replace(",", "")
      (uid.toInt, r._2)
    }).toSeq.sortBy(_._2).reverse.take(ap.numNearestUsers).toMap


    val candidateItems = userLikedMap
      .filter(r => userNearestMap.contains(r._1))
      //生成用户的候选列表
      .flatMap(_._2)
      //过滤已经看过的
      .filter(r => {
      currentUserSawSet.nonEmpty && !currentUserSawSet.contains(r.item)
    })
      // 计算评分与用户之间相似度的乘积
      .map(r => {
      (r.item, r.rating * userNearestMap(r.user))
    }).groupBy(_._1).map(r => {
      val scores = r._2.map(_._2).sum
      (r._1, scores)
    }).toSeq.sortBy(_._2).reverse.take(400)

    //4.批量预测
    logger.info(s"候选列表长度为：${candidateItems.size}")

    val result = if (ap.oneHot) {

      val userInputs = Nd4j.create(candidateItems.size, 1)
      val itemInputs = Nd4j.create(candidateItems.size, 1)

      val indexToItem = candidateItems.zipWithIndex.map({ case ((itemID, _), index) =>
        userInputs.putScalar(Array[Int](index, 0), query.user)
        itemInputs.putScalar(Array[Int](index, 0), itemID)
        (index, itemID)
      }).toMap

      val vs: Array[INDArray] = ncfModel.output(userInputs, itemInputs, userInputs, itemInputs)
      (0 until vs(0).length().toInt).map(idx => {
        val score = vs(0).getFloat(idx)
        (indexToItem(idx), score)
      })

    } else {
      val userV = newUserVector(query.user)
      val userInputs = Nd4j.create(candidateItems.size, userV.size)
      val itemSize = newItemVector(1).size
      val itemInputs = Nd4j.create(candidateItems.size, itemSize)

      val indexToItem = candidateItems.zipWithIndex.map({ case ((itemID, _), index) =>
        userInputs.putRow(index, Nd4j.create(userV.toArray))
        itemInputs.putRow(index, Nd4j.create(newItemVector(itemID).toArray))
        (index, itemID)
      }).toMap

      val vs: Array[INDArray] = ncfModel.output(userInputs, itemInputs)
      (0 until vs(0).length().toInt).map(idx => {
        val score = vs(0).getFloat(idx)
        (indexToItem(idx), score)
      })
    }


    logger.info(s"生成的推荐列表的长度:${result.size}")
    val sum: Double = result.map(_._2).sum
    if (sum == 0) return PredictedResult(Array.empty)

    val weight = 1.0
    val returnResult = result.map(r => {
      ItemScore(r._1, r._2 / sum * weight)
    }).toArray.sortBy(_.score).reverse.take(query.num)


    /**
      * 调试信息
      * 用于判断聚类产生候选物品的效果。
      ***/


    /*//用户所有的物品，包含训练集和测试集中的物品
    val myAllItems = allUserItemSet(query.user)
    //测试集物品=所有物品-当前用户在训练集中的物品
    val tSet = myAllItems.diff(currentUserSawSet.toSet)

    //前400
    val h400 = candidateItems.take(400)
    val hit400 = h400.map(_._1).toSet.intersect(tSet)
    logger.info(s"前400中，含有测试集数据的个数:${hit400.size}")


    //筛选过后包含测试集的数目
    val iSet = result.map(_._1).toSet.intersect(tSet)
    //推荐列表中的物品
    val rSet = returnResult.map(_.item).toSet.intersect(tSet)
    logger.info(s"测试集中物品数目：${tSet.size},筛选过后包含测试集中命中数目:${iSet.size}，推荐列表中命中数目：${rSet.size}")*/

    //fw.append(s"${query.user},${hit100.size},${hit200.size},${hit300.size},${hit400.size},${iSet.size},${rSet.size}\r\n")
    //fw.flush()


    //排序，返回结果
    PredictedResult(returnResult)
  }

  override def finalize(): Unit = {
    //fw.close()
    super.finalize()
  }

}
