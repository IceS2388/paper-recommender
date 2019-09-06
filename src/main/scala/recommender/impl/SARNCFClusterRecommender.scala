package recommender.impl

import breeze.linalg.DenseMatrix
import org.apache.spark.mllib.clustering.{BisectingKMeans, GaussianMixture, KMeans}
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
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.slf4j.{Logger, LoggerFactory}
import recommender._
import recommender.tools.{BiMap, Correlation, NearestUserAccumulator}

import scala.collection.mutable
import scala.util.Random

/**
  * Author:IceS
  * Date:2019-09-03 12:08:26
  * Description:
  * NONE
  */
case class SARNCFClusterParams(
                                clusterMethod: String = "BisectingKMeans",
                                k: Int = 2,
                                maxIterations: Int = 20,
                                method: String = "Cosine",
                                numNearestUsers: Int = 5,
                                numUserLikeMovies: Int = 5) extends Params {
  override def getName(): String = {
    this.getClass.getSimpleName.replace("Params", "") + s"_$clusterMethod"
  }

  override def toString: String = s"聚类{聚类方法：$clusterMethod,簇心数量:$k,maxIterations:$maxIterations,相似度方法:$method,numNearestUsers:$numNearestUsers,numUserLikeMovies:$numUserLikeMovies}\r\n"
}

class SARNCFClusterRecommender(ap: SARNCFClusterParams) extends Recommender {
  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def getParams: Params = ap


  //每个用户所有的物品
  private var userAllItemSet: Map[Int, Set[Int]] = _

  override def prepare(data: Seq[Rating]): PrepairedData = {

    require(data.nonEmpty, "原始数据不能为空！")

    userAllItemSet = data.groupBy(_.user).map(r => {
      val userId = r._1
      //
      val itemSet = r._2.map(_.item).toSet
      (userId, itemSet)
    })

    new PrepairedData(data)
  }

  // 用户的评分向量
  private var afterClusterRDD: Seq[(Int, (Int, linalg.Vector))] = _
  // 训练集中用户所拥有item
  private var userGroup: Map[Int, Seq[Rating]] = _
  // 根据物品分组，用于创建物品特征
  private var itemsGroup: Map[Int, Set[Int]] = _

  private var newItemVector: collection.Map[Int, linalg.Vector] = _
  private var newUserVector: collection.Map[Int, linalg.Vector] = _

  override def train(data: TrainingData): Unit = {

    require(data.ratings.nonEmpty, "训练数据不能为空！")
    require(Set("BisectingKMeans", "K-means", "GaussianMixture").contains(ap.clusterMethod), "聚类方法必须在是：[BisectingKMeans,K-means,GaussianMixture]其中之一!")

    // 把数据按照用户进行分组
    userGroup = data.ratings.groupBy(_.user)
    // 把数据按照物品分组
    itemsGroup = data.ratings.groupBy(_.item).map(r => (r._1, r._2.map(_.user).toSet))
    // 初始化新的特征向量
    initVectors(data)

    /** -------------------对用户评分向量进行聚类--------------------- **/
    clusterUsers(newUserVector.toSeq)

    // 获取每个用户在观看记录中，评分最靠前的N个电影
    userLikedMap = userLikedItems()

    // 根据用户评分向量生成用户最邻近用户的列表
    logger.info("计算用户邻近的相似用户中....")
    nearestUser = userNearestNeighbours()

    // 初始化神经网络
    initalNCF(data)

  }

  private def initVectors(data: TrainingData): Unit = {
    //实现新的统计方法
    val userVectors = userGroup.map(r => {
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

  }

  private def initalNCF(data: TrainingData): Unit = {
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
      .addLayer("out", new OutputLayer.Builder(LossFunction.XENT).nIn(unionSize + unionSize).activation(Activation.SIGMOID).nOut(1).build(), "ncf")
      .setOutputs("out")
      .build()


    ncfModel = new ComputationGraph(computationGraphConf)
    ncfModel.init()


    val size = data.ratings.size

    //2.正样本数据
    logger.info("正样本数据")
    val positiveData: Seq[(Int, INDArray, INDArray, INDArray)] = data.ratings.map(r => {
      //构建特征数据
      val userV: linalg.Vector = newUserVector(r.user)
      val itemV: linalg.Vector = newItemVector(r.item)

      //生成标签
      val la = new Array[Double](1)
      la(0) = 1


      (Random.nextInt(size),
        Nd4j.create(userV.toArray).reshape(1, userV.size),
        Nd4j.create(itemV.toArray).reshape(1, itemV.size),
        Nd4j.create(la).reshape(1, 1))
    })

    //3.负样本数据
    logger.info("负样本数据")
    val trainingItemSet = data.ratings.map(_.item).distinct.toSet
    val negativeData: Seq[(Int, INDArray, INDArray, INDArray)] = userGroup.flatMap(r => {
      //当前用户拥有的物品
      val userHadSet = r._2.map(_.item).distinct.toSet
      //未分割前的所有物品
      val userHadAllSet = userAllItemSet(r._1)
      //用户测试集中的物品
      val userTestSet = userHadAllSet.diff(userHadSet)

      val negativeSet = trainingItemSet.diff(userHadSet).diff(userTestSet)
      //保证负样本的数量，和正样本数量一致 4=kFold-1
      val nSet = Random.shuffle(negativeSet).take(userHadSet.size)

      val userV = newUserVector(r._1)

      nSet.map(itemID => {
        val itemV = newItemVector(itemID)

        //生成标签
        val la = new Array[Double](1)
        la(0) = 0

        (Random.nextInt(size),
          Nd4j.create(userV.toArray).reshape(1, userV.size),
          Nd4j.create(itemV.toArray).reshape(1, itemV.size),
          Nd4j.create(la).reshape(1, 1))
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
      if (logIdx % 1000 == 0) {
        logger.info(s"index:$logIdx")
      }

      ncfModel.fit(Array(r._1, r._2), Array(r._3))
      logIdx += 1
    })
    logger.info("训练模型完成")
  }

  /**
    * 对用户向量进行聚类。
    **/
  private def clusterUsers(userVectors: Seq[(Int, linalg.Vector)]) = {
    logger.info("正在对用户评分向量进行聚类，需要些时间...")
    //3.准备聚类
    val sparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()

    val d = sparkSession.sparkContext.parallelize(userVectors)
    val dtrain = d.map(_._2)

    //选择聚类算法
    if (ap.clusterMethod.toLowerCase() == "BisectingKMeans".toLowerCase) {
      val bkm = new BisectingKMeans().setK(ap.k).setMaxIterations(ap.maxIterations)
      val model = bkm.run(dtrain)


      afterClusterRDD = userVectors.map(r => {
        (model.predict(r._2), r)
      })
    } else if (ap.clusterMethod.toLowerCase() == "K-means".toLowerCase) {
      val clusters = KMeans.train(dtrain, ap.k, ap.maxIterations)


      afterClusterRDD = userVectors.map(r => {
        (clusters.predict(r._2), r)
      })
    } else if (ap.clusterMethod.toLowerCase() == "GaussianMixture".toLowerCase()) {
      val gmm = new GaussianMixture().setK(ap.k).run(dtrain)

      /*//调试信息
      //查看集合内偏差的误差平方和
      for (i <- 0 until gmm.k) {
        println("weight=%f\nmu=%s\nsigma=\n%s\n" format
          (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
      }
      */

      afterClusterRDD = userVectors.map(r => {
        (gmm.predict(r._2), r)
      })
    }

    sparkSession.close()
  }

  private def userLikedItems(): Map[Int, Seq[Rating]] = {


    val groupRDD = userGroup
    //1.计算用户的平均分
    val userMean = groupRDD.map(r => {
      val userLikes: Seq[Rating] = r._2.toList.sortBy(_.rating).reverse.take(ap.numUserLikeMovies)
      (r._1, userLikes)
    })
    userMean
  }

  private def userNearestNeighbours(): mutable.Map[String, Double] = {
    //afterClusterRDD: RDD[(Int, (Int, linalg.Vector))]
    //                簇Index  Uid    评分向量

    //计算所有用户之间的相似度
    val userNearestAccumulator = new NearestUserAccumulator

    //1.考虑从族中进行计算相似度
    for (idx <- 0 until ap.k) {

      val curUsersVectors = afterClusterRDD.filter(_._1 == idx).map(_._2)
      logger.info(s"簇$idx 中用户数量为：${curUsersVectors.length}")

      val uids = curUsersVectors.sortBy(_._1)

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

  // 用户看过电影的前TopN
  private var userLikedMap: Map[Int, Seq[Rating]] = _
  // 与用户相似度最高的前n个用户
  private var nearestUser: mutable.Map[String, Double] = _
  // NCF模型
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
    val userNearestMap = userNearestRDD
      .map(r => {
        val uid = r._1.replace(s",${query.user},", "").replace(",", "")
        //近邻ID，相似度
        (uid.toInt, r._2)
      }).toSeq.sortBy(_._2).reverse.take(ap.numNearestUsers).toMap


    //3. 用户的已经观看列表
    val currentUserSawSet = userGroup(query.user).map(_.item).toSet
    logger.info(s"已经观看的电影列表长度为:${currentUserSawSet.size}")

    //计算候选列表中，用户未观看的推荐度最高的前400个电影
    val candidateMovies = userLikedMap
      //在所有用户的喜欢列表中,查找当前用户的邻近用户相关的
      .filter(r => userNearestMap.contains(r._1))
      //提取出每个用户的喜欢的评分电影列表
      .flatMap(_._2)
      //过滤用户已经看过的电影
      .filter(r => {
      currentUserSawSet.nonEmpty && !currentUserSawSet.contains(r.item)
    })
      //计算每部电影的推荐程度=评分*相似度
      .map(r => {
      //r.rating
      //r.item
      //userNearestMap(r.user)
      (r.item, r.rating * userNearestMap(r.user))
    })
      //聚合同一item的权重
      .groupBy(_._1)
      .map(r => {
        val itemID = r._1
        val scores = r._2.map(_._2).sum
        (itemID, scores)
      }).toArray.sortBy(_._2).reverse.take(400)

    //实现神经网络筛选
    val candidateMoviesSet = candidateMovies.filter(r => {

      val userV = newUserVector(query.user)
      val itemV = newItemVector(r._1)

      //生成特征向量
      val vU = Nd4j.create(userV.toArray).reshape(1, userV.size)
      val vI = Nd4j.create(itemV.toArray).reshape(1, itemV.size)

      val vs: Array[INDArray] = ncfModel.output(vU, vI)

      val sc: Double = vs(0).getDouble(0L)

      sc > 0.6
    }).map(_._1).toSet

    //对应候选列表的索引
    val candidateItems2Index = BiMap.toIndex(candidateMoviesSet)
    val sawItems2Index = BiMap.toIndex(currentUserSawSet)

    //1.生成用户关联矩阵
    val affinityMatrix = DenseMatrix.ones[Float](1, currentUserSawSet.size)

    //2.生成item2item的相似度矩阵
    val itemToItemMatrix: DenseMatrix[Float] = DenseMatrix.zeros[Float](currentUserSawSet.size, candidateMoviesSet.size)

    //赋予矩阵相似度值,这个过程比较费时间
    for {
      sawID <- currentUserSawSet
      cID <- candidateMoviesSet
    } {
      //计算相关系数值
      val s = Correlation.getJaccardSAR(1, sawID, cID, itemsGroup).toFloat

      val index1 = sawItems2Index(sawID).toInt
      val index2 = candidateItems2Index(cID).toInt
      itemToItemMatrix.update(index1, index2, s)
    }
    //生成推荐矩阵
    val resultMatrix = affinityMatrix * itemToItemMatrix
    val row = resultMatrix(0, ::)

    val indexToItem: BiMap[Long, Int] = candidateItems2Index.inverse
    //索引变成itemID
    val result = indexToItem.toSeq.map(r => { //r._1:index,r._2:权重
      (r._2, row.apply(r._1.toInt))
    }).sortBy(_._2).reverse.take(query.num)


    // 返回权重归一化
    val sum: Double = result.map(_._2).sum
    if (sum == 0) return PredictedResult(Array.empty)

    val weight = 1.0
    val returnResult = result.map(r => {
      ItemScore(r._1, r._2 / sum * weight)
    }).toArray

    //排序，返回结果
    PredictedResult(returnResult)
  }


}
