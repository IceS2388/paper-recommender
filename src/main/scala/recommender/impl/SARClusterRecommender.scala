package recommender.impl

import breeze.linalg.DenseMatrix
import org.apache.spark.mllib.clustering.{BisectingKMeans, GaussianMixture, KMeans}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.slf4j.{Logger, LoggerFactory}
import recommender._
import recommender.tools.{BiMap, Correlation, NearestUserAccumulator}

import scala.collection.mutable

case class SARClusterParams(
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

/**
  * Author:IceS
  * Date:2019-08-27 15:22:20
  * Description:
  * NONE
  */
class SARClusterRecommender(ap: SARClusterParams) extends Recommender {
  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def getParams: Params = ap


  override def prepare(data: Seq[Rating]): PrepairedData = {

    require(data.nonEmpty, "原始数据不能为空！")


    new PrepairedData(data)
  }

  // 用户的评分向量
  private var afterClusterRDD: Seq[(Int, (Int, linalg.Vector))] = _
  // 训练集中用户所拥有item
  private var userGroup: Map[Int, Seq[Rating]] = _



  private var newUserVector: collection.Map[Int, linalg.Vector] = _

  override def train(data: TrainingData): Unit = {

    require(data.ratings.nonEmpty, "训练数据不能为空！")
    require(Set("BisectingKMeans", "K-means", "GaussianMixture").contains(ap.clusterMethod), "聚类方法必须在是：[BisectingKMeans,K-means,GaussianMixture]其中之一!")

    // 把数据按照用户进行分组
    userGroup = data.ratings.groupBy(_.user)
    // 把数据按照物品分组,SAR中计算电影之间的相似度时使用
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

    newUserVector = userVectors.toMap


  }

  /**
    * 对用户向量进行聚类。
    **/
  private def clusterUsers(userVectors: Seq[(Int, linalg.Vector)]): Unit = {
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
  // 根据物品分组，用于创建物品特征
  private var itemsGroup: Map[Int, Set[Int]] = _

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
      }).toArray.sortBy(_._2).reverse.take(400).map(_._1).toSet



    //对应候选列表的索引
    val candidateItems2Index = BiMap.toIndex(candidateMovies)
    val sawItems2Index = BiMap.toIndex(currentUserSawSet)

    //1.生成用户关联矩阵
    val affinityMatrix = DenseMatrix.ones[Float](1, currentUserSawSet.size)

    //2.生成item2item的相似度矩阵
    val itemToItemMatrix: DenseMatrix[Float] = DenseMatrix.zeros[Float](currentUserSawSet.size, candidateMovies.size)

    //赋予矩阵相似度值,这个过程比较费时间
    for {
      sawID <- currentUserSawSet
      cID <- candidateMovies
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
