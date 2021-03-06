package recommender.impl

import org.apache.spark.mllib.clustering.{BisectingKMeans, GaussianMixture, KMeans}
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.slf4j.{Logger, LoggerFactory}
import recommender._
import recommender.tools.{Correlation, NearestUserAccumulator}

import scala.collection.mutable

/**
  * Author:IceS
  * Date:2019-08-10 23:27:26
  * Description:
  * 这个类的重点是测试聚类算法搭配各种相似度算法的效果。
  */
case class ClusterParams(
                          CM: String = "BisectingKMeans", //聚类算法名
                          CC: Int = 2, //类簇数量
                          MI: Int = 20, //maxIterations最大迭代次数
                          SM: String = "Cosine", //相似度计算方法
                          K: Int = 5, //近邻的数目
                          L: Int = 5//用户喜欢的电影列表长度
                        ) extends Params {
  override def getName(): String = {
    this.getClass.getSimpleName.replace("Params", "") + s"_$CM"
  }

  override def toString: String = s"聚类{聚类方法：$CM,簇心数量:$CC,最大迭代次数:$MI,相似度方法:$SM,近邻的数目:$K,用户喜欢的电影列表长度:$L}\r\n"
}


class ClusterRecommender(ap: ClusterParams) extends Recommender {
  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def getParams: Params = ap

  //用户的评分向量
  private var afterClusterRDD: Seq[(Int, (Int, linalg.Vector))] = _
  //训练集中用户所拥有item
  private var userHadItem: Map[Int, Seq[Rating]] = _


  override def prepare(data: Seq[Rating]): PreparedData = {

    require(data.nonEmpty, "原始数据不能为空！")

    new PreparedData(data)
  }

  override def train(data: TrainingData): Unit = {

    require(data.ratings.nonEmpty, "训练数据不能为空！")
    require(Set("BisectingKMeans", "K-means", "GaussianMixture").contains(ap.CM), "聚类方法必须在是：[BisectingKMeans,K-means,GaussianMixture]其中之一!")

    //1.获取训练集中每个用户的观看列表
    userHadItem = data.ratings.groupBy(_.user)

    //2.根据用户评分中，各个评分等级的数量进行构造
    val userVectors = userHadItem.map(r => {
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

      (uid, Vectors.dense(c1 , c2 , c3 , c4 , c5 ))
    }).toSeq


    /** -------------------对用户评分向量进行聚类--------------------- **/
    logger.info("正在对用户评分向量进行聚类，需要些时间...")
    //3.准备聚类
    val sparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()
    //并行读入数据
    val pData = sparkSession.sparkContext.parallelize(userVectors)
    val dTrain = pData.map(_._2)

    //选择聚类算法
    if (ap.CM.toLowerCase() == "BisectingKMeans".toLowerCase) {
      //二分k-means
      val bkm = new BisectingKMeans().setK(ap.CC).setMaxIterations(ap.MI)
      val model = bkm.run(dTrain)

      /*//调试信息
      //查看集合内偏差的误差平方和
      //数据集内偏差的误差平方和：47.21731266112625
      val WSSSE = model.computeCost(dtrain)
      logger.info(s"数据集内偏差的误差平方和：$WSSSE")
      throw new Exception("集群偏差测试")*/
      //对新用户特征向量进行簇分类
      afterClusterRDD = userVectors.map(r => {
        (model.predict(r._2), r)
      })
    } else if (ap.CM.toLowerCase() == "K-means".toLowerCase) {
      //k-means II
      val clusters = KMeans.train(dTrain, ap.CC, ap.MI)

      /*//调试信息
      //查看集合内偏差的误差平方和
      val WSSSE = clusters.computeCost(dtrain)
      logger.info(s"数据集内偏差的误差平方和：$WSSSE")
      Thread.sleep(5000)*/

      //对新用户特征向量进行簇分类
      afterClusterRDD = userVectors.map(r => {
        (clusters.predict(r._2), r)
      })
    } else if (ap.CM.toLowerCase() == "GaussianMixture".toLowerCase()) {
      //高斯
      val gmm = new GaussianMixture().setK(ap.CC).run(dTrain)

     /* //调试信息
      //查看集合内偏差的误差平方和
      for (i <- 0 until gmm.k) {
        println("weight=%f\nmu=%s\nsigma=\n%s\n" format
          (gmm.weights(i), gmm.gaussians(i).mu, gmm.gaussians(i).sigma))
      }*/
      //对新用户特征向量进行簇分类
      afterClusterRDD = userVectors.map(r => {
        (gmm.predict(r._2), r)
      })
    }


    /** -------------------生成共享数据--------------------- **/

    sparkSession.close()

    //4.生成用户喜欢的电影
    userLikedMap = userLikedItems()
    //调试信息
    logger.info("训练集中的用户总数:" + userLikedMap.size)

    //5.根据用户评分向量生成用户最邻近用户的列表
    logger.info("计算用户邻近的相似用户中....")
    nearestUser = userNearestTopK()

  }


  def userLikedItems(): Map[Int, Seq[Rating]] = {
   userHadItem.map(r => {
      val userLikes: Seq[Rating] = r._2.toList.sortBy(_.rating).reverse.take(ap.L)
      (r._1, userLikes)
    })
  }

  def userNearestTopK(): mutable.Map[String, Double] = {
    //afterClusterRDD: RDD[(Int, (Int, linalg.Vector))]
    //                簇Index  (Uid    评分向量)

    //计算所有用户之间的相似度
    val userNearestAccumulator = new NearestUserAccumulator

    //1.考虑从族中进行计算相似度
    for (idx <- 0 until ap.CC) {

      val currentClusterUserVectors = afterClusterRDD.filter(_._1 == idx).map(_._2)
      logger.info(s"簇$idx 中用户数量为：${currentClusterUserVectors.length}")

      val uids = currentClusterUserVectors.sortBy(_._1)

      for {
        (u1, v1) <- uids
        (u2, v2) <- uids
        if u1 < u2
      } {

        val score = if (ap.SM.toLowerCase() == "cosine") {
          Correlation.getCosine(v1, v2)
        } else if (ap.SM.toLowerCase() == "improvedpearson") {
          Correlation.getImprovedPearson(v1, v2)
        } else if (ap.SM.toLowerCase() == "pearson") {
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
    }).toSeq.sortBy(_._2).reverse.take(ap.K).toMap

    //logger.info(s"${query.user}的相似用户列表的长度为：${userNearestMap.size}")

    //用户的已经观看列表
    val currentUserSawSet = userHadItem(query.user).map(_.item)
    logger.info(s"已经观看的列表长度为:${currentUserSawSet.size}")

    val result = userLikedMap.
      //在所有用户的喜欢列表中,查找当前用户的邻近用户相关的
      filter(r => userNearestMap.contains(r._1)).
      //提取出每个用户的喜欢的评分列表
      flatMap(_._2).
      //过滤已经看过的
      filter(r => {
      currentUserSawSet.nonEmpty && !currentUserSawSet.contains(r.item)
    }).
      map(r => {
        //评分乘以相似度
        //r.rating
        //r.item
        //userNearestMap(r.user)
        (r.item, r.rating * userNearestMap(r.user))
      }).groupBy(_._1).map(r => {
      val itemID = r._1
      val scores = r._2.map(_._2).sum
      //logger.info(s"累加的相似度：${scores},物品的评论数量:${itemCountSeq(r._1)}")
      (itemID, scores)
    })

    logger.info(s"生成候选物品列表的长度为：${result.size}")
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


