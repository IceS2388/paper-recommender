import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

import scala.collection. mutable

/**
  * Author:IceS
  * Date:2019-08-10 23:27:26
  * Description:
  * NONE
  */
case class ClusterParams(
                          userThreashold: Int = 20,
                          itemThreashold: Int = 2,
                          method: String = "Cosine",
                          k: Int = 5,
                          maxIterations: Int = 20,
                          numNearestUsers: Int = 60,
                          numUserLikeMovies: Int = 100) extends Params {
  override def getName(): String = {
    this.getClass.getSimpleName.replace("Params", "") + s"_$method"
  }

  override def toString: String = s"聚类参数{userThreashold:$userThreashold,itemThreashold:$itemThreashold,method:$method,k:$k,maxIterations:$maxIterations,numNearestUsers:$numNearestUsers,numUserLikeMovies:$numUserLikeMovies}\r\n"
}


class ClusterRecommender(ap: ClusterParams) extends Recommender {
  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def getParams: Params = ap

  //物品的购买频率
  private var itemCountSeq: scala.collection.Map[Int, Long] = _
  //用户的评分向量
  private var afterClusterRDD: Array[(Int, (Int, linalg.Vector))] = _
  //训练集中用户所拥有item
  private var userHasItem: Map[Int, Seq[Rating]] = _

  override def prepare(data: Seq[Rating]): PrepairedData = {

    require(data.nonEmpty, "原始数据不能为空！")

    val fields: Seq[StructField] = List(
      StructField("uid", IntegerType, nullable = false),
      StructField("iid", IntegerType, nullable = false),
      StructField("rating", DoubleType, nullable = false),
      StructField("tt", LongType, nullable = false)
    )
    val schema = StructType(fields)
    val sparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()

    val rowRDD = sparkSession.sparkContext.parallelize(data.map(r => Row(r.user, r.item, r.rating, r.timestamp)))

    val ratingsDF = sparkSession.createDataFrame(rowRDD, schema)
    ratingsDF.createOrReplaceTempView("ratings")

    /**
      * 思路：
      * 1.根据Item进行筛选，过滤访问次数小于阀值的物品。
      * 2.根据User的访问数量必须大于20个。
      * 3.用户的评分列表中，不能仅仅包含评分，同样也需要包含被使用的次数。
      **/

    //1.过滤物品
    val itemDF = sparkSession.sql("SELECT iid,COUNT(rating) AS num,SUM(rating) AS total  FROM ratings GROUP BY iid ")
    itemDF.createOrReplaceTempView("istatic")
    val itemFilterRatingsDF = sparkSession.sql(
      s"""
         |SELECT uid,iid,rating,tt
         |FROM ratings
         |WHERE iid NOT IN (SELECT iid FROM istatic WHERE num < ${ap.itemThreashold})
      """.stripMargin)
    itemFilterRatingsDF.createOrReplaceTempView("ifratings")

    //2.过滤用户
    val userDF = sparkSession.sql("SELECT uid,COUNT(rating) AS num,SUM(rating) AS total  FROM ifratings GROUP BY uid ")
    userDF.createOrReplaceTempView("ustatic")
    val userFilterRatingsDF = sparkSession.sql(
      s"""
         |SELECT uid,iid,rating,tt
         |FROM ifratings
         |WHERE uid NOT IN (SELECT uid FROM ustatic WHERE num < ${ap.userThreashold})
      """.stripMargin)
    userFilterRatingsDF.createOrReplaceTempView("uiratings")
    userFilterRatingsDF.cache()

    //3.生成物品受欢迎的程度
    val itemCount = sparkSession.sql(
      """
        |SELECT iid,COUNT(rating) AS num FROM uiratings GROUP BY iid
      """.stripMargin)
    //itemCount.printSchema()
    //Thread.sleep(1000)


    //4.生成用户的评分向量
    val userVectorsDF = sparkSession.sql(
      """
        |SELECT uid,
        |COUNT(CASE WHEN rating=0.5 THEN 1 END) AS c1,
        |COUNT(CASE WHEN rating=1.0 THEN 1 END) AS c2,
        |COUNT(CASE WHEN rating=1.5 THEN 1 END) AS c3,
        |COUNT(CASE WHEN rating=2.0 THEN 1 END) AS c4,
        |COUNT(CASE WHEN rating=2.5 THEN 1 END) AS c5,
        |COUNT(CASE WHEN rating=3.0 THEN 1 END) AS c6,
        |COUNT(CASE WHEN rating=3.5 THEN 1 END) AS c7,
        |COUNT(CASE WHEN rating=4.0 THEN 1 END) AS c8,
        |COUNT(CASE WHEN rating=4.5 THEN 1 END) AS c9,
        |COUNT(CASE WHEN rating=5.0 THEN 1 END) AS c10,
        |COUNT(rating) AS total
        |FROM uiratings
        |GROUP BY uid
        |ORDER BY total ASC
      """.stripMargin)
    userVectorsDF.createOrReplaceTempView("uv")
    userVectorsDF.show()
    /*高级版本再使用
    val itemVectorsDF = sparkSession.sql(
      """
        |SELECT iid,
        |COUNT(CASE WHEN rating=0.5 THEN 1 END) AS c1,
        |COUNT(CASE WHEN rating=1.0 THEN 1 END) AS c2,
        |COUNT(CASE WHEN rating=1.5 THEN 1 END) AS c3,
        |COUNT(CASE WHEN rating=2.0 THEN 1 END) AS c4,
        |COUNT(CASE WHEN rating=2.5 THEN 1 END) AS c5,
        |COUNT(CASE WHEN rating=3.0 THEN 1 END) AS c6,
        |COUNT(CASE WHEN rating=3.5 THEN 1 END) AS c7,
        |COUNT(CASE WHEN rating=4.0 THEN 1 END) AS c8,
        |COUNT(CASE WHEN rating=4.5 THEN 1 END) AS c9,
        |COUNT(CASE WHEN rating=5.0 THEN 1 END) AS c10,
        |COUNT(rating) AS total
        |FROM ratings
        |GROUP BY iid
        |HAVING total > 1
        |ORDER BY total ASC
      """.stripMargin)
    itemVectorsDF.createOrReplaceTempView("iv")
    //itemVectorsDF.show(4000)*/

    //5.把数据生成Seq
    itemCountSeq = itemCount.rdd.map(r =>
      (r.getAs[Int]("iid"), r.getAs[Long]("num"))).collectAsMap()


    val ratingsSeq = userFilterRatingsDF.rdd.map(r => {
      //uid,iid,rating,tt
      Rating(r.getAs[Int]("uid"), r.getAs[Int]("iid"), r.getAs[Double]("rating"), r.getAs[Long]("tt"))
    }).collect()

    logger.info("正在对用户评分向量进行聚类，需要些时间...")
    //6.准备聚类
    userVectorsDF.printSchema()
    val userVectorsRDD = userVectorsDF.rdd.map(r => {
      (r.getAs[Int]("uid"), Vectors.dense(
        r.getAs[Long]("c1"),
        r.getAs[Long]("c2"),
        r.getAs[Long]("c3"),
        r.getAs[Long]("c4"),
        r.getAs[Long]("c5"),
        r.getAs[Long]("c6"),
        r.getAs[Long]("c7"),
        r.getAs[Long]("c8"),
        r.getAs[Long]("c9"),
        r.getAs[Long]("c10"),
        r.getAs[Long]("total")
      ))
    })
    //7.聚类
    val bkm = new BisectingKMeans().setK(ap.k).setMaxIterations(ap.maxIterations)
    val model = bkm.run(userVectorsRDD.map(_._2))

    //调试信息
    model.clusterCenters.foreach(println)

    //4.聚类用户评分向量(族ID,评分向量)
    afterClusterRDD = userVectorsRDD.map(r => {
      (model.predict(r._2), r)
    }).collect()

    sparkSession.close()
    new PrepairedData(ratingsSeq)
  }

  override def train(data: TrainingData): Unit = {

    require(data.ratings.nonEmpty, "测试评论数据不能为空！")

    /**
      * 思路：
      *  1.生成该族所有所有用户距离中心点距离的倒数系数，作为权重系数。
      *  2.把族中每个用户评分的Item和Rating，然后，同时对rating*权重系数，最后，累加获得族中用户推荐列表。
      *  3.存储该推荐列表，然后用于预测。
      **/

    //3.获取所有用户的观看列表
    userHasItem = data.ratings.groupBy(_.user)

    //1.生成用户喜欢的电影
    userLikedMap = userLikedItems()
    //调试信息
    logger.info("训练集中的用户总数:" + userLikedMap.size)

    //2.根据用户评分向量生成用户最邻近用户的列表
    logger.info("计算用户邻近的相似用户中....")
    nearestUser = userNearestTopN()



    //调试信息
    logger.info("nearestUser.count():" + nearestUser.size)
    nearestUser.take(10).foreach(println)

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


        /*  //限制u1相似度列表的大小
          val u1SCount = userNearestAccumulator.value.count(r => r._1.indexOf(s",$u1,") > -1)
          //限制u2相似度列表的大小
          val u2SCount = userNearestAccumulator.value.count(r => r._1.indexOf(s",$u2,") > -1)
          //logger.info(s"u1SCount:$u1SCount,u2SCount:$u2SCount")


          if (u1SCount < ap.numNearestUsers && u2SCount < ap.numNearestUsers) {
            userNearestAccumulator.add(u1, u2, score)
          } else {

            if (u1SCount >= ap.numNearestUsers) {
              //选择小的替换
              val min_p: (String, Double) = userNearestAccumulator.value.filter(r => r._1.indexOf("," + u1 + ",") > -1).minBy(_._2)
              if (score > min_p._2) {
                userNearestAccumulator.value.remove(min_p._1)
                userNearestAccumulator.add(u1, u2, score)
              }
            }

            if (u2SCount >= ap.numNearestUsers) {
              //选择小的替换
              val min_p: (String, Double) = userNearestAccumulator.value.filter(r => r._1.indexOf("," + u2 + ",") > -1).minBy(_._2)
              if (score > min_p._2) {
                userNearestAccumulator.value.remove(min_p._1)
                userNearestAccumulator.add(u1, u2, score)
              }
            }
          }*/
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
    val userNearestMap= userNearestRDD.filter(r=>{
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
      val itemid = r._1
      val scores = r._2.map(_._2).sum
      //logger.info(s"累加的相似度：${scores},物品的评论数量:${itemCountSeq(r._1)}")
      (itemid, scores+itemCountSeq(r._1))
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


