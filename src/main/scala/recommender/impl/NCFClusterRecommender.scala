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
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}
import recommender.tools.NearestUserAccumulator

import scala.collection.mutable
import scala.util.Random


case class NCFClusterParams(
                             userThreashold: Int = 20,
                             itemThreashold: Int = 2,
                             method: String = "Cosine",
                             k: Int = 4,
                             maxIterations: Int = 20,
                             numNearestUsers: Int = 60,
                             numUserLikeMovies: Int = 100) extends Params {
  override def getName(): String = {
    this.getClass.getSimpleName.replace("Params", "") + s"_$method"
  }

  override def toString: String = s"聚类参数{userThreashold:$userThreashold,itemThreashold:$itemThreashold,method:$method,k:$k,maxIterations:$maxIterations,numNearestUsers:$numNearestUsers,numUserLikeMovies:$numUserLikeMovies}\r\n"
}


class NCFClusterRecommender(ap: NCFClusterParams) extends Recommender {
  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def getParams: Params = ap


  //用户的评分向量
  private var afterClusterRDD: Array[(Int, (Int, linalg.Vector))] = _
  //训练集中用户所拥有item
  private var userHasItem: Map[Int, Seq[Rating]] = _

  //每个用户所有的物品
  private var allUserItemSet: Map[Int, Set[Int]] = _

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
    /** -------------------构建用户和物品的特征向量--------------------- **/
    val fields: Seq[StructField] = List(
      StructField("uid", IntegerType, nullable = false),
      StructField("iid", IntegerType, nullable = false),
      StructField("rating", DoubleType, nullable = false),
      StructField("tt", LongType, nullable = false)
    )
    val schema = StructType(fields)
    val sparkSession = SparkSession.builder().master("local[*]").appName(this.getClass.getSimpleName).getOrCreate()

    val rowRDD = sparkSession.sparkContext.parallelize(data.ratings.map(r => Row(r.user, r.item, r.rating, r.timestamp)))

    val ratingsDF = sparkSession.createDataFrame(rowRDD, schema)
    ratingsDF.createOrReplaceTempView("ratings")


    //1.生成用户的评分向量
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
        |FROM ratings
        |GROUP BY uid
        |ORDER BY total ASC
      """.stripMargin)
    userVectorsDF.createOrReplaceTempView("uv")
    userVectorsDF.show()

    //2.生成物品的评分向量
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
        |ORDER BY total ASC
      """.stripMargin)
    itemVectorsDF.createOrReplaceTempView("iv")
    itemVectorsDF.show()


    /** -------------------对用户评分向量进行聚类--------------------- **/
    logger.info("正在对用户评分向量进行聚类，需要些时间...")
    //3.准备聚类
    userVectorsDF.printSchema()
    val userVectorsRDD = userVectorsDF.rdd.map(r => {
      val total= r.getAs[Long]("total").toFloat
      (r.getAs[Int]("uid"), Vectors.dense(
        r.getAs[Long]("c1")/total,
        r.getAs[Long]("c2")/total,
        r.getAs[Long]("c3")/total,
        r.getAs[Long]("c4")/total,
        r.getAs[Long]("c5")/total,
        r.getAs[Long]("c6")/total,
        r.getAs[Long]("c7")/total,
        r.getAs[Long]("c8")/total,
        r.getAs[Long]("c9")/total,
        r.getAs[Long]("c10")/total
      ))
    })
    val itemVectorsRDD = itemVectorsDF.rdd.map(r => {
      val total=r.getAs[Long]("total").toFloat
      (r.getAs[Int]("iid"), Vectors.dense(
        r.getAs[Long]("c1")/total,
        r.getAs[Long]("c2")/total,
        r.getAs[Long]("c3")/total,
        r.getAs[Long]("c4")/total,
        r.getAs[Long]("c5")/total,
        r.getAs[Long]("c6")/total,
        r.getAs[Long]("c7")/total,
        r.getAs[Long]("c8")/total,
        r.getAs[Long]("c9")/total,
        r.getAs[Long]("c10")/total
      ))
    })

    val bkm = new BisectingKMeans().setK(ap.k).setMaxIterations(ap.maxIterations)
    val model = bkm.run(userVectorsRDD.map(_._2))

    //model.clusterCenters.foreach(println)

    //4.聚类用户评分向量(族ID,评分向量),根据聚类计算用户之间的相似度使用
    afterClusterRDD = userVectorsRDD.map(r => {
      (model.predict(r._2), r)
    }).collect()

    /** -------------------生成共享数据--------------------- **/
    newUserVector = userVectorsRDD.collectAsMap()
    newItemVector = itemVectorsRDD.collectAsMap()

    sparkSession.close()

    //1.获取训练集中每个用户的观看列表
    userHasItem = data.ratings.groupBy(_.user)

    //2.生成用户喜欢的电影
    userLikedMap = userLikedItems()
    //调试信息
    logger.info("训练集中的用户总数:" + userLikedMap.size)

    //3.根据用户评分向量生成用户最邻近用户的列表
    logger.info("计算用户邻近的相似用户中....")
    nearestUser = userNearestTopN()


    /** -------------------神经网络--------------------- **/
    val unit=newUserVector.head._2.size+newItemVector.head._2.size
    //1.配置网络结构
    val computationGraphConf = new NeuralNetConfiguration.Builder()
      .seed(123)
      .activation(Activation.SIGMOID)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new org.nd4j.linalg.learning.config.AdaDelta())
      .l2(1e-3)
      .graphBuilder()
      .addInputs("input")
      .addLayer("GML", new DenseLayer.Builder().nIn(unit).activation(Activation.RELU).nOut(1).build(), "input")
      .addLayer("MLP4", new DenseLayer.Builder().nIn(unit).activation(Activation.RELU).nOut(4*unit).build(), "input")
      .addLayer("MLP2", new DenseLayer.Builder().nIn(4*unit).nOut(2*unit).build(), "MLP4")
      .addLayer("MLP1", new DenseLayer.Builder().nIn(2*unit).nOut(unit).build(), "MLP2")
      .addVertex("ncf", new MergeVertex(), "GML", "MLP1")
      .addLayer("out", new OutputLayer.Builder().nIn(unit+1).nOut(2).build(), "ncf")
      .setOutputs("out")
      .build()


    ncfModel = new ComputationGraph(computationGraphConf)
    ncfModel.init()

    val allItemsSet = data.ratings.map(_.item).distinct.toSet
    val size = data.ratings.size

    //2.正样本数据
    logger.info("正样本数据")
    val positiveData: Seq[(Int, INDArray, INDArray)] = data.ratings.map(r => {
      //构建特征数据
      val userV: linalg.Vector = newUserVector(r.user)
      val itemV: linalg.Vector = newItemVector(r.item)
      val arr = new Array[Float](userV.size + itemV.size)

      for (idx <- 0 until userV.size) {
        arr(idx) = userV(idx).toFloat
      }

      for (idx <- 0 until itemV.size) {
        arr(idx + userV.size) = itemV(idx).toFloat
      }

      //生成标签
      val la = new Array[Float](2)
      la(0) = 0
      la(1) = 1

      (Random.nextInt(size), Nd4j.create(arr).reshape(1, arr.length), Nd4j.create(la).reshape(1, 2))
    })

    //3.负样本数据
    logger.info("负样本数据")
    val negativeData: Seq[(Int, INDArray, INDArray)] = userHasItem.flatMap(r => {
      val userHadSet = r._2.map(_.item).distinct.toSet
      //TODO 这里有bug，测试集的数据可能在里面
      val negativeSet = allItemsSet.diff(userHadSet)
      //保证负样本的数量，和正样本数量一致
      val nSet = Random.shuffle(negativeSet).take(100)

      val userV = newUserVector(r._1)

      nSet.map(itemID => {
        val itemV = newItemVector(itemID)
        val arr = new Array[Float](userV.size + itemV.size)

        for (idx <- 0 until userV.size) {
          arr(idx) = userV(idx).toFloat
        }

        for (idx <- 0 until itemV.size) {
          arr(idx + userV.size) = itemV(idx).toFloat
        }

        //生成标签
        val la = new Array[Float](2)
        la(0) = 1
        la(1) = 0
        (Random.nextInt(size), Nd4j.create(arr).reshape(1, arr.length), Nd4j.create(la).reshape(1, 2))
      })
    }).toSeq

    //4.数据打散
    logger.info("数据打散")
    val trainTempData = new Array[(Int, INDArray, INDArray)](positiveData.size + negativeData.size)
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
      (r._2, r._3)
    })

    //训练模型
    logger.info("开始训练模型...")
    var logIdx=0
    finalData.foreach(r => {
      if(logIdx%100==0){
        logger.info(s"index:$logIdx")
      }

      ncfModel.fit(Array(r._1), Array(r._2))
      logIdx+=1
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

    var logN=0
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
      val arr = new Array[Float](userV.size + itemV.size)

      for (idx <- 0 until userV.size) {
        arr(idx) = userV(idx).toFloat
      }
      for (idx <- 0 until itemV.size) {
        arr(idx + userV.size) = itemV(idx).toFloat
      }

      val vs: Array[INDArray] = ncfModel.output(Nd4j.create(arr).reshape(1, arr.length))

      val sc=vs(0).getFloat(1)
      if(logN<2){
        logger.info(s"itemID:${r._1},正样例率:$sc,负样例率:${vs(0).getFloat(0)}")
        logN+=1
      }

      val itemid = r._1
      val scores = r._2.map(_._2).sum*sc
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
