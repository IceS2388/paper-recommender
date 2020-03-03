package recommender.impl

import org.apache.spark.mllib.clustering.BisectingKMeans
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}
import recommender._
import recommender.tools.{Correlation, NearestUserAccumulator}

import scala.collection.mutable
import scala.util.Random

/**
  * Author:IceS
  * Date:2019-09-02 19:13:32
  * Description:
  * 使用Keras模型来尝试推荐
  */
case class KerasClusterParams(
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

class KerasClusterRecommender(ap: KerasClusterParams) extends Recommender {
  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def getParams: Params = ap

  //训练集中用户所拥有item
  private var userHasItem: Map[Int, Seq[Rating]] = _

  //每个用户所有的物品
  private var allUserItemSet: Map[Int, Set[Int]] = _


  //用户的评分向量
  private var afterClusterRDD: Array[(Int, (Int, linalg.Vector))] = _

  override def prepare(data: Seq[Rating]): PreparedData = {

    //val hitFile = Paths.get("spark-warehouse", s"hitRecord_${new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss").format(new Date)}.txt").toFile()

    //fw = new FileWriter(hitFile)

    allUserItemSet = data.groupBy(_.user).map(r => {
      val userId = r._1
      //
      val itemSet = r._2.map(_.item).toSet
      (userId, itemSet)
    })
    //数据分割前
    new PreparedData(data)
  }


  private var newItemVector: collection.Map[Int, linalg.Vector] = _
  private var newUserVector: collection.Map[Int, linalg.Vector] = _

  override def train(data: TrainingData): Unit = {

    require(Set("cosine", "improvedpearson", "pearson").contains(ap.method.toLowerCase()))
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


    /** -------------------导入keras神经网络--------------------- **/

    val kerasPath = "D:\\model\\empty_model.h5"
    ncfModel = KerasModelImport.importKerasModelAndWeights(kerasPath, true)

    val size = data.ratings.size

    //2.正样本数据
    logger.info("正样本数据")
    val positiveData = data.ratings.map(r => {
      //构建特征数据
      val userV = r.user
      val itemV = r.item

      (Random.nextInt(size), userV, itemV, 1)
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

        //生成标签
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
      (r._2.toFloat, r._3.toFloat, r._4.toFloat)
    })



    //require(uidArray.nonEmpty && iidArray.nonEmpty && labelArray.nonEmpty,"训练数据不能为空！")
    require(ncfModel != null, "ncfModel不能为null")
    //训练模型
    logger.info("开始训练模型...")
    logger.info(s"最终数据的数量为:${finalData.length}")

   /* //创建批量模型
    val uidArray = finalData.map(_._1)
    val iidArray = finalData.map(_._2)
    val labelArray: Array[Float] = finalData.map(_._3)

    val minibatch=1
    val inputs: Array[INDArray] = Array(
      Nd4j.create(uidArray).reshape(minibatch,finalData.length/minibatch),
      Nd4j.create(iidArray).reshape(minibatch,finalData.length/minibatch))
    val labels: Array[INDArray] = Array(Nd4j.create(labelArray).reshape(minibatch,finalData.length/minibatch))

    ncfModel.fit(inputs,labels)*/
   val minibatch=1
    for (elem <- finalData) {
      val inputs: Array[INDArray] = Array(
        Nd4j.create(Array(elem._1)).reshape(minibatch,1),
        Nd4j.create(Array(elem._2)).reshape(minibatch,1))
      val labels: Array[INDArray] = Array(Nd4j.create(Array(elem._3)).reshape(minibatch,1))
      ncfModel.fit(inputs,labels)

    }

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


    //筛选相近用户
    val result = candidateItems.filter(r => {

      //新增NCF筛选
      val userV = newUserVector(query.user)
      val itemV = newItemVector(r._1)

      //生成特征向量
      val vU = Nd4j.create(userV.toArray).reshape(1, userV.size)
      val vI = Nd4j.create(itemV.toArray).reshape(1, itemV.size)

      val vs: Array[INDArray] = ncfModel.output(vU, vI)

      val sc: Double = vs(0).getDouble(0L)

      logger.info(s"返回值：$sc")
      sc > 0.6
    })

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


    //用户所有的物品，包含训练集和测试集中的物品
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
    logger.info(s"测试集中物品数目：${tSet.size},筛选过后包含测试集中命中数目:${iSet.size}，推荐列表中命中数目：${rSet.size}")

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
