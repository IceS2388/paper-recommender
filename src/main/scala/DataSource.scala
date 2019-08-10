import org.slf4j.{Logger, LoggerFactory}

import scala.io.Source
import scala.util.Random

/**
  * 单条评分记录
  **/
case class Rating(user: String, item: String, rating: Double, timestampe: Long)

/**
  * TrainingData包含所有上面定义的Rating类型数据。
  **/
class TrainingData(val ratings: Seq[Rating]) {
  override def toString = {
    s"ratings: [${ratings.size}] (${ratings.take(2).toList}...)"
  }
}

/**
  * 为验证的用户评分，Rating类型的数组。
  * 用户ID
  * 物品ID
  * 评分
  **/
case class ActualResult(ratings: Array[Rating])

/**
  * 用户ID和查询数量
  **/
case class Query(user: String, num: Int) {
  override def toString: String = {
    s"{user:$user,num:$num}"
  }
}

/**
  * Author:IceS
  * Date:2019-08-09 15:17:01
  * Description:
  * 数据源
  */
class DataSource(dataFilePath: String = "data/ratings.csv") {
  @transient lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  private def getRatings(): Seq[Rating] = {

    Source.fromFile(dataFilePath).getLines().map(line => {
      val data = line.split(",")
      Rating(data(0), data(3), data(1).toDouble, data(2).toLong)
    }).toSeq
  }

  def spliteRatings(kFold: Float, topN: Int): (TrainingData, Seq[(Query, ActualResult)]) = {

    require(0 < kFold && kFold < 1, "测试集的所占百分比，必须是0至1之间!")

   /* val userRatings=getRatings().groupBy(_.user)
    userRatings.map(r=>{
      r.
    })*/

    val ratings = getRatings().zipWithIndex

    val trainThreashold = ((1 - kFold) * 100).toInt
    val trainingData: Seq[(Rating, Int)] = ratings.filter(rating => {
      val r = Random.nextInt(100)
      r < trainThreashold
    })
    val trainIndexs = trainingData.map(_._2).toSet

    logger.info(s"训练集：${trainingData.size},测试集:${ratings.size-trainingData.size}")
    trainingData.take(10).foreach(println)

    val testingData: Seq[(String, Seq[Rating])] = ratings.filter(r => {
      !trainIndexs.contains(r._2)
    }).map(_._1).groupBy(_.user).toSeq.sortBy(_._1)
    testingData.take(10).foreach(println)

    (new TrainingData(trainingData.map(_._1)), testingData.map {
      case (user, testRatings) => (Query(user, topN), ActualResult(testRatings.toArray))
    })


  }

}
