import org.slf4j.{Logger, LoggerFactory}

import scala.io.Source

/**
  * 单条评分记录
  **/
case class Rating(user: Int, item: Int, rating: Double, timestamp: Long) {
  override def toString: String = {
    s"Rating:{user:$user,item:$item,rating:$rating,timestamp:$timestamp}"
  }
}
/**
  * 数据筛选完毕后的数据
  * */
class PrepairedData(val ratings: Seq[Rating]) {
  override  def toString:String = {
    s"PrepairedData: [${ratings.size}] (${ratings.take(2).toList}...)"
  }
}

/**
  * TrainingData包含所有上面定义的Rating类型数据。
  **/
class TrainingData(val ratings: Seq[Rating]) {
  override def toString = {
    s"TrainingData: [${ratings.size}] (${ratings.take(2).toList}...)"
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
case class Query(user: Int, num: Int) {
  override def toString: String = {
    s"Query:{user:$user,num:$num}"
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

   def getRatings(): Seq[Rating] = {

    Source.fromFile(dataFilePath).getLines().map(line => {
      val data = line.toString.trim.split(",")
      /* logger.info(line)*/
      Rating(user = data(0).toInt, item = data(3).toInt, rating = data(1).toDouble, timestamp = data(2).toLong)
    }).toSeq
  }

  def spliteRatings(kFold: Int, topN: Int,originRatings: PrepairedData): Seq[(TrainingData, Map[Query, ActualResult])] = {

    val ratings: Seq[(Rating, Int)] = originRatings.ratings.zipWithIndex

    (0 until kFold).map(idx => {
      logger.info(s"正在进行${idx+1}次数据分割.")


      //训练集:每条Rating的索引%KFold，若余数不等于当前idx，则添加到训练集
      val trainingRatings = ratings.filter(_._2 % kFold != idx).map(_._1)
      //测试集,若余数等idx则为测试集.
      val testingRatings = ratings.filter(_._2 % kFold == idx).map(_._1)
      //测试集按照用户ID进行分组，便于验证。
      val testingUsers = testingRatings.groupBy(r => r.user)
      logger.info(s"训练集大小：${trainingRatings.size},测试集大小：${testingRatings.size}")
      (new TrainingData(trainingRatings), testingUsers.map {
        case (user, testRatings) => (Query(user, topN), ActualResult(testRatings.toArray))
      })

    })
  }

}
