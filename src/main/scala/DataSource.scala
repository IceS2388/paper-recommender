import org.slf4j.{Logger, LoggerFactory}

import scala.io.Source

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
  * Author:IceS
  * Date:2019-08-09 15:17:01
  * Description:
  * 数据源
  */
class DataSource(dataFilePath: String = "data/ratings.csv") {
  @transient lazy val logger: Logger =LoggerFactory.getLogger(this.getClass)

  private def getRatings(): Seq[Rating] = {

    Source.fromFile(dataFilePath).getLines().map(line => {
      val data = line.split(",")
      Rating(data(0), data(1), data(2).toDouble, data(3).toLong)
    }).toSeq
  }

  def spliteRatings(kFold: Int, topN: Int): Seq[(TrainingData, Map[Query, ActualResult])] = {

    val ratings: Seq[(Rating, Int)] = getRatings().zipWithIndex

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
