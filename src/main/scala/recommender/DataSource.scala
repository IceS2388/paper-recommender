
package recommender

import org.slf4j.{Logger, LoggerFactory}

import scala.io.Source

/**
  * Author:IceS
  * Date:2019-08-09 15:17:01
  * Description:
  * 数据源，用于读取数据和分割数据。
  */
class DataSource(dataFilePath: String = "data/u.data") {
  @transient lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  def getRatings(): Seq[Rating] = {

    Source.fromFile(dataFilePath).getLines().map(line => {
      val data = line.toString.trim.split("\t")
      /* logger.info(line)*/
      Rating(user = data(0).toInt, item = data(1).toInt, rating = data(2).toDouble, timestamp = data(3).toLong)
    }).toSeq
  }

  def splitRatings(kFold: Int, topN: Int, originRatings: PreparedData): Seq[(TrainingData, Map[Query, ActualResult])] = {

    val ratings: Seq[(Rating, Int)] = originRatings.ratings.zipWithIndex

    (0 until kFold).map(idx => {
      logger.info(s"正在进行${idx + 1}次数据分割.")


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
