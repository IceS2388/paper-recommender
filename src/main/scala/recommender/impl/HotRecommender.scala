package recommender.impl

import org.slf4j.{Logger, LoggerFactory}
import recommender._

/**
  * Author:IceS
  * Date:2019-08-12 10:14:41
  * Description:
  * 作为对比，热榜推荐。
  */
case class HotParams() extends Params {
  override def getName(): String = this.getClass.getSimpleName.replace("Params", "")

  override def toString: String = this.getName() + "\r\n"
}

class HotRecommender(ap: HotParams) extends Recommender {

  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def getParams: Params = ap

  override def prepare(data: Seq[Rating]): PreparedData = {
    new PreparedData(data)
  }

  private var hotestMovies: Array[(Int, Int)] = _

  private var userHasItem: Map[Int, Seq[Rating]] = _

  override def train(data: TrainingData): Unit = {
    //1.生成热榜
    hotestMovies = data.ratings.groupBy(_.item).map(r => {
      //r._1//Int item
      //r._2.size//评论的数量
      (r._1, r._2.size)
    }).toArray.sortBy(_._2).reverse

    //2.生成用户观看列表
    userHasItem = data.ratings.groupBy(_.user)
  }

  override def predict(query: Query): PredictedResult = {

    //用户的已经观看列表
    val currentUserSawSet = userHasItem(query.user).map(_.item)
    logger.info(s"已经观看的列表长度为:${currentUserSawSet.size}")
    //筛选相近用户
    val result: Array[(Int, Int)] = hotestMovies.
      //过滤已经看过的
      filter(r => {
      currentUserSawSet.nonEmpty && !currentUserSawSet.contains(r._1)
    }).take(query.num)
    logger.info(s"生成的推荐列表的长度:${result.length}")
    val sum = result.map(r => r._2).sum
    if (sum == 0) return PredictedResult(Array.empty)

    val weight = 1.0
    val returnResult = result.map(r => {
      ItemScore(r._1, r._2 / sum * weight)
    })

    //排序，返回结果
    PredictedResult(returnResult)
  }
}
