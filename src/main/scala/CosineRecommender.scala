import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable

/**
  * Author:IceS
  * Date:2019-08-09 18:57:18
  * Description:
  * NONE
  */
case class CosineParams(commonThreashold: Int = 10, numNearestUsers: Int = 100, numUserLikeMovies: Int = 1000) extends Params

class CosineRecommender(ap: CosineParams) extends Recommender {

  private var nearestUsers: Map[String, List[(String, Double)]] = _
  private var usersLikeMovies: Map[String, List[Rating]] = _
  private var userWatchedItem: Map[String, Seq[Rating]] = _

  @transient private lazy val logger: Logger =LoggerFactory.getLogger(this.getClass)

  override def train(data: TrainingData): Unit = {
    //验证数据
    require(data.ratings.nonEmpty, "评论数据不能为空！")

    //1.转换为HashMap,方便计算Pearson相似度,这是个昂贵的操作
    userWatchedItem = data.ratings.groupBy(r => r.user)

    //2.计算用户与用户之间Pearson系数，并返回用户观看过后喜欢的列表和pearson系数最大的前TopN个用户的列表
    nearestUsers = getNearestUsers(userWatchedItem)
    usersLikeMovies = getUsersLikeMovies(userWatchedItem)

  }

  private def getUsersLikeMovies(userRatings: Map[String, Seq[Rating]]) = {
    //2.从用户的观看记录中选择用户喜欢的电影,用于后续的用户与用户之间的推荐
    val userLikesBeyondMean: Map[String, List[Rating]] = userRatings.map(r => {

      //当前用户的平均评分
      val count = r._2.size

      //用户浏览的小于numNearst，全部返回
      val userLikes = if (count < ap.numUserLikeMovies) {
        //排序后，直接返回
        r._2.toList.sortBy(_.rating).reverse
      } else {
        r._2.toList.sortBy(_.rating).reverse.take(ap.numUserLikeMovies)
      }

      //logger.info(s"user:${r._1} likes Movies Count ${userLikes.count(_=>true)}")

      (r._1, userLikes)
    })

    userLikesBeyondMean
  }

  private def getNearestUsers(userRatings: Map[String, Seq[Rating]]) = {
    //1.获取用户的ID(这里的用户ID，只是包含测试集中的用户ID)
    val users: Seq[String] = userRatings.keySet.toList.sortBy(_.toInt)
    //TODO 调试

    val userNearestPearson = new mutable.HashMap[String, List[(String, Double)]]()
    for {
      u1 <- users
    } {

      val maxPearson: mutable.Map[String, Double] = mutable.HashMap.empty
      for {u2 <- users
          if u1!=u2
      } {
        val ps = Correlation.getCosine(ap.commonThreashold, u1, u2, userRatings)
        if (ps > 0) {
          //有用的相似度
          if (maxPearson.size < ap.numNearestUsers) {
            maxPearson.put(u2, ps)
          } else {
            val min_p = maxPearson.map(r => (r._1, r._2)).minBy(r => r._2)
            if (ps > min_p._2) {
              maxPearson.remove(min_p._1)
              maxPearson.put(u2, ps)
            }

          }
        }
      }
      /*logger.info(s"训练:$u1 , ${maxPearson.size}")
      Thread.sleep(1000)*/
      //logger.info(s"user:$u1 nearest pearson users count:${maxPearson.count(_=>true)}")
      userNearestPearson.put(u1, maxPearson.toList.sortBy(_._2).reverse)
    }

    userNearestPearson.toMap
  }

  override def predict(query: Query): PredictedResult = {

    //1.判断当前用户有没有看过电影
    val currentUserRDD = userWatchedItem.filter(r => r._1 == query.user)
    if (currentUserRDD.isEmpty) {
      //该用户没有过评分记录，返回空值
      logger.warn(s"该用户:${query.user}没有过评分记录，无法生成推荐！")
      return PredictedResult(Array.empty)
    }

    //2.获取当前用户的Pearson值最大的用户列表
    //2.1 判断有没有列表
    val similaryUers = nearestUsers.filter(r => r._1 == query.user)
    if (similaryUers.isEmpty) {
      //该用户没有最相似的Pearson用户列表
      logger.warn(s"该用户:${query.user}没有cosine相似用户列表，无法生成推荐！")
      return PredictedResult(Array.empty)
    }

    val pUsersMap = similaryUers.flatMap(r => r._2)

    /*logger.info(s"相似度用户数量为:${similaryUers.size}，组成的未筛选的列表为:${pUsersMap.size}")
    Thread.sleep(1000)*/

    //这是当前查询用户已经看过的电影
    val userSawMovie = currentUserRDD.flatMap(r => r._2.map(rr => rr.item)).toSet


    //3. 从用户喜欢的电影列表，获取相似度用户看过的电影
    //原先的版本是从用户看过的列表中选择
    val result= usersLikeMovies.filter(r => {
      // r._1 用户ID
      //3.1 筛选相关用户看过的电影列表
      pUsersMap.contains(r._1)
    }).flatMap(r => {
      //r: (String, Iterable[Rating])
      //3.2 生成每一个item的积分
      r._2.map(r2 => {
        (r2.item, r2.rating * pUsersMap(r._1))
      })
    }).filter(r => {
      //r._1 itemID
      // 3.3 过滤掉用户已经看过的电影
      !userSawMovie.contains(r._1)
    }).groupBy(_._1).map(r=>{
      val itemid= r._1
      val scores= r._2.values.sum
      (itemid,scores)
    })


    val sum: Double = result.map(r => r._2).sum
    if (sum == 0) return PredictedResult(Array.empty)


    //logger.info(s"生成的Pearson相似度的长度为：${result.count()}")
    val weight = 1.0
    val returnResult = result.map(r => {
      ItemScore(r._1, r._2 / sum * weight)
    }).toArray.sortBy(_.score).reverse.take(query.num)


    //排序，返回结果
    PredictedResult(returnResult)
  }
}
