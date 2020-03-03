package recommender.impl

/**
  * Author:IceS
  * Date:2019-08-10 11:15:48
  * Description:
  * 纯粹的Pearson相似度验证
  */

import org.slf4j.{Logger, LoggerFactory}
import recommender._
import recommender.tools.Correlation

import scala.collection.mutable

case class BasicParams(//计算相似度的方法
                       method: String = "Cosine",
                       //计算用户之间相似度的共同评分电影的阀值
                       T: Int = 5,
                       //最近邻用户的个数
                       K: Int = 100,
                       //用户喜欢的电影列表长度
                       L: Int = 1000) extends Params {
  override def toString: String = {
    s"参数：{method:$method,T:$T,K:$K,L:$L}\r\n"
  }

  override def getName(): String = {
    s"${this.getClass.getSimpleName.replace("Params", "")}_$method"
  }
}

class BasicRecommender(val ap: BasicParams) extends Recommender {

  //最近邻用户列表
  private var nearestUsers: Map[Int, List[(Int, Double)]] = _
  //用户喜欢电影列表
  private var usersLikeMovies: Map[Int, List[Rating]] = _
  //用户已经观看电影的列表
  private var userWatchedItem: Map[Int, Seq[Rating]] = _

  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  /** 保存结果时使用 */
  override def getParams: Params = ap

  override def prepare(data: Seq[Rating]): PreparedData = {
    new PreparedData(data)
  }

  override def train(data: TrainingData): Unit = {
    //验证数据
    require(data.ratings.nonEmpty, "评论数据不能为空！")

    //1.转换为HashMap,方便计算用户相似度,这是个昂贵的操作
    userWatchedItem = data.ratings.groupBy(r => r.user)

    //2.计算用户与用户之间相关系数
    nearestUsers = getNearestUsers(userWatchedItem)

    //3.返回用户观看过后喜欢的列表和相关系数最大的前K个用户的列表
    usersLikeMovies = getUsersLikeMovies(userWatchedItem)

  }

  private def getUsersLikeMovies(userRatings: Map[Int, Seq[Rating]]): Map[Int, List[Rating]] = {
    //从用户的观看记录中选择用户喜欢的电影,用于后续的用户与用户之间的推荐
    val userLikesBeyondMean = userRatings.map(r => {
      //当前用户的平均评分
      val mean = r._2.map(_.rating).sum / r._2.size

      //用户浏览的小于L，全部返回
      val userLikes = r._2.filter(r => r.rating > mean).toList.sortBy(_.rating).reverse.take(ap.L)

      (r._1, userLikes)
    })

    userLikesBeyondMean
  }

  private def getNearestUsers(userRatings: Map[Int, Seq[Rating]]) = {
    //1.获取用户的ID(这里的用户ID，只是包含测试集中的用户ID)
    val users = userRatings.keySet

    val userNearestPearson = new mutable.HashMap[Int, List[(Int, Double)]]()

    val userSimilarMap = new SimilarHashMap()
    for {
      u1 <- users
    } {

      val maxPearson: mutable.Map[Int, Double] = mutable.HashMap.empty
      for {u2 <- users
           if u1 != u2
      } {

        if (!userSimilarMap.contains(u1, u2)) {

          val ps = if (ap.method.toLowerCase() == "cosine") {
            Correlation.getCosine(ap.T, u1, u2, userRatings)
          } else if (ap.method.toLowerCase() == "pearson") {
            Correlation.getPearson(ap.T, u1, u2, userRatings)
          } else if (ap.method.toLowerCase() == "improvedpearson") {
            Correlation.getImprovedPearson(ap.T, u1, u2, userRatings)
          } else if (ap.method.toLowerCase() == "jaccard") {
            Correlation.getJaccard(ap.T, u1, u2, userRatings)
          } else if (ap.method.toLowerCase() == "jaccardmsd") {
            Correlation.getJaccardMSD(ap.T, u1, u2, userRatings)
          } else if (ap.method.toLowerCase() == "adjustcosine") {
            Correlation.getAdjustCosine(ap.T, u1, u2, userRatings)
          }
          else {
            throw new Exception("没有找到对应的方法！")
          }

          userSimilarMap.put(u1, u2, ps)
        }

        val ps = userSimilarMap.get(u1, u2)

        if (ps > 0) {
          //有用的相似度
          maxPearson.put(u2, ps)
        }
      }
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

    //2.获取当前用户的相似度最大的用户列表
    //2.1 判断有没有列表
    val similarUser: Map[Int, List[(Int, Double)]] = nearestUsers.filter(r => r._1 == query.user)
    if (similarUser.isEmpty) {
      //该用户没有最相似的用户列表
      logger.warn(s"该用户:${query.user}没有${ap.method}相似用户列表，无法生成推荐！")
      return PredictedResult(Array.empty)
    }
    logger.info(s"当前用户的相似度用户数量为：${similarUser(query.user).size}")

    //2.2 返回当前用户的相似用户和相似度
    val currentUsersAndSimilarMap: Map[Int, Double] = similarUser.map(r => {
      val uid = r._1
      val nearestUser = r._2.sortBy(_._2).reverse.take(ap.K)
      (uid, nearestUser)
    }).flatMap(_._2)
    logger.info(s"相似度用户的物品数量为：${currentUsersAndSimilarMap.size}")


    //2.3 查询当前用户已经看过的电影
    val userSawMovies = currentUserRDD.flatMap(r => r._2.map(rr => rr.item)).toSet

    //3. 从用户喜欢的电影列表，获取相似度用户看过的电影
    //原先的版本是从用户看过的列表中选择
    val result = usersLikeMovies.filter(r => {
      // r._1 用户ID
      //3.1 筛选相关用户看过的电影列表
      currentUsersAndSimilarMap.nonEmpty && currentUsersAndSimilarMap.contains(r._1)
    }).flatMap(r => {
      //r: (String, Iterable[Rating])
      //3.2 生成每一个item的积分
      r._2.map(r2 => {
        (r2.item, r2.rating * currentUsersAndSimilarMap(r._1))
      })
    }).filter(r => {
      //r._1 itemID
      // 3.3 过滤掉用户已经看过的电影
      !userSawMovies.contains(r._1)
    }).groupBy(_._1).map(r => {
      val itemID = r._1
      val scores = r._2.values.sum[Double]
      (itemID, scores)
    })
    logger.info(s"生成物品列表的数量为：${result.size}")

    val sum: Double = result.values.sum
    if (sum == 0) return PredictedResult(Array.empty)

    logger.info(s"生成候选物品列表的长度为：${result.size}")
    val weight = 1.0D
    val returnResult = result.map(r => {
      ItemScore(r._1, r._2 / sum * weight)
    }).toArray.sortBy(_.score).reverse.take(query.num)

    //排序，返回结果
    PredictedResult(returnResult)
  }


}

class SimilarHashMap {
  private val myMap = new mutable.HashMap[String, Double]()

  def put(u1: Int, u2: Int, score: Double): Unit = {
    val key1 = s",$u1,$u2,"
    val key2 = s",$u2,$u1,"
    if (!myMap.contains(key1) && !myMap.contains(key2)) {
      myMap.put(key1, score)
    } else if (myMap.contains(key1)) {
      myMap.put(key1, score)
    } else if (myMap.contains(key2)) {
      myMap.put(key2, score)
    }
  }

  def get(u1: Int, u2: Int): Double = {
    val key1 = s",$u1,$u2,"
    val key2 = s",$u2,$u1,"
    if (!myMap.contains(key1) && !myMap.contains(key2)) {
      throw new Exception(s"没有找到key:$u1 和 $u2 对应的项!")
    } else if (myMap.contains(key1)) {
      myMap(key1)
    } else if (myMap.contains(key2)) {
      myMap(key2)
    } else {
      0D
    }
  }

  def contains(u1: Int, u2: Int): Boolean = {
    val key1 = s",$u1,$u2,"
    val key2 = s",$u2,$u1,"
    myMap.contains(key1) || myMap.contains(key2)
  }
}