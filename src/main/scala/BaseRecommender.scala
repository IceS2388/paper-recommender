/**
  * Author:IceS
  * Date:2019-08-10 11:15:48
  * Description:
  * 纯粹的Pearson相似度验证
  */

import org.slf4j.{Logger, LoggerFactory}

import scala.collection.mutable

case class BaseParams(method: String = "Cosine", commonThreashold: Int = 5, numNearestUsers: Int = 100, numUserLikeMovies: Int = 1000) extends Params {
  override def toString: String = s"参数：{method:$method,commonThreashold:$commonThreashold,numNearestUsers:$numNearestUsers,numUserLikeMovies:$numUserLikeMovies}\r\n"

  override def getName(): String = s"${this.getClass.getSimpleName.replace("Params", "")}_$method"
}

class BaseRecommender(val ap: BaseParams) extends Recommender {

  private var nearestUsers: Map[Int, List[(Int, Double)]] = _
  private var usersLikeMovies: Map[Int, List[Rating]] = _
  private var userWatchedItem: Map[Int, Seq[Rating]] = _

  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  /** 保存结果时使用 */
  override def getParams: Params = ap

  override def prepare(data: Seq[Rating]): PrepairedData = {
    new PrepairedData(data)
  }

  override def train(data: TrainingData): Unit = {
    //验证数据
    require(data.ratings.nonEmpty, "评论数据不能为空！")

    //1.转换为HashMap,方便计算Pearson相似度,这是个昂贵的操作
    userWatchedItem = data.ratings.groupBy(r => r.user)

    //2.计算用户与用户之间Pearson系数，并返回用户观看过后喜欢的列表和pearson系数最大的前TopN个用户的列表
    nearestUsers = getNearestUsers(userWatchedItem)
    usersLikeMovies = getUsersLikeMovies(userWatchedItem)

  }

  private def getUsersLikeMovies(userRatings: Map[Int, Seq[Rating]]) = {
    //2.从用户的观看记录中选择用户喜欢的电影,用于后续的用户与用户之间的推荐
    val userLikesBeyondMean = userRatings.map(r => {

      //当前用户的平均评分
      val count = r._2.size
      val mean = r._2.map(_.rating).sum / r._2.size

      //用户浏览的小于numNearst，全部返回
      val userLikes = r._2.filter(r => r.rating > mean).toList.sortBy(_.rating).reverse.take(ap.numUserLikeMovies)

      //userLikes.foreach(println)
      //Thread.sleep(1000)
      (r._1, userLikes)
    })

    userLikesBeyondMean
  }

  private def getNearestUsers(userRatings: Map[Int, Seq[Rating]]) = {
    //1.获取用户的ID(这里的用户ID，只是包含测试集中的用户ID)
    val users = userRatings.keySet

    val userNearestPearson = new mutable.HashMap[Int, List[(Int, Double)]]()

    val userSimilaryMap = new SimilaryHashMap()
    for {
      u1 <- users
    } {

      val maxPearson: mutable.Map[Int, Double] = mutable.HashMap.empty
      for {u2 <- users
           if u1 != u2
      } {

        if (!userSimilaryMap.contains(u1, u2)) {

          val ps = if (ap.method.toLowerCase() == "cosine") {
            Correlation.getCosine(ap.commonThreashold, u1, u2, userRatings)
          } else if (ap.method.toLowerCase() == "pearson") {
            Correlation.getPearson(ap.commonThreashold, u1, u2, userRatings)
          } else if (ap.method.toLowerCase() == "improvedpearson") {
            Correlation.getImprovedPearson(ap.commonThreashold, u1, u2, userRatings)
          } else if (ap.method.toLowerCase() == "jaccard") {
            Correlation.getJaccard(ap.commonThreashold, u1, u2, userRatings)
          } else if (ap.method.toLowerCase() == "jaccardmsd") {
            Correlation.getJaccardMSD(ap.commonThreashold, u1, u2, userRatings)
          }else if(ap.method.toLowerCase()=="adjustcosine"){
            Correlation.getAdjustCosine(ap.commonThreashold, u1, u2, userRatings)
          }
          else {
            throw new Exception("没有找到对应的方法！")
          }

          userSimilaryMap.put(u1, u2, ps)
        }

        val ps = userSimilaryMap.get(u1, u2)

        if (ps > 0) {
          //有用的相似度
          maxPearson.put(u2, ps)
        }
      }
      /*logger.info(s"训练:$u1 , ${maxPearson.size}")
      Thread.sleep(1000)*/
      //logger.info(s"user:$u1 nearest pearson users count:${maxPearson.count(_=>true)}")
      userNearestPearson.put(u1, maxPearson.toList.sortBy(_._2).reverse)
      //userNearestPearson(u1).foreach(println)
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
    val similaryUers: Map[Int, List[(Int, Double)]] = nearestUsers.filter(r => r._1 == query.user)
    if (similaryUers.isEmpty) {
      //该用户没有最相似的Pearson用户列表
      logger.warn(s"该用户:${query.user}没有${ap.method}相似用户列表，无法生成推荐！")
      return PredictedResult(Array.empty)
    }
    logger.info(s"当前用户的相似度用户数量为：${similaryUers(query.user).size}")

    //返回当前用户的相似用户和相似度
    val pUsersMap: Map[Int, Double] = similaryUers.map(r => {
      val uid = r._1
      //uid
      val nearestUser = r._2.sortBy(_._2).reverse.take(ap.numNearestUsers)
      (uid, nearestUser)
    }).flatMap(_._2)
    logger.info(s"相似度用户的物品数量为：${pUsersMap.size}")

    /*logger.info(s"相似度用户数量为:${similaryUers.size}，组成的未筛选的列表为:${pUsersMap.size}")
    Thread.sleep(1000)*/

    //这是当前查询用户已经看过的电影
    val userSawMovie = currentUserRDD.flatMap(r => r._2.map(rr => rr.item)).toSet


    //3. 从用户喜欢的电影列表，获取相似度用户看过的电影
    //原先的版本是从用户看过的列表中选择
    val result = usersLikeMovies.filter(r => {
      // r._1 用户ID
      //3.1 筛选相关用户看过的电影列表
      pUsersMap.nonEmpty && pUsersMap.contains(r._1)
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
    }).groupBy(_._1).map(r => {
      val itemid = r._1
      val scores = r._2.values.sum[Double]
      (itemid, scores)
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

class SimilaryHashMap {
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