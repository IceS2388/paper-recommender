import org.apache.spark.mllib.linalg

/**
  * Author:IceS
  * Date:2019-08-09 16:35:58
  * Description:
  * NONE
  */
object Correlation {


  //尝试cos相似度
  def getCosine(v1: linalg.Vector, v2: linalg.Vector): Double = {
    var sum = 0D
    var v1Len = 0D
    var v2Len = 0D
    for (idx <- 0 until v1.size) {
      sum += v1.apply(idx) * v2.apply(idx)
      v1Len += Math.pow(v1.apply(idx), 2)
      v2Len += Math.pow(v2.apply(idx), 2)
    }
    if (v1Len == 0 || v2Len == 0)
      0D
    else
      sum / Math.sqrt(v1Len * v2Len)
  }

  /** 改进Pearson算法
    * r=sum((x-x_mean)*(y-y_mean))/(Math.pow(sum(x-x_mean),0.5)*Math.pow(sum(y-y_mean),0.5))
    * */
  def getPearson(v1: linalg.Vector, v2: linalg.Vector): Double = {

    var sum1 = 0D
    var sum2 = 0D
    for (idx <- 0 until v1.size) {
      sum1 += v1.apply(idx)
      sum2 += v2.apply(idx)
    }
    val mean1 = sum1 / v1.size
    val mean2 = sum2 / v2.size
    var sum = 0D
    sum1 = 0
    sum2 = 0
    for (idx <- 0 until v1.size) {
      sum += (v1.apply(idx) - mean1) * (v2.apply(idx) - mean2)
      sum1 += Math.pow(v1.apply(idx) - mean1, 2)
      sum2 += Math.pow(v2.apply(idx) - mean2, 2)
    }
    val sum1sum2 = Math.sqrt(sum1 * sum2)

    if (sum1sum2 == 0)
      0
    else
      sum / sum1sum2

  }

  def getImprovedPearson(v1: linalg.Vector, v2: linalg.Vector): Double = {
    //偏差因子
    var w = 0.0


    var sum1 = 0D
    var sum2 = 0D
    for (idx <- 0 until v1.size) {
      sum1 += v1.apply(idx)
      sum2 += v2.apply(idx)

      w += Math.pow(v1.apply(idx) - v2.apply(idx), 2)
    }
    val mean1 = sum1 / v1.size
    val mean2 = sum2 / v2.size
    var sum = 0D
    sum1 = 0
    sum2 = 0
    for (idx <- 0 until v1.size) {
      sum += (v1.apply(idx) - mean1) * (v2.apply(idx) - mean2)
      sum1 += Math.pow(v1.apply(idx) - mean1, 2)
      sum2 += Math.pow(v2.apply(idx) - mean2, 2)
    }
    val sum1sum2 = Math.sqrt(sum1 * sum2)

    //计算偏差指数
    w = Math.pow(Math.E, Math.sqrt(w) * (-1) / v1.size)

    if (sum1sum2 == 0)
      0
    else
      sum / sum1sum2 * w

  }

  def getJaccardSAR(threashold: Int,
                 key1: Int,
                 key2: Int,
                 hashItems: Map[Int, Set[Int]]): Double = {
    if (!hashItems.contains(key1) || !hashItems.contains(key2)) {
      //不相关
      return 0D
    }

    val user1Data = hashItems(key1)
    val user2Data = hashItems(key2)

    //1.求u1与u2共同的物品ID
    val comItemSet = user1Data.intersect(user2Data)

    if (comItemSet.size < threashold) {
      //小于共同物品的阀值，直接退出
      return 0D
    }

    comItemSet.size * 1.0 / (user1Data.size + user2Data.size - comItemSet.size)
  }

  def getJaccard(threashold: Int,
                 key1: Int,
                 key2: Int,
                 userHashRatings: Map[Int, Seq[Rating]]): Double = {
    if (!userHashRatings.contains(key1) || !userHashRatings.contains(key2)) {
      //不相关
      return 0D
    }

    val user1Data = userHashRatings(key1)
    val user2Data = userHashRatings(key2)

    //1.求u1与u2共同的物品ID
    val comItemSet = user1Data.map(r => r.item).toSet.intersect(user2Data.map(r => r.item).toSet)

    if (comItemSet.size < threashold) {
      //小于共同物品的阀值，直接退出
      return 0D
    }

    comItemSet.size * 1.0 / (user1Data.size + user2Data.size - comItemSet.size)
  }

  def getJaccardMSD(threashold: Int,
                    userid1: Int,
                    userid2: Int,
                    userHashRatings: Map[Int, Seq[Rating]]): Double = {
    if (!userHashRatings.contains(userid1) || !userHashRatings.contains(userid2)) {
      //不相关
      return 0D
    }

    val user1Data: Iterable[Rating] = userHashRatings(userid1)
    val user2Data: Iterable[Rating] = userHashRatings(userid2)

    //1.求u1与u2共同的物品ID
    val comItemSet = user1Data.map(r => r.item).toSet.intersect(user2Data.map(r => r.item).toSet)
    if (comItemSet.size < threashold) {
      //小于共同物品的阀值，直接退出
      return 0D
    }
    val user1ComData = user1Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap
    val user2ComData = user2Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap

    //2.把共同物品转变为对应的评分
    val comItems = comItemSet.map(r => (r, (user1ComData(r), user2ComData(r))))

    val differ = comItems.map(r => {
      Math.pow(r._2._1 - r._2._2, 2)
    }).sum
    val msd = 1 - differ * 1.0 / comItems.size

    val jaccard = comItems.size * 1.0 / (user1Data.size + user2Data.size - comItems.size)

    msd * jaccard

  }

  //修正的Cosine相似度
  def getAdjustCosine(threashold: Int,
                userid1: Int,
                userid2: Int,
                userHashRatings: Map[Int, Seq[Rating]]): Double = {

    if (!userHashRatings.contains(userid1) || !userHashRatings.contains(userid2)) {
      //不相关
      return 0D
    }

    val user1Data: Iterable[Rating] = userHashRatings(userid1)
    val user2Data: Iterable[Rating] = userHashRatings(userid2)

    //1.求u1与u2共同的物品ID
    val comItemSet = user1Data.map(r => r.item).toSet.intersect(user2Data.map(r => r.item).toSet)
    if (comItemSet.size < threashold) {
      //小于共同物品的阀值，直接退出
      return 0D
    }

    //求两者的并集
   val allItems= user1Data.map(r => r.item).toSet ++ user2Data.map(r => r.item).toSet


    //计算平均值和标准差
    val sum1 = user1Data.map(_.rating).sum
    val sum2 = user2Data.map(_.rating).sum
    //平均值
    val x_mean = sum1 / user1Data.size
    val y_mean = sum2 / user2Data.size

    val user1ItemData = user1Data.map(r => (r.item, r.rating)).toMap
    val user2ItemData = user2Data.map(r => (r.item, r.rating)).toMap


    var xy=0D
    var x_var=0D
    var y_var=0D

    //标准差
    allItems.map(i => {
      // i iid
      val u1=if(user1ItemData.contains(i)){
        user1ItemData(i)
      }else
        0D
      val u2=if(user2ItemData.contains(i))
        user2ItemData(i)
      else
        0D

      val x =u1-x_mean
      val y =u2-y_mean
      xy+=x*y
      x_var += Math.pow(x,2)
      y_var += Math.pow(y,2)
    })

    xy/Math.sqrt(x_var*y_var)


  }

  def getCosine(threashold: Int,
                userid1: Int,
                userid2: Int,
                userHashRatings: Map[Int, Seq[Rating]]): Double = {

    if (!userHashRatings.contains(userid1) || !userHashRatings.contains(userid2)) {
      //不相关
      return 0D
    }

    val user1Data: Iterable[Rating] = userHashRatings(userid1)
    val user2Data: Iterable[Rating] = userHashRatings(userid2)

    //1.求u1与u2共同的物品ID
    val comItemSet = user1Data.map(r => r.item).toSet.intersect(user2Data.map(r => r.item).toSet)
    if (comItemSet.size < threashold) {
      //小于共同物品的阀值，直接退出
      return 0D
    }

    val user1ComData = user1Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap
    val user2ComData = user2Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap

    //2.把共同物品转变为对应的评分
    val comItems = comItemSet.map(r => (r, (user1ComData(r), user2ComData(r))))


    //标准差
    var xy = comItems.map(i => {
      //         item  u1_rating u2_rating
      //val t: (String, (Double, Double))
      i._2._1 * i._2._2
    }).sum

    //2.求分母
    val x_var = user1Data.map(r => Math.pow(r.rating, 2)).sum
    val y_var = user2Data.map(r => Math.pow(r.rating, 2)).sum


    //Cosine系数
    xy / Math.sqrt(x_var * y_var)
  }

  /**
    * Pearson相似度计算公式:
    * r=sum((x-x_mean)*(y-y_mean))/(Math.sqrt(sum(Math.pow(x-x_mean,2)))*Math.sqrt(sum(Math.pow(y-y_mean,2))))
    * 要求，两者的共同因子必须达到阀值。默认为10
    *
    * 增加评分差距因子。2019年7月26日
    **/
  def getPearson(threashold: Int,
                 userid1: Int,
                 userid2: Int,
                 userHashRatings: Map[Int, Seq[Rating]]): Double = {

    if (!userHashRatings.contains(userid1) || !userHashRatings.contains(userid2)) {
      //不相关
      return 0D
    }

    val user1Data: Iterable[Rating] = userHashRatings(userid1)
    val user2Data: Iterable[Rating] = userHashRatings(userid2)

    //1.求u1与u2共同的物品ID
    val comItemSet = user1Data.map(r => r.item).toSet.intersect(user2Data.map(r => r.item).toSet)
    if (comItemSet.size < threashold) {
      //小于共同物品的阀值，直接退出
      return 0D
    }

    //计算平均值和标准差
    val sum1 = user1Data.map(_.rating).sum
    val sum2 = user2Data.map(_.rating).sum

    //平均值
    val x_mean = sum1 / user1Data.size
    val y_mean = sum2 / user2Data.size


    val user1ComData = user1Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap
    val user2ComData = user2Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap

    //2.把共同物品转变为对应的评分
    val comItems = comItemSet.map(r => (r, (user1ComData(r), user2ComData(r))))

    //标准差
    var xy = 0D
    var x_var = 0D
    var y_var = 0D

    comItems.foreach(i => {
      //         item  u1_rating u2_rating
      //val t: (String, (Double, Double))

      //计算Pearson系数
      val x_vt = i._2._1 - x_mean
      val y_vt = i._2._2 - y_mean
      xy += x_vt * y_vt

      x_var += (x_vt * x_vt)
      y_var += (y_vt * y_vt)

    })

    //Pearson系数
    xy / Math.sqrt(x_var * y_var)
  }

  def getImprovedPearson(threashold: Int,
                         userid1: Int,
                         userid2: Int,
                         userHashRatings: Map[Int, Seq[Rating]]): Double = {

    if (!userHashRatings.contains(userid1) || !userHashRatings.contains(userid2)) {
      //不相关
      return 0D
    }

    val user1Data: Iterable[Rating] = userHashRatings(userid1)
    val user2Data: Iterable[Rating] = userHashRatings(userid2)

    //1.求u1与u2共同的物品ID
    val comItemSet = user1Data.map(r => r.item).toSet.intersect(user2Data.map(r => r.item).toSet)
    if (comItemSet.size < threashold) {
      //小于共同物品的阀值，直接退出
      return 0D
    }

    //计算平均值和标准差
    val sum1 = user1Data.map(_.rating).sum
    val sum2 = user2Data.map(_.rating).sum

    //平均值
    val x_mean = sum1 / user1Data.size
    val y_mean = sum2 / user2Data.size

    val user1ComData = user1Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap
    val user2ComData = user2Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap

    //2.把共同物品转变为对应的评分
    val comItems = comItemSet.map(r => (r, (user1ComData(r), user2ComData(r))))
    val count = comItems.size

    //标准差
    var xy = 0D
    var x_var = 0D
    var y_var = 0D

    //偏差因素
    var w = 0.0
    comItems.foreach(i => {
      //         item  u1_rating u2_rating
      //val t: (String, (Double, Double))
      w += Math.pow(i._2._1 - i._2._2, 2)

      //计算Pearson系数
      val x_vt = i._2._1 - x_mean
      val y_vt = i._2._2 - y_mean
      xy += x_vt * y_vt

      x_var += Math.pow(x_vt, 2)
      y_var += Math.pow(y_vt, 2)
    })


    //计算偏差指数
    w = Math.pow(Math.E, Math.sqrt(w) * (-1) / count)

    //Pearson系数
    val pearson = xy / Math.sqrt(x_var * y_var)

    //改良过后的相似度计算方法
    pearson * w
  }

  //获取两个点之间的距离
  def getDistance(v1: linalg.Vector, v2: linalg.Vector): Double = {
    val len = Math.max(v1.size, v2.size)
    var distance = 0D
    for (i <- 0 until len) {
      distance += Math.pow(v1.apply(i) - v2.apply(i), 2)
    }

    Math.sqrt(distance)
  }
}
