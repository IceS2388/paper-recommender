/**
  * Author:IceS
  * Date:2019-08-09 16:35:58
  * Description:
  * NONE
  */
object Correlation {

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
    var xy =comItems.map(i => {
      //         item  u1_rating u2_rating
      //val t: (String, (Double, Double))
      i._2._1 * i._2._2
    }).sum

    //2.求分母
    val x_var=user1Data.map(r=>Math.pow(r.rating,2)).sum
    val y_var=user2Data.map(r=>Math.pow(r.rating,2)).sum


    //Cosine系数
    xy / (Math.sqrt(x_var) * Math.sqrt(y_var))
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

    val user1ComData = user1Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap
    val user2ComData = user2Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap

    //2.把共同物品转变为对应的评分
    val comItems = comItemSet.map(r => (r, (user1ComData(r), user2ComData(r))))

    //计算平均值和标准差
    val sum1 = user1Data.map(_.rating).sum
    val sum2 = user2Data.map(_.rating).sum

    //平均值
    val x_mean = sum1 / user1Data.size
    val y_mean = sum2 / user2Data.size

    //标准差
    var xy = 0D

    comItems.foreach(i => {
      //         item  u1_rating u2_rating
      //val t: (String, (Double, Double))

      //计算Pearson系数
      val x_vt = i._2._1 - x_mean
      val y_vt = i._2._2 - y_mean
      xy += x_vt * y_vt

    })
    val x_var=user1Data.map(r=>{
      Math.pow(r.rating-x_mean,2)
    }).sum
    val y_var=user2Data.map(r=>{
      Math.pow(r.rating-y_mean,2)
    }).sum

    //Pearson系数
    xy / (Math.sqrt(x_var) * Math.sqrt(y_var))
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

    val user1ComData = user1Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap
    val user2ComData = user2Data.filter(r => comItemSet.contains(r.item)).map(r => (r.item, r.rating)).toMap

    //2.把共同物品转变为对应的评分
    val comItems = comItemSet.map(r => (r, (user1ComData(r), user2ComData(r))))

    //计算平均值和标准差
    val count = comItems.size
    val sum1 = user1Data.map(_.rating).sum
    val sum2 = user2Data.map(_.rating).sum

    //平均值
    val x_mean = sum1 / user1Data.size
    val y_mean = sum2 / user2Data.size

    //标准差
    var xy = 0D


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

      /*x_var += Math.pow(x_vt , 2)
      y_var += Math.pow(y_vt, 2)*/
    })

    val x_var=user1Data.map(r=>{
      Math.pow(r.rating-x_mean,2)
    }).sum
    val y_var=user2Data.map(r=>{
      Math.pow(r.rating-y_mean,2)
    }).sum

    //计算偏差指数
    w = Math.pow(Math.E, Math.sqrt(w) * (-1) / count)

    //Pearson系数
    val pearson = xy / (Math.sqrt(x_var) * Math.sqrt(y_var))

    //改良过后的相似度计算方法
    pearson * w
  }
}
