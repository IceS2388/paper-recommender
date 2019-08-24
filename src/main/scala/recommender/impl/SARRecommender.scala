package recommender.impl

import breeze.linalg.DenseMatrix
import org.slf4j.{Logger, LoggerFactory}
import recommender._
import recommender.tools.{BiMap, Correlation}

/**
  * Author:IceS
  * Date:2019-08-12 17:41:56
  * Description:
  * 简单推荐算法实现
  * item co-occurrence and item similarity:item-item维度的矩阵，值为两个物品拥有共同用户的个数
  */
case class SARParams() extends Params {
  override def getName(): String = {
    this.getClass.getSimpleName.replace("Params", "")
  }

  override def toString: String = super.toString + "\r\n"
}

class SARRecommender(ap: SARParams) extends Recommender {

  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  override def getParams: Params = ap

  override def prepare(data: Seq[Rating]): PrepairedData = {

    new PrepairedData(data)
  }

  private var userId2Index: BiMap[Int, Long] = _
  private var itemId2Index: BiMap[Int, Long] = _

  private var userHasItems: Map[Int, Seq[Rating]] = _

  override def train(data: TrainingData): Unit = {
    userHasItems = data.ratings.groupBy(_.user)
    //生成全部用户ID
    val userIDSet: Seq[Int] = data.ratings.map(_.user).distinct
    userId2Index = BiMap.toIndex(userIDSet)

    //生成全部物品ID
    val itemIDSet = data.ratings.map(_.item).distinct
    itemId2Index = BiMap.toIndex(itemIDSet)

    //1.生成关联矩阵
    val affinityMatrix = DenseMatrix.zeros[Float](userIDSet.size, itemIDSet.size)
    data.ratings.foreach(r => {
      affinityMatrix.update(userId2Index(r.user).toInt, itemId2Index(r.item).toInt, 1F)
    })

    //2.生成item2item的相似度矩阵
    val itemToItemMatrix: DenseMatrix[Float] = DenseMatrix.zeros[Float](itemIDSet.size, itemIDSet.size)

    val itemsGroup = data.ratings.groupBy(_.item).map(r => (r._1, r._2.map(_.user).toSet))

    for {(item1, _) <- itemsGroup
         (item2, _) <- itemsGroup
         if item1 <= item2
    } {
      val s = Correlation.getJaccardSAR(1, item1, item2.toInt, itemsGroup).toFloat

      val index1 = itemId2Index(item1).toInt
      val index2 = itemId2Index(item2).toInt
      itemToItemMatrix.update(index1, index2, s)
      itemToItemMatrix.update(index2, index1, s)
    }
    //3.生成评分预测矩阵
    resultMatrix = affinityMatrix * itemToItemMatrix
    logger.info(s"A.rows:${affinityMatrix.rows},A.columns:${affinityMatrix.cols} B.rows:${itemToItemMatrix.rows},B.columns:${itemToItemMatrix.cols}")
    logger.info(s"R.rows:${resultMatrix.rows},R.columns:${resultMatrix.cols}")
    Thread.sleep(5000)
  }

  private var resultMatrix: DenseMatrix[Float] = _

  override def predict(query: Query): PredictedResult = {

    //1.判断是否在矩阵中
    if (!userId2Index.contains(query.user)) {
      logger.warn(s"该用户:${query.user}没有过评分记录，无法生成推荐！")
      return PredictedResult(Array.empty)
    }

    //2.获取用户在预测评分中的记录
    val row = resultMatrix(userId2Index(query.user).toInt, ::)

    logger.info(s"评分记录：${row.inner.length}")

    require(row.inner.length==itemId2Index.size,"列的长度与索引的数量不相同！")
    val indexToItem: BiMap[Long, Int] = itemId2Index.inverse

    //row.apply()
    val items =indexToItem.toSeq.map( r=>{
      (r._2, row.apply(r._1.toInt))
    })

    //3.过滤已经看过的
    val userSaw = userHasItems(query.user).map(_.item).toSet
    val result = items.filter(r => {
      userSaw.nonEmpty && !userSaw.contains(r._1)
    }).sortBy(_._2).reverse.take(query.num).map(r => {
      ItemScore(r._1, r._2)
    })
    PredictedResult(result.toArray)
  }
}

