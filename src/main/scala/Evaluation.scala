import org.slf4j.{Logger, LoggerFactory}

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
case class Query(user: String, num: Int) {
  override def toString: String = {
    s"{user:$user,num:$num}"
  }
}

/**
  * 验证结果。
  **/
case class VerifiedResult(precision: Double, recall: Double, f1: Double) {
  override def toString: String = {
    s"{precision:$precision,recall:$recall,f1:$f1}"
  }

  def +(other: VerifiedResult): VerifiedResult = {
    VerifiedResult(this.precision + other.precision, this.recall + other.recall, this.f1 + other.f1)
  }
}

/**
  * Author:IceS
  * Date:2019-08-09 15:20:35
  * Description:
  * NONE
  */
class Evaluation {
  @transient private lazy val logger: Logger =LoggerFactory.getLogger(this.getClass)

  def run(recommender: Recommender): Unit = {

    //: Seq[(TrainingData, Map[Query, ActualResult])]
    logger.info("这是交叉循环验证：训练集80%，验证集20%")
    logger.info("正在进行数据分割处理，需要些时间...")
    val data = new DataSource().spliteRatings(5, 20)
    logger.info("数据分割完毕")
    //TODO 实现更好的处理方法
    for ((trainingData, testingData) <- data) {
      logger.info("训练模型中...")
      recommender.train(trainingData)
      logger.info("训练模型完毕，开始进行预测评估")
      val vmean = calulate(testingData, recommender)
      logger.info("最终的平均结果值")
      logger.info(vmean.toString)
      Thread.sleep(1000)

    }


  }

  private def calulate(testingData: Map[Query, ActualResult], recommender: Recommender): VerifiedResult = {
    /**
      * P(Predicted)      N(Predicted)
      *
      * P(Actual)     True Positive      False Negative
      *
      * N(Actual)     False Positive     True Negative
      *
      * Precision = TP / (TP + FP)      分母为预测时推荐的记录条数
      *
      * Recall = TP / (TP + FN)         分母为测试时该用户拥有的测试记录条数
      *
      * F1 = 2TP / (2TP + FP + FN)
      **/

    logger.info("----------------单次结果--------------------")
    logger.info(s"(Query, ActualResult)的数量：${testingData.size}")
    val vsum = testingData.map(r => {
      //计算指标
      logger.info(s"Query：${r._1.user},条数:${r._1.num}")

      val predicts = recommender.predict(r._1)

      val actuallyItems = r._2.ratings.map(ar => ar.item)
      val predictedItems = predicts.itemScores.map(ir => ir.item)
      if (predictedItems.length == 0) {
        //返回每一个用户ID的验证结果
        val re = VerifiedResult(0, 0, 0)
        logger.info("没有生成预测列表！样本数量:${r._2.ratings.length}")
        logger.info(re.toString)
        re
      } else {
        //命中的数量TP
        val hit = actuallyItems.toSet.intersect(predictedItems.toSet).size
        //Precision = TP / (TP + FP)
        val precision = hit * 1.0 / predictedItems.length
        //Recall = TP / (TP + FN)
        val recall = hit * 1.0 / actuallyItems.length
        //F1 = 2TP / (2TP + FP + FN)
        val f1 = 2.0 * hit / (predictedItems.length + actuallyItems.length)

        //TODO 注释
        logger.info(s"user:${r._1.user},命中数量:${hit}，样本数量:${r._2.ratings.length}")
        //返回每一个用户ID的验证结果
        val re = VerifiedResult(precision, recall, f1)
        logger.info(re.toString)
        re
      }
    }).reduce(_ + _)

    VerifiedResult(vsum.precision / testingData.size,
      vsum.recall / testingData.size,
      vsum.f1 / testingData.size
    )
  }
}
