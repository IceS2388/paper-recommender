package recommender

import java.io.FileWriter
import java.nio.file.Paths
import java.text.SimpleDateFormat
import java.util.Date

import org.slf4j.{Logger, LoggerFactory}


/**
  * Author:IceS
  * Date:2019-08-09 15:20:35
  * Description:
  * 评估类
  */
class Evaluation {
  @transient private lazy val logger: Logger = LoggerFactory.getLogger(this.getClass)

  def run(recommender: Recommender): Unit = {
    //: Seq[(TrainingData, Map[Query, ActualResult])]
    val ds = new DataSource()
    logger.info("读取所有数据，并进行初始处理。")
    val preparedData = recommender.prepare(ds.getRatings())
    logger.info("划分数据，训练集80%，验证集20%")
    logger.info("正在进行数据分割处理，需要些时间...")
    val topN = 10
    val data = ds.splitRatings(5, topN, preparedData)
    logger.info("数据分割完毕")

    val resultFile = Paths.get(s"result/${recommender.getParams.getName()}_${topN}_${new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss").format(new Date)}.txt").toFile

    val fw = new FileWriter(resultFile)
    fw.append(recommender.getParams.toString)

    var finalResult = data.map(r => {
      val trainingData = r._1
      val testingData = r._2.toSeq

      logger.info("训练模型中...")
      recommender.train(trainingData)
      logger.info("训练模型完毕，开始进行预测评估")

      val vmean = calulate(testingData, recommender)

      fw.append(s"训练数据：${trainingData.ratings.size}条,测试数据:${testingData.map(_._2.ratings.length).sum}条，用户数量：${testingData.size} \r\n")
      fw.append(s"$vmean \r\n")
      fw.flush()

      logger.info("终值" + vmean.toString + "\r\n")

      vmean
    }).reduce(_ + _)

    finalResult = VerifiedResult(finalResult.precision / data.size, finalResult.recall / data.size, finalResult.f1 / data.size, finalResult.exectime / data.size)

    fw.append(s"最终：$finalResult \r\n")
    fw.close()



  }

  private def calulate(testingData: Seq[(Query, ActualResult)], recommender: Recommender): VerifiedResult = {
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
    //调试信息
    logger.info(s"(Query, ActualResult)的数量：${testingData.size}")
    val userCount=testingData.size
    var idx=0
    val vsum = testingData.map(r => {
      //计算指标
      logger.info(s"第${idx+1}个用户，还剩下：${userCount-idx-1}个。Query：${r._1.user},条数:${r._1.num}")
      idx+=1

      val startTime = System.currentTimeMillis()
      val predicts = recommender.predict(r._1)

      val actuallyItems = r._2.ratings.map(ar => ar.item)
      val predictedItems = predicts.itemScores.map(ir => ir.item)

      if (predictedItems.length == 0) {
        //返回每一个用户ID的验证结果
        val re = VerifiedResult(0, 0, 0, System.currentTimeMillis() - startTime)
        logger.info(s"没有生成预测列表！样本数量:${r._2.ratings.length}")
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


        logger.info(s"user:${r._1.user},命中数量:$hit，样本数量:${r._2.ratings.length}")
        //返回每一个用户ID的验证结果
        val re = VerifiedResult(precision, recall, f1, System.currentTimeMillis() - startTime)
        logger.info(re.toString)
        re
      }
    }).reduce(_ + _)

    VerifiedResult(vsum.precision / testingData.size,
      vsum.recall / testingData.size,
      vsum.f1 / testingData.size,
      vsum.exectime / testingData.size
    )
  }
}
