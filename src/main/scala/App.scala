import org.slf4j.{Logger, LoggerFactory}

/**
  * Author:IceS
  * Date:2019-08-09 15:12:37
  * Description:
  * NONE
  */
object App {
  @transient private lazy val logger: Logger =LoggerFactory.getLogger(this.getClass)
  def main(args: Array[String]): Unit = {
    runPearson()
  }

  def runPearson()={
    val ap=new PearsonParams()
    val recommender=new PearsonRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }

  def runCosine()={
    val ap=new CosineParams()
    val recommender=new CosineRecommender(ap)
    val eval=new Evaluation()
    eval.run(recommender)
  }
}
