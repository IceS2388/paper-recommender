

/**
  * Author:IceS
  * Date:2019-08-09 15:24:48
  * Description:
  * NONE
  */


/**
  * 物品的ID和评分
  */
case class ItemScore(item: Int,score: Double) {
  override def toString: String = {
    s"item:$item,score:$score"
  }
}

/**
  * ItemScore的数组，最后返回给用户的结果
  **/
case class PredictedResult(itemScores: Array[ItemScore]) {
  override def toString: String = {
    s"PredictedResult{itemScores:Array[ItemScore]},itemScores.length=${itemScores.length}"
  }
}

trait Params{
  def getName():String
}
case class EmptyParams() extends Params {
  override def toString(): String = "Empty"

  override def getName(): String = this.getName()
}
abstract class Recommender{
   def train(data:TrainingData):Unit
   def predict(query: Query): PredictedResult
  def getParams:Params
}
