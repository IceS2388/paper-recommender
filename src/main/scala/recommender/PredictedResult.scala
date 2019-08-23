package recommender

/**
  * ItemScore的数组，最后返回给用户的结果
  **/
case class PredictedResult(itemScores: Array[ItemScore]) {
  override def toString: String = {
    s"PredictedResult{itemScores:Array[ItemScore]},itemScores.length=${itemScores.length}"
  }
}
