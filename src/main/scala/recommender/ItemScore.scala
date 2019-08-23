package recommender

/**
  * 物品的ID和评分
  */
case class ItemScore(item: Int, score: Double) {
  override def toString: String = {
    s"item:$item,score:$score"
  }
}
