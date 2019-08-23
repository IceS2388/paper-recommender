package recommender

/**
  * 单条评分记录
  **/
case class Rating(user: Int, item: Int, rating: Double, timestamp: Long) {
  override def toString: String = {
    s"Rating:{user:$user,item:$item,rating:$rating,timestamp:$timestamp}"
  }
}
