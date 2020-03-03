package recommender

/**
  * 预处理后的数据
  * */
class PreparedData(val ratings: Seq[Rating]) {
  override  def toString:String = {
    s"PreparedData: [${ratings.size}] (${ratings.take(2).toList}...)"
  }
}
