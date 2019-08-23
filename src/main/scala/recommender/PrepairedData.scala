package recommender

/**
  * 数据筛选完毕后的数据
  * */
class PrepairedData(val ratings: Seq[Rating]) {
  override  def toString:String = {
    s"PrepairedData: [${ratings.size}] (${ratings.take(2).toList}...)"
  }
}
