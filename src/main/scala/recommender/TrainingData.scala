package recommender

/**
  * TrainingData包含所有上面定义的Rating类型数据。
  **/
class TrainingData(val ratings: Seq[Rating]) {
  override def toString = {
    s"TrainingData: [${ratings.size}] (${ratings.take(2).toList}...)"
  }
}
