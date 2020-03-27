package recommender

/**
  * 验证结果。
  **/
case class VerifiedResult(precision: Double, recall: Double, f1: Double, exectime: Long) {
  override def toString: String = {
    s"准确率:%.4f,召回率:%.4f,f1:%.4f,时间:%d(ms)".format(precision, recall, f1, exectime)
  }

  def +(other: VerifiedResult): VerifiedResult = {
    VerifiedResult(this.precision + other.precision, this.recall + other.recall, this.f1 +
      other.f1, this.exectime + other.exectime)
  }
}
