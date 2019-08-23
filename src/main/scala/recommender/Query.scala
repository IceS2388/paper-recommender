package recommender

/**
  * 用户ID和查询数量
  **/
case class Query(user: Int, num: Int) {
  override def toString: String = {
    s"Query:{user:$user,num:$num}"
  }
}
