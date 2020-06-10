package recommender

/**
  * Author:IceS
  * Date:2019-08-09 15:24:48
  * Description:
  * 定义参数必须实现的接口，用于在保存结果时，形成文件名使用。
  */


trait Params {
  def getName(): String
}
