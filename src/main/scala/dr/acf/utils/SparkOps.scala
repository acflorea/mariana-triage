package dr.acf.utils

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Spark Context related functionality
  * Created by acflorea on 16/10/2016.
  */
trait SparkOps {

  def master: String

  //We'll use Spark local to handle our data
  private lazy val conf: SparkConf = {
    val _conf = new SparkConf

    _conf.setMaster(master)
    _conf.setAppName("mariana-triage")
    _conf.set("spark.driver.maxResultSize", "3g")
    _conf.set("spark.executor.extraJavaOptions", "-Dorg.bytedeco.javacpp.maxbytes=5368709120")
    _conf.set("spark.driver.extraJavaOptions", "-Dorg.bytedeco.javacpp.maxbytes=5368709120")
    _conf
  }

  lazy val sc: SparkContext = {
    val _sc = new SparkContext(conf)

    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.WARN)

    _sc
  }

}
