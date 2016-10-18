package dr.acf.experiments

import java.io.File
import java.util
import java.util.{Arrays, TimeZone}

import dr.acf.utils.SparkOps
import org.apache.spark.rdd.RDD
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.records.writer.impl.misc.{LibSvmRecordWriter, SVMLightRecordWriter}
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.condition.ConditionOp
import org.datavec.api.transform.condition.column.{CategoricalColumnCondition, IntegerColumnCondition}
import org.datavec.api.transform.filter.ConditionFilter
import org.datavec.api.transform.schema.Schema
import org.datavec.api.util.ClassPathResource
import org.datavec.api.writable.Writable
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer
import scala.util.Try

/**
 * Created by acflorea on 15/10/2016.
 */
object ParagraphToVec extends SparkOps {

  // Let's define the schema of the data that we want to import
  // The order in which columns are defined here should match
  // the order in which they appear in the input data
  val inputDataSchema: Schema = new Schema.Builder()
    .addColumnInteger("row_id")
    .addColumnInteger("assigned_to")
    .addColumnInteger("bug_id")
    .addColumnCategorical("bug_severity", util.Arrays.asList("normal", "enhancement", "major", "trivial", "critical", "minor"))
    .addColumnCategorical("bug_status", util.Arrays.asList("CLOSED", "RESOLVED", "VERIFIED", "trivial", "critical", "minor"))
    .addColumnInteger("component_id")
    .addColumnTime("creation_ts", TimeZone.getDefault)
    .addColumnTime("delta_ts", TimeZone.getDefault)
    .addColumnInteger("product_id")
    .addColumnCategorical("resolution", util.Arrays.asList("FIXED"))
    .addColumnsString("short_desc", "original_text", "text")
    .addColumnInteger("class")
    .build()

  val filteredDataSchema: Schema = new Schema.Builder()
    .addColumnInteger("component_id")
    .addColumnInteger("product_id")
    .addColumnInteger("class")
    .build()

  val log = LoggerFactory.getLogger(ParagraphToVec.getClass)

  def main(args: Array[String]): Unit = {

    System.out.println("OMP_NUM_THREADS " + System.getenv().get("OMP_NUM_THREADS"))

    //Print out the schema:
    log.info("Input data schema details:")
    log.info(s"$inputDataSchema")

    log.info("\n\nOther information obtainable from schema:")
    log.info(s"Number of columns: ${inputDataSchema.numColumns}")
    log.info(s"Column names: ${inputDataSchema.getColumnNames}")
    log.info(s"Column types: ${inputDataSchema.getColumnTypes}")

    //=====================================================================
    //            Step 2.a: Define the operations we want to do
    //=====================================================================
    val filterColumnsTransform: TransformProcess = new TransformProcess.Builder(inputDataSchema)
      //Let's remove some column we don't need
      //.filter(new ConditionFilter(new IntegerColumnCondition("class", ConditionOp.GreaterOrEqual, 25)))
      .removeAllColumnsExceptFor("component_id", "product_id", "class")
      .build()

    // After executing all of these operations, we have a new and different schema:
    val outputSchema: Schema = filterColumnsTransform.getFinalSchema

    log.info("\n\n\nSchema after transforming data:")
    log.info(s"$outputSchema")


    //=====================================================================
    //            Step 2.b: Transform
    //=====================================================================
    val directory: String = new ClassPathResource("netbeansbugs.csv").getFile.getPath
    val stringData: RDD[String] = sc.textFile(directory)

    //We first need to parse this format. It's comma-delimited (CSV) format, so let's parse it using CSVRecordReader:
    val rr: RecordReader = new CSVRecordReader(0, CSVRecordReader.DEFAULT_DELIMITER)
    val parsedInputData: RDD[util.List[Writable]] = stringData.map(new StringToWritablesFunction(rr).call(_))

    //Now, let's execute the transforms we defined earlier:
    val filteredData: RDD[util.List[Writable]] = SparkTransformExecutor.execute(parsedInputData, filterColumnsTransform)

    val components: java.util.Map[java.lang.Integer, java.lang.String] =
      new java.util.HashMap[java.lang.Integer, java.lang.String]()

    val products: java.util.Map[java.lang.Integer, java.lang.String] =
      new java.util.HashMap[java.lang.Integer, java.lang.String]()

    val classes: java.util.Map[java.lang.Integer, java.lang.String] =
      new java.util.HashMap[java.lang.Integer, java.lang.String]()

    filteredData.collect().foreach {
      writables =>
        if (Try(writables.get(0).toInt).isSuccess) components.put(writables.get(0).toInt, writables.get(0).toString)
        if (Try(writables.get(1).toInt).isSuccess) products.put(writables.get(1).toInt, writables.get(1).toString)
        if (Try(writables.last.toInt).isSuccess) classes.put(writables.last.toInt, writables.last.toString)
    }

    //=====================================================================
    //            Step 3.a: More transformations
    //=====================================================================
    val numericToCategoricalTransform: TransformProcess = new TransformProcess.Builder(filteredDataSchema)
      .integerToCategorical("component_id", components)
      .integerToCategorical("product_id", products)
      .categoricalToOneHot("component_id")
      .categoricalToOneHot("product_id")
      .integerToCategorical("class", classes)
      .build()

    //=====================================================================
    //            Step 3.b: Transform
    //=====================================================================
    //Now, let's execute the transforms we defined earlier:
    val transformedData: RDD[util.List[Writable]] = SparkTransformExecutor.execute(filteredData, numericToCategoricalTransform)

    val possibleLabels = transformedData.map(_.last.toString).distinct().count().toInt

    val data = transformedData.collect().toList

    //    val asJava: java.util.Collection[util.List[Writable]] = ListBuffer(data: _*)

    //    val outLibSVM = new ClassPathResource("netbeansbugs.libsvm").getFile
    //    // outLibSVM.createNewFile()
    //    val writer = new SVMLightRecordWriter(outLibSVM, true)
    //    data foreach {
    //      writer.write(_)
    //    }

    val (_trainingData, _testData) = data.splitAt(7 * data.size / 10)

    // train data
    val trainRecordReader = new CollectionRecordReader(_trainingData)
    val trainIterator: DataSetIterator =
      new RecordReaderDataSetIterator(trainRecordReader, 100, components.size() + products.size(), possibleLabels)

    // test data
    val testRecordReader = new CollectionRecordReader(_testData)
    val testIterator: DataSetIterator =
      new RecordReaderDataSetIterator(testRecordReader, _testData.length, components.size() + products.size(), possibleLabels)

    val numInputs = components.size() + products.size()
    val outputNum = possibleLabels
    val iterations = 1000
    val seed = 6

    val h1size = 100
    val h2size = 100
    val h3size = 100
    val learningRate = 0.15
    val activation = "relu"
    val activation_end = "softmax"

    log.info("Build model....")
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation(activation)
      .weightInit(WeightInit.XAVIER)
      .learningRate(learningRate)
      .regularization(true).l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(h1size)
        .build())
      .layer(1, new DenseLayer.Builder().nIn(h1size).nOut(h2size)
        .build())
      .layer(2, new DenseLayer.Builder().nIn(h2size).nOut(h3size)
        .build())
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(activation_end)
        .nIn(h3size).nOut(outputNum).build())
      .backprop(true).pretrain(false)
      .build()

    //run the model
    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100))

    model.fit(trainIterator)

    //evaluate the model on the test set
    val eval: Evaluation = new Evaluation(outputNum)
    //val output: INDArray = model.output(testData.getFeatures)
    //eval.eval(testData.getLabels, output)
    val output: INDArray = model.output(testIterator)
    eval.eval(testIterator.next().getLabels, output)
    log.info(eval.stats())

  }

}
