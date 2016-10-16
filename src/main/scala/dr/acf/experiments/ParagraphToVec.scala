package dr.acf.experiments

import java.io.{File, FileInputStream}
import java.util
import java.util.TimeZone

import dr.acf.utils.{RestrictedCSVRecordReader, SparkOps}
import org.apache.spark.rdd.RDD
import org.datavec.api.conf.Configuration
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.InputStreamInputSplit
import org.datavec.api.transform.TransformProcess
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

import scala.collection.mutable

/**
  * Created by acflorea on 15/10/2016.
  */
object ParagraphToVec extends SparkOps {

  // Let's define the schema of the data that we want to import
  // The order in which columns are defined here should match
  // the order in which they appear in the input data
  val inputDataSchema: Schema = new Schema.Builder()
    // record id
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

  val log = LoggerFactory.getLogger(ParagraphToVec.getClass)

  def main(args: Array[String]): Unit = {

    //Print out the schema:
    log.info("Input data schema details:")
    log.info(s"$inputDataSchema")

    log.info("\n\nOther information obtainable from schema:")
    log.info(s"Number of columns: ${inputDataSchema.numColumns}")
    log.info(s"Column names: ${inputDataSchema.getColumnNames}")
    log.info(s"Column types: ${inputDataSchema.getColumnTypes}")

    //=====================================================================
    //            Step 2: Define the operations we want to do
    //=====================================================================
    val transformProcess: TransformProcess = new TransformProcess.Builder(inputDataSchema)
      //Let's remove some column we don't need
      .removeAllColumnsExceptFor("component_id", "class")
      .build()

    // After executing all of these operations, we have a new and different schema:
    val outputSchema: Schema = transformProcess.getFinalSchema

    log.info("\n\n\nSchema after transforming data:")
    log.info(s"$outputSchema")


    //=====================================================================
    //            Step 2: Transform
    //=====================================================================
    val directory: String = new ClassPathResource("netbeansbugs.csv").getFile.getPath //Normally just define your directory like "file:/..." or "hdfs:/..."
    val stringData: RDD[String] = sc.textFile(directory)

    //We first need to parse this format. It's comma-delimited (CSV) format, so let's parse it using CSVRecordReader:
    val rr: RecordReader = new CSVRecordReader
    val parsedInputData: RDD[util.List[Writable]] = stringData.map(new StringToWritablesFunction(rr).call(_))

    //Now, let's execute the transforms we defined earlier:
    val processedData: RDD[util.List[Writable]] = SparkTransformExecutor.execute(parsedInputData, transformProcess)

    processedData.take(100) foreach println

    // Global params
    val batchSize: Int = 50

    // Load training file
    // assigned_to	bug_id	bug_severity	bug_status
    // component_id	creation_ts	delta_ts	product_id
    // resolution short_desc original_text text class
    val file: File = new ClassPathResource("netbeansbugs.csv").getFile

    val inputSplit = new InputStreamInputSplit(new FileInputStream(file))

    val recordReader = new RestrictedCSVRecordReader(1, CSVRecordReader.DEFAULT_DELIMITER, Seq(1, 5))
    recordReader.initialize(new Configuration(), inputSplit)

    val possibleLabels: mutable.Set[Int] = mutable.Set.empty[Int]
    while (recordReader.hasNext) {
      possibleLabels.+=(recordReader.next().get(1).toInt)
    }

    recordReader.initialize(new Configuration(), new InputStreamInputSplit(new FileInputStream(file)))
    recordReader.reset()

    //reader,label index,number of possible labels
    val iterator: DataSetIterator = new RecordReaderDataSetIterator(recordReader, 100, 1, possibleLabels.size)

    val allData: DataSet = iterator.next()
    allData.shuffle()
    val testAndTrain: SplitTestAndTrain = allData.splitTestAndTrain(0.65); //Use 65% of data for training

    val trainingData: DataSet = testAndTrain.getTrain
    val testData: DataSet = testAndTrain.getTest

    val numInputs = 1
    val outputNum = possibleLabels.size
    val iterations = 1000
    val seed = 6

    log.info("Build model....")
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .activation("tanh")
      .weightInit(WeightInit.XAVIER)
      .learningRate(0.1)
      .regularization(true).l2(1e-4)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(100)
        .build())
      .layer(1, new DenseLayer.Builder().nIn(100).nOut(100)
        .build())
      .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation("softmax")
        .nIn(100).nOut(outputNum).build())
      .backprop(true).pretrain(false)
      .build()

    //run the model
    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(100))

    model.fit(trainingData)

    //evaluate the model on the test set
    val eval: Evaluation = new Evaluation(outputNum)
    val output: INDArray = model.output(testData.getFeatures)
    eval.eval(testData.getLabels, output)
    log.info(eval.stats())

  }

}
