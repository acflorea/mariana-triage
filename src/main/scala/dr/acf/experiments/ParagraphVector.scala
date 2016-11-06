package dr.acf.experiments

import java.io.File
import java.util
import java.util.TimeZone

import com.beust.jcommander.{JCommander, Parameter}
import dr.acf.utils.{SmartEvaluation, SparkOps, WordVectorSmartSerializer}
import org.apache.spark.rdd.RDD
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.schema.Schema
import org.datavec.api.writable.{DoubleWritable, Writable}
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, OutputLayer, RBM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.util.Try

/**
 * Created by acflorea on 15/10/2016.
 */
object ParagraphVector extends SparkOps {

  object Args {
    // Declared as var because JCommander assigns a new collection declared
    // as java.util.List because that's what JCommander will replace it with.
    // It'd be nice if JCommander would just use the provided List so this
    // could be a val and a Scala LinkedList.
    @Parameter(
      names = Array("-rf", "--resourceFolder"),
      description = "Location of resource files.")
    var resourceFolder: String = "/Users/triage/"

    @Parameter(
      names = Array("-f", "--inputFile"),
      description = "Input File name")
    val inputFileName = "data/netbeansbugs_filtered.csv"

    @Parameter(
      names = Array("-ce", "--computeEmbeddings"),
      description = "Weather to compute the embedings or attempt to load an existing model")
    val computeEmbeddings = false

    @Parameter(
      names = Array("-ep", "--epochs"),
      description = "Nomber of epochs for embedings")
    val epochsForEmbeddings = 20
  }

  lazy val modelName = if (Args.epochsForEmbeddings < 10)
    s"netbeans_0${Args.epochsForEmbeddings}.model"
  else
    s"netbeans_${Args.epochsForEmbeddings}.model"

  val severityValues = util.Arrays.asList("normal", "enhancement", "major", "trivial", "critical", "minor", "blocker")
  val statusValues = util.Arrays.asList("CLOSED", "RESOLVED", "VERIFIED", "trivial", "critical", "minor")

  // Let's define the schema of the data that we want to import
  // The order in which columns are defined here should match
  // the order in which they appear in the input data
  val inputDataSchema: Schema = new Schema.Builder()
    .addColumnInteger("row_id")
    .addColumnInteger("assigned_to")
    .addColumnInteger("bug_id")
    .addColumnCategorical("bug_severity", severityValues)
    .addColumnCategorical("bug_status", statusValues)
    .addColumnInteger("component_id")
    .addColumnTime("creation_ts", TimeZone.getDefault)
    .addColumnTime("delta_ts", TimeZone.getDefault)
    .addColumnInteger("product_id")
    .addColumnCategorical("resolution", util.Arrays.asList("FIXED"))
    .addColumnsString("short_desc", "original_text", "text")
    .addColumnInteger("class")
    .build()

  val filteredDataSchema: Schema = new Schema.Builder()
    .addColumnsString("text")
    .addColumnCategorical("bug_severity", severityValues)
    .addColumnInteger("component_id")
    .addColumnInteger("product_id")
    .addColumnInteger("class")
    .build()

  val log = LoggerFactory.getLogger(Classifier.getClass)

  def main(args: Array[String]): Unit = {

    // Initialize jCommander
    new JCommander(Args, args.toArray: _*)

    log.debug(s"Resource folder is ${Args.resourceFolder}")
    log.debug(s"Input file is ${Args.inputFileName}")
    log.debug(s"Compute embeddings is ${Args.computeEmbeddings}")
    log.debug(s"Number of embedding epochs is ${Args.epochsForEmbeddings}")

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
      //            .filter(new ConditionFilter(new IntegerColumnCondition("class", ConditionOp.GreaterOrEqual, 2)))
      //.filter(new ConditionFilter(new IntegerColumnCondition("component_id", ConditionOp.NotEqual, 128)))
      .removeAllColumnsExceptFor("text", "bug_severity", "component_id", "product_id", "class")
      .reorderColumns("text", "bug_severity", "component_id", "product_id", "class")
      //.removeAllColumnsExceptFor("text", "class")
      //.reorderColumns("text", "class")
      .build()

    // After executing all of these operations, we have a new and different schema:
    val outputSchema: Schema = filterColumnsTransform.getFinalSchema

    log.info("\n\n\nSchema after transforming data:")
    log.info(s"$outputSchema")


    //=====================================================================
    //            Step 2.b: Transform
    //=====================================================================
    val directory: String = Args.resourceFolder + Args.inputFileName
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
      new java.util.concurrent.ConcurrentHashMap[java.lang.Integer, java.lang.String]()
    val distinctClasses = mutable.Set.empty[Int]

    val descs = filteredData.collect().map {
      writables =>
        if (Try(writables.get(2).toInt).isSuccess) components.put(writables.get(2).toInt, writables.get(2).toString)
        if (Try(writables.get(3).toInt).isSuccess) products.put(writables.get(3).toInt, writables.get(3).toString)
        if (Try(writables.last.toInt).isSuccess) distinctClasses += writables.last.toInt
        writables.get(0).toString
    }

    distinctClasses.zipWithIndex map { classWithIndex =>
      classes.put(classWithIndex._1, classWithIndex._2.toString)
    }

    //=====================================================================
    //            PARAGRAPH VECTOR !!!
    //=====================================================================

    log.info("Build Embedded Vectors ....")

    // build a iterator for our dataset
    val ptvIterator = new CollectionSentenceIterator(descs.flatMap(_.split("\\. ")).toList)
    //
    val tokenizerFactory = new DefaultTokenizerFactory
    tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor)

    val paragraphVectors = if (Args.computeEmbeddings) {

      // ParagraphVectors training configuration
      val _paragraphVectors = new ParagraphVectors.Builder()
        .learningRate(0.025).minLearningRate(0.001)
        .batchSize(2500).epochs(Args.epochsForEmbeddings)
        .iterate(ptvIterator).trainWordVectors(true)
        .tokenizerFactory(tokenizerFactory).build

      // Start model training
      _paragraphVectors.fit()

      log.info("Save vectors....")
      WordVectorSmartSerializer.writeParagraphVectors(_paragraphVectors, Args.resourceFolder + modelName)
      _paragraphVectors

    } else {

      val _paragraphVectors = WordVectorSmartSerializer.readParagraphVectors(new File(Args.resourceFolder + modelName))
      _paragraphVectors.setTokenizerFactory(tokenizerFactory)
      _paragraphVectors

    }

    log.info("Embedded Vectors OK ....")

    //=====================================================================
    //            Step 3.a: More transformations
    //=====================================================================
    val numericToCategoricalTransform: TransformProcess = new TransformProcess.Builder(filteredDataSchema)
      .integerToCategorical("component_id", components)
      .integerToCategorical("product_id", products)
      .categoricalToOneHot("bug_severity")
      .categoricalToOneHot("component_id")
      .categoricalToOneHot("product_id")
      .integerToCategorical("class", classes)
      .build()

    //=====================================================================
    //            Step 3.b: Transform
    //=====================================================================
    //Now, let's execute the transforms we defined earlier:
    val transformedData: RDD[util.List[Writable]] = SparkTransformExecutor.execute(filteredData, numericToCategoricalTransform)

    val possibleLabels: Int = classes.size()

    val data = transformedData.collect().toList.collect { case row if row.head.toString.nonEmpty =>
      seqAsJavaList(paragraphVectors.inferVector(row.head.toString).
        data().asDouble().map(new DoubleWritable(_)) ++ row.drop(1))
    }

    val featureSpaceSize = paragraphVectors.getLayerSize + components.size() + products.size() + severityValues.size()

    val batchSize = 100
    val averagingFrequency = 5

    val outputNum = possibleLabels
    val iterations = 1000

    val layer1width = 50
    val learningRate = 0.0018
    val activation = "softsign"

    val activation_end = "softmax"

    val (_trainingData, _testData) = data.splitAt(9 * data.size / 10)

    // train data
    val trainRecordReader = new CollectionRecordReader(_trainingData)
    val trainIterator: DataSetIterator =
      new RecordReaderDataSetIterator(trainRecordReader, Math.min(_trainingData.size, batchSize), featureSpaceSize, possibleLabels)

    // test data
    val testRecordReader = new CollectionRecordReader(_testData)
    val testIterator: DataSetIterator =
      new RecordReaderDataSetIterator(testRecordReader, _testData.length, featureSpaceSize, possibleLabels)

    log.info("Build model....")
    log.info(s"Number of iterations $iterations")

    //Set up network configuration
    val rnn_conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .iterations(iterations)
      .updater(Updater.RMSPROP)
      .regularization(true).l2(1e-5)
      .weightInit(WeightInit.XAVIER)
      .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
      .learningRate(learningRate)
      .list
      .layer(0, new GravesLSTM.Builder().name("GravesLSTM")
        .nIn(featureSpaceSize).nOut(layer1width)
        .activation(activation)
        .dropOut(0.5)
        .build())
      .layer(1, new RnnOutputLayer.Builder().name("RnnOutputLayer")
        .activation(activation_end)
        .nIn(layer1width).nOut(outputNum)
        .build())
      .pretrain(false).backprop(true).build

    val seed = 12345

    val deep_conf = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .iterations(iterations)
      .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
      .list()
      .layer(0, new RBM.Builder().nIn(featureSpaceSize).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(1, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(2, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(3, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(4, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(5, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(6, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(7, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(8, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(9, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(10, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(11, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(12, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(13, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(14, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(15, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(16, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(17, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(18, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(19, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(20, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(21, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(22, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(23, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(24, new RBM.Builder().nIn(50).nOut(50).lossFunction(LossFunctions.LossFunction.MCXENT).build())
      .layer(25, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation("sigmoid").nIn(50).nOut(outputNum).build())
      .pretrain(true).backprop(true)
      .build()

    log.info(s"Network configuration ${rnn_conf.toString}")

    val tm = new ParameterAveragingTrainingMaster.Builder(batchSize)
      .averagingFrequency(averagingFrequency)
      .workerPrefetchNumBatches(2)
      .batchSizePerWorker(batchSize)
      .build()

    //Create the Spark network
    val sparkNet = new SparkDl4jMultiLayer(sc, rnn_conf, tm)

    //Execute training:
    val numEpochs = 150

    //Perform evaluation (distributed)
    val testData = sc.parallelize(testIterator.toList)

    val trainingData = sc.parallelize(trainIterator.toList)
    (1 to numEpochs) foreach { i =>
      sparkNet.fit(trainingData)
      log.info(s"Completed Epoch $i")

      val evaluationTrain: SmartEvaluation = new SmartEvaluation(sparkNet.evaluate(trainingData))
      log.info("***** Evaluation TRAIN DATA *****")
      log.info(evaluationTrain.stats(false, false))

      val evaluationTest: SmartEvaluation = new SmartEvaluation(sparkNet.evaluate(testData))
      log.info("***** Evaluation TEST DATA *****")
      log.info(evaluationTest.stats(false, false))
    }

    //Delete the temp training files, now that we are done with them
    tm.deleteTempFiles(sc)

  }

}
