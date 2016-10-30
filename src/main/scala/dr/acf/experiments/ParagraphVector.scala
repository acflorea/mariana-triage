package dr.acf.experiments

import java.io.File
import java.util
import java.util.TimeZone

import dr.acf.utils.{SparkOps, WordVectorSmartSerializer}
import org.apache.spark.rdd.RDD
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.TransformProcess
import org.datavec.api.transform.schema.Schema
import org.datavec.api.util.ClassPathResource
import org.datavec.api.writable.{DoubleWritable, Writable}
import org.datavec.spark.transform.SparkTransformExecutor
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.layers.{GravesLSTM, RnnOutputLayer}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.slf4j.LoggerFactory

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.util.Try

/**
  * Created by acflorea on 15/10/2016.
  */
object ParagraphVector extends SparkOps {

  val inputFileName = "netbeansbugs_filtered.csv"
  val computeEmbeddings = false
  val epochsForEmbeddings = 20
  val modelName = if (epochsForEmbeddings < 10)
    s"netbeans_0$epochsForEmbeddings.model"
  else
    s"netbeans_$epochsForEmbeddings.model"

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
    .addColumnsString("original_text")
    .addColumnCategorical("bug_severity", severityValues)
    .addColumnInteger("component_id")
    .addColumnInteger("product_id")
    .addColumnInteger("class")
    .build()

  val log = LoggerFactory.getLogger(Classifier.getClass)

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
      //            .filter(new ConditionFilter(new IntegerColumnCondition("class", ConditionOp.GreaterOrEqual, 2)))
      //.filter(new ConditionFilter(new IntegerColumnCondition("component_id", ConditionOp.NotEqual, 128)))
      .removeAllColumnsExceptFor("text", "bug_severity", "component_id", "product_id", "class")
      .reorderColumns("text", "bug_severity", "component_id", "product_id", "class")
      .build()

    // After executing all of these operations, we have a new and different schema:
    val outputSchema: Schema = filterColumnsTransform.getFinalSchema

    log.info("\n\n\nSchema after transforming data:")
    log.info(s"$outputSchema")


    //=====================================================================
    //            Step 2.b: Transform
    //=====================================================================
    val directory: String = new ClassPathResource(inputFileName).getFile.getPath
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

    val paragraphVectors = if (computeEmbeddings) {

      // ParagraphVectors training configuration
      val _paragraphVectors = new ParagraphVectors.Builder()
        .learningRate(0.025).minLearningRate(0.001)
        .batchSize(2500).epochs(epochsForEmbeddings)
        .iterate(ptvIterator).trainWordVectors(true)
        .tokenizerFactory(tokenizerFactory).build

      // Start model training
      _paragraphVectors.fit()

      log.info("Save vectors....")
      WordVectorSmartSerializer.writeParagraphVectors(_paragraphVectors, modelName)
      _paragraphVectors

    } else {

      val _paragraphVectors = WordVectorSmartSerializer.readParagraphVectors(new File(modelName))
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

    val layer1width = 100
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
        .build())
      .layer(1, new RnnOutputLayer.Builder().name("RnnOutputLayer")
        .activation(activation_end)
        .nIn(layer1width).nOut(outputNum)
        .build())
      .pretrain(false).backprop(true).build

    //run the model
    val model: MultiLayerNetwork = new MultiLayerNetwork(rnn_conf)

    val tm = new ParameterAveragingTrainingMaster.Builder(batchSize)
      .averagingFrequency(averagingFrequency)
      .workerPrefetchNumBatches(2)
      .batchSizePerWorker(batchSize)
      .build()

    //Create the Spark network
    val sparkNet = new SparkDl4jMultiLayer(sc, rnn_conf, tm)

    //Execute training:
    val numEpochs = 15

    //Perform evaluation (distributed)
    val testData = sc.parallelize(testIterator.toList)

    val trainingData = sc.parallelize(trainIterator.toList)
    (1 to numEpochs) foreach { i =>
      sparkNet.fit(trainingData)
      log.info("Completed Epoch {}", i);
      val evaluation: Evaluation = sparkNet.evaluate(testData)
      log.info("***** Evaluation *****")
      log.info(evaluation.stats)
    }

    //Delete the temp training files, now that we are done with them
    tm.deleteTempFiles(sc)

  }

}
