package dr.acf.utils

import java.lang
import java.text.DecimalFormat
import org.deeplearning4j.eval.Evaluation
import scala.collection.JavaConversions._

/**
  * Created by acflorea on 30/10/2016.
  * TODO - make it more scalish
  */
class SmartEvaluation(eval: Evaluation) {
  /**
    * Method to obtain the classification report as a String
    *
    * @param suppressWarnings whether or not to output warnings related to the evaluation results
    * @return A (multi-line) String with accuracy, precision, recall, f1 score etc
    */
  def stats(suppressWarnings: Boolean): String = {
    var actual: String = null
    var expected: String = null
    val builder: StringBuilder = new StringBuilder().append("\n")
    val warnings: StringBuilder = new StringBuilder
    val classes = eval.getConfusionMatrix.getClasses
    import scala.collection.JavaConversions._
    for (clazz <- classes) {
      actual = eval.getClassLabel(clazz)
      //Output confusion matrix
      import scala.collection.JavaConversions._
      for (clazz2 <- classes) {
        val count: Int = eval.getConfusionMatrix.getCount(clazz, clazz2)
        if (count != 0) {
          expected = eval.getClassLabel(clazz2)
          builder.append(String.format("Examples labeled as %s classified by model as %s: %d times%n", actual, expected, count:Integer))
        }
      }
      //Output possible warnings regarding precision/recall calculation
      if (!suppressWarnings && eval.truePositives.get(clazz) == 0) {
        if (eval.falsePositives.get(clazz) == 0) warnings.append(String.format("Warning: class %s was never predicted by the model. This class was excluded from the average precision%n", actual))
        if (eval.falseNegatives.get(clazz) == 0) warnings.append(String.format("Warning: class %s has never appeared as a true label. This class was excluded from the average recall%n", actual))
      }
    }
    builder.append("\n")
    builder.append(warnings)
    val df: DecimalFormat = new DecimalFormat("#.####")
    val acc: Double = eval.accuracy
    val prec: Double = eval.precision
    val rec: Double = eval.recall
    val f1: Double = eval.f1
    val wprec: Double = wprecision
    val wrec: Double = wrecall
    val _wf1: Double = wf1
    builder.append("\n==========================Scores========================================")
    builder.append("\n Accuracy:  ").append(format(df, acc))
    builder.append("\n Precision: ").append(format(df, prec))
    builder.append("\n Recall:    ").append(format(df, rec))
    builder.append("\n F1 Score:  ").append(format(df, f1))
    builder.append("\n WPrecision: ").append(format(df, wprec))
    builder.append("\n WRecall:    ").append(format(df, wrec))
    builder.append("\n WF1 Score:  ").append(format(df, wf1))
    builder.append("\n========================================================================")
    builder.toString
  }


  private def format(f: DecimalFormat, num: Double): String = {
    if (lang.Double.isNaN(num) || lang.Double.isInfinite(num)) return String.valueOf(num)
    f.format(num)
  }

  /**
    * Precision based on guesses so far
    * Takes into account all known classes and outputs average precision across all of them
    *
    * @return the total precision based on guesses so far
    */
  def wprecision: Double = {
    var precisionAcc: Double = 0.0
    var classCount: Int = 0
    import scala.collection.JavaConversions._
    for (classLabel <- eval.getConfusionMatrix.getClasses) {
      val precision: Double = eval.precision(classLabel, -1)
      if (precision != -1) {
        precisionAcc += eval.precision(classLabel) * eval.classCount(classLabel)
        classCount += eval.classCount(classLabel)
      }
    }
    precisionAcc / classCount.toDouble
  }

  /**
    * Recall based on guesses so far
    * Takes into account all known classes and outputs average recall across all of them
    *
    * @return the recall for the outcomes
    */
  def wrecall: Double = {
    var recallAcc: Double = 0.0
    var classCount: Int = 0
    import scala.collection.JavaConversions._
    for (classLabel <- eval.getConfusionMatrix.getClasses()) {
      val recall: Double = eval.recall(classLabel, -1.0)
      if (recall != -1.0) {
        recallAcc += eval.recall(classLabel) * eval.classCount(classLabel)
        classCount += eval.classCount(classLabel)
      }
    }
    recallAcc / classCount.toDouble
  }

  /**
    * TP: true positive
    * FP: False Positive
    * FN: False Negative
    * F1 score: 2 * TP / (2TP + FP + FN)
    *
    * @return the f1 score or harmonic mean based on current guesses
    */
  def wf1: Double = {
    var f1Acc: Double = 0.0
    var classCount: Int = 0
    for (classLabel <- eval.getConfusionMatrix.getClasses()) {
      val precision: Double = eval.precision(classLabel, -1)
      val recall: Double = eval.recall(classLabel, -1.0)
      if (precision != -1 && recall != -1.0) {
        f1Acc += 2.0 * (precision * recall / (precision + recall))
        classCount += eval.classCount(classLabel)
      }
    }
    wf1 / classCount.toDouble
  }
}
