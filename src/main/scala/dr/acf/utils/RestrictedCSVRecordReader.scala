package dr.acf.utils

import java.util

import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.transform.categorical.CategoricalToOneHotTransform
import org.datavec.api.writable.Writable

import scala.collection.JavaConversions._

/**
  * Created by acflorea on 16/10/2016.
  */
class RestrictedCSVRecordReader(skipNumLines: Int, delimiter: String, columnsToKeep: Seq[Int])
  extends CSVRecordReader(skipNumLines: Int, delimiter: String) {

  val catToOH = new CategoricalToOneHotTransform("")

  override def next(): util.List[Writable] = {
    val entireLine = super.next()
    catToOH.map(List(entireLine(5), entireLine(13)))
  }

}
