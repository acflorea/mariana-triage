package dr.acf.experiments

import java.io.{File, FileInputStream, InputStream}

import org.datavec.api.conf
import org.datavec.api.conf.Configuration
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.InputStreamInputSplit
import org.datavec.api.util.ClassPathResource

/**
  * Created by acflorea on 15/10/2016.
  */
object ParagraphToVec {

  def main(args: Array[String]): Unit = {

    // Load training file
    // assigned_to	bug_id	bug_severity	bug_status
    // component_id	creation_ts	delta_ts	product_id	resolution
    // short_desc original_text text
    val file: File = new ClassPathResource("netbeansbugs.csv").getFile

    val inputSplit = new InputStreamInputSplit(new FileInputStream(file))

    val recordReader = new CSVRecordReader(1, CSVRecordReader.DEFAULT_DELIMITER)
    recordReader.initialize(new Configuration(), inputSplit)

    recordReader.next()
    recordReader.next()

  }

}
