name := "mariana-triage"

version := "1.0"

scalaVersion := "2.11.8"

resolvers += "Local Maven Repository" at "file://" + "/Users/acflorea" + "/.m2/repository"

libraryDependencies ++= Seq(
  "org.deeplearning4j" % "deeplearning4j-core" % "0.6.0"
  , "org.deeplearning4j" % "deeplearning4j-nlp" % "0.6.0" excludeAll ExclusionRule(organization = "ch.qos.logback")
  , "org.nd4j" % "nd4j-native-platform" % "0.6.1-SNAPSHOT"
  , "org.datavec" % "datavec-api" % "0.6.0"
  , "org.datavec" %% "datavec-spark" % "0.6.0" excludeAll ExclusionRule(artifact = "slf4j-log4j12")
)

dependencyOverrides ++= Set(
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4"
)
