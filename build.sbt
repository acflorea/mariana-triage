name := "mariana-triage"

version := "1.2"

packAutoSettings

scalaVersion := "2.10.6"
autoScalaLibrary := false

val dl4jVersion = "0.7.1"
val nd4jVersion = "0.7.1"
val datavecVersion = "0.7.1"
val sparkVersion = "1.6.2"
val hadoopVersion = "2.2.0"
val hadoopDependenciesScope = "provided"

val isALibrary = true

/*if it's a library the scope is "compile" since we want the transitive dependencies on the library
  otherwise we set up the scope to "provided" because those dependencies will be assembled in the "assembly"*/
lazy val assemblyDependenciesScope: String = if (isALibrary) "compile" else "provided"

val sparkExcludes =
  (moduleId: ModuleID) => moduleId.
    exclude("org.apache.hadoop", "hadoop-client").
    exclude("org.apache.hadoop", "hadoop-yarn-client").
    exclude("org.apache.hadoop", "hadoop-yarn-api").
    exclude("org.apache.hadoop", "hadoop-yarn-common").
    exclude("org.apache.hadoop", "hadoop-yarn-server-common").
    exclude("org.apache.hadoop", "hadoop-yarn-server-web-proxy")

val hadoopClientExcludes =
  (moduleId: ModuleID) => moduleId.
    exclude("org.slf4j", "slf4j-api").
    exclude("javax.servlet", "servlet-api")

val assemblyDependencies = (scope: String) => Seq(
  sparkExcludes("org.nd4j" % "nd4j-native-platform" % nd4jVersion % scope)
    exclude("com.fasterxml.jackson.core", "jackson-annotations")
    exclude("com.fasterxml.jackson.core", "jackson-core")
    exclude("com.fasterxml.jackson.core", "jackson-databind")
    exclude("com.fasterxml.jackson.dataformat", "jackson-dataformat-yaml"),
  sparkExcludes("org.deeplearning4j" %% "dl4j-spark" % dl4jVersion % scope)
    exclude("org.apache.spark", "*")
    exclude("com.fasterxml.jackson.core", "jackson-annotations")
    exclude("com.fasterxml.jackson.core", "jackson-core")
    exclude("com.fasterxml.jackson.core", "jackson-databind")
    exclude("com.fasterxml.jackson.dataformat", "jackson-dataformat-yaml"),
  sparkExcludes("org.deeplearning4j" % "deeplearning4j-nlp" % dl4jVersion % scope)
    exclude("org.apache.spark", "*")
    exclude("com.fasterxml.jackson.core", "jackson-annotations")
    exclude("com.fasterxml.jackson.core", "jackson-core")
    exclude("com.fasterxml.jackson.core", "jackson-databind")
    exclude("com.fasterxml.jackson.dataformat", "jackson-dataformat-yaml"),
  sparkExcludes("org.nd4j" %% "nd4j-kryo" % nd4jVersion % scope)
    exclude("com.esotericsoftware.kryo", "kryo"),
  sparkExcludes("org.nd4j" %% "nd4s" % nd4jVersion % scope),
  sparkExcludes("org.datavec" % "datavec-api" % datavecVersion % scope),
  sparkExcludes("org.datavec" %% "datavec-spark" % datavecVersion % scope)
    exclude("org.apache.spark", "*"),
  "com.fasterxml.jackson.core" % "jackson-annotations" % "2.4.4" % scope,
  "com.fasterxml.jackson.core" % "jackson-core" % "2.4.4" % scope,
  "com.fasterxml.jackson.core" % "jackson-databind" % "2.4.4" % scope,
  "com.fasterxml.jackson.dataformat" % "jackson-dataformat-yaml" % "2.4.4" % scope,
  "com.typesafe" % "config" % "1.2.1" % scope
)

libraryDependencies ++= Seq(
  sparkExcludes("org.apache.spark" %% "spark-core" % sparkVersion % hadoopDependenciesScope),
  sparkExcludes("org.apache.spark" %% "spark-sql" % sparkVersion % hadoopDependenciesScope),
  sparkExcludes("org.apache.spark" %% "spark-yarn" % sparkVersion % hadoopDependenciesScope),
  sparkExcludes("org.apache.spark" %% "spark-mllib" % sparkVersion % hadoopDependenciesScope),
  sparkExcludes("org.apache.spark" %% "spark-streaming" % sparkVersion % hadoopDependenciesScope),
  hadoopClientExcludes("org.apache.hadoop" % "hadoop-yarn-api" % hadoopVersion % hadoopDependenciesScope),
  hadoopClientExcludes("org.apache.hadoop" % "hadoop-yarn-client" % hadoopVersion % hadoopDependenciesScope),
  hadoopClientExcludes("org.apache.hadoop" % "hadoop-yarn-common" % hadoopVersion % hadoopDependenciesScope),
  hadoopClientExcludes("org.apache.hadoop" % "hadoop-yarn-applications-distributedshell" % hadoopVersion % hadoopDependenciesScope),
  hadoopClientExcludes("org.apache.hadoop" % "hadoop-yarn-server-web-proxy" % hadoopVersion % hadoopDependenciesScope),
  hadoopClientExcludes("org.apache.hadoop" % "hadoop-client" % hadoopVersion % hadoopDependenciesScope)
) ++ assemblyDependencies(assemblyDependenciesScope)
