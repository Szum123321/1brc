#!/bin/sh
#
#  Copyright 2023 The original authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


#JAVA_OPTS=" -XX:+UnlockDiagnosticVMOptions -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 -XX:InlineSmallCode=10000 -XX:FreqInlineSize=10000 -XX:PrintAssemblyOptions=intel -XX:+UnlockExperimentalVMOptions -XX:+TrustFinalNonStaticFields -XX:+UseTransparentHugePages -verbose:gc --enable-preview -Xms2g -Xmx2g --add-modules=jdk.incubator.vector --add-opens java.base/jdk.internal.foreign=ALL-UNNAMED --add-exports java.base/jdk.internal.foreign=ALL-UNNAMED --enable-native-access=ALL-UNNAMED"
#JAVA_OPTS="--enable-preview -Xms2g -Xmx2g --add-modules=jdk.incubator.vector --add-opens java.base/jdk.internal.foreign=ALL-UNNAMED --add-exports java.base/jdk.internal.foreign=ALL-UNNAMED --enable-native-access=ALL-UNNAMED"
JAVA_OPTS="-Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 -XX:InlineSmallCode=10000 -XX:FreqInlineSize=10000 -XX:+UnlockExperimentalVMOptions -XX:+TrustFinalNonStaticFields -XX:+UseTransparentHugePages -verbose:gc --enable-preview -Xms4g -Xmx4g --add-modules=jdk.incubator.vector --add-opens java.base/jdk.internal.foreign=ALL-UNNAMED --add-exports java.base/jdk.internal.foreign=ALL-UNNAMED --enable-native-access=ALL-UNNAMED"

time java $JAVA_OPTS --class-path target/average-1.0.0-SNAPSHOT.jar dev.morling.onebrc.CalculateAverage_Szum123321
