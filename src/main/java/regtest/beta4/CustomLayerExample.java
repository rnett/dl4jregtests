/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package regtest.beta4;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CustomLayerExample {

    static{
        //Double precision for the gradient checks. See comments in the doGradientCheck() method
        // See also http://nd4j.org/userguide.html#miscdatatype
        Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    }

    public static void main(String[] args) throws IOException {int nIn = 5;
        int nOut = 8;

        //Let's create a network with our custom layer

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()

                .updater( new RmsProp(0.95))
                .weightInit(WeightInit.XAVIER)
                .l2(0.03)
                .list()
                .layer(0, new DenseLayer.Builder().activation(Activation.TANH).nIn(nIn).nOut(6).build())     //Standard DenseLayer
                .layer(1, new CustomLayer.Builder()
                        .activation(Activation.TANH)                                                    //Property inherited from FeedForwardLayer
                        .secondActivationFunction(Activation.SIGMOID)                                   //Custom property we defined for our layer
                        .nIn(6).nOut(7)                                                                 //nIn and nOut also inherited from FeedForwardLayer
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)                //Standard OutputLayer
                        .activation(Activation.SOFTMAX).nIn(7).nOut(nOut).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();

        net.save(new File("output/CustomLayerExample_100b4.bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(new int[]{3, nIn});
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/CustomLayerExample_Input_100b4.bin")))) {
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input);
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/CustomLayerExample_Output_100b4.bin")))) {
            Nd4j.write(output, dos);
        }
    }

}