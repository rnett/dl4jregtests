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
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.BernoulliReconstructionDistribution;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

public class VaeMNISTAnomaly {

    public static void main(String[] args) throws IOException {
        int minibatchSize = 128;
        int rngSeed = 12345;
        int nEpochs = 5;                    //Total number of training epochs
        int reconstructionNumSamples = 16;  //Reconstruction probabilities are estimated using Monte-Carlo techniques; see An & Cho for details

        //MNIST data for training
        DataSetIterator trainIter = new MnistDataSetIterator(minibatchSize, true, rngSeed);

        //Neural net configuration
        Nd4j.getRandom().setSeed(rngSeed);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngSeed)
            .updater(new Adam(1e-3))
            .weightInit(WeightInit.XAVIER)
            .l2(1e-4)
            .list()
            .layer(new VariationalAutoencoder.Builder()
                .activation(Activation.LEAKYRELU)
                .encoderLayerSizes(256, 256)                    //2 encoder layers, each of size 256
                .decoderLayerSizes(256, 256)                    //2 decoder layers, each of size 256
                .pzxActivationFunction(Activation.IDENTITY)     //p(z|data) activation function
                //Bernoulli reconstruction distribution + sigmoid activation - for modelling binary data (or data in range 0 to 1)
                .reconstructionDistribution(new BernoulliReconstructionDistribution(Activation.SIGMOID))
                .nIn(28 * 28)                                   //Input size: 28x28
                .nOut(32)                                       //Size of the latent variable space: p(z|x) - 32 values
                .build())
            .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        net.save(new File("output/VaeMNISTAnomaly_100b4.bin"));
        Nd4j.getRandom().setSeed(12345);
        INDArray input = Nd4j.rand(3, 28*28);
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/VaeMNISTAnomaly_Input_100b4.bin")))) {
            Nd4j.write(input, dos);
        }
        INDArray output = net.output(input);
        try (DataOutputStream dos = new DataOutputStream(
                new FileOutputStream(new File("output/VaeMNISTAnomaly_Output_100b4.bin")))) {
            Nd4j.write(output, dos);
        }
    }
}