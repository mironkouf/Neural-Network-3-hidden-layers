// Nefeli Katsilerou AM:4385
// Myron Koufopoulos AM:4398
package exercise1;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class MLP {
	// 1 erwthma 
		static final int d = 2; // inputs in MLP
		static final int K = 3; // number of categories
		static final int H1 = 12; // number of neuron in hidden layer 1
		static final int H2 = 10; // number of neuron in hidden layer 2
		static final int H3 = 4; // number of neuron in hidden layer 3
		static final int sigmoid = 0; //logistic
		static final int tanh = 1;
		static final int relu = 2;
		static final int linear = 3;	// ATTENTION !!!! only for output layer
		static final int activFunc = relu; // activation function
		static final int outputActivFunc=sigmoid; // only for output layer
		static final int N = 4000; // training data
		// 1 : stohastic gradient descent (seiriaki enhmerwsh)
		// 0<B<N : mini batch gradient descent (enhmerwsh ana omades/group)
		// N : batch gradient descent (omadikh enhmerwsh)
		static final int B = 1; // batch size
		static final int epochs =200;
		static final double n = 0.07; // learning rate
		static final double katwfli_termatismou = 0.5;
	
		// keeps the training data
		private static ArrayList<TrainData> trainingData = new ArrayList<TrainData>();
		private static ArrayList<TrainData> testData = new ArrayList<TrainData>();
		private static ArrayList<Integer> correctTestOutputs = new ArrayList<Integer>();
		private static ArrayList<Integer> keepSelectedOutputs = new ArrayList<Integer>();

		private static ArrayList<Layer> layers = new ArrayList<Layer>();
				
		public static void main(String[] args) {
			try {
	            // read data and store it in arraylist(trainingData)
	            createData("data_train.csv", N);
	            
	        } catch (IOException e){
	            e.printStackTrace();
	        }
			
			System.out.println("trainingDataSize: "+trainingData.size());
			
			// initialize Min-Max values for Weights in every Neuron
			Neuron.setRangeWeights(-1,1);
			
			// 1) initialize layers and the neurons inside, 
			// 2) initialize random weights and bias for every neuron
			initializeLayers();
			
			train(n);
			//TEST
			try {
	            // read data and store it in arraylist(trainingData)
	            createTest("data_test.csv", N);    
	        } catch (IOException e){
	            e.printStackTrace();
	        }
			test(epochs);
			createCsvOutput("samplePredictions.csv");
		}
		// This function sums up all the gradient connecting a given neuron in a given layer
	    public static double sumGradient(int neuron_index,int layer_index) {
	    	double gradient_sum = 0;
	    	Layer current_layer = layers.get(layer_index);
	    	for(int i = 0; i < current_layer.neurons.size(); i++) {
	    		Neuron current_neuron = current_layer.neurons.get(i);
	    		// with bias we mean d(delta) of that neuron
	    		gradient_sum += current_neuron.weights.get(neuron_index)*current_neuron.bias;
	    	}
	    	return gradient_sum;
	    }
	    
		public static void backward(double learning_rate,TrainData data) {
	    	int number_layers = layers.size();
	    	int out_index = number_layers-1;
	    	double output,target,correction,derivative,delta,outputNeuron,sum,deltaNeuron;
	    	// Update the output layer 
	    	// For each output neuron
	    	for(int i = 0; i < layers.get(out_index).neurons.size(); i++) {
		    	// back-prop for the output Neuron
	    		output = layers.get(out_index).neurons.get(i).value;
	    		target = data.expectedOutput.get(i);
	    		correction = output-target;
	    		derivative = Statistics.derivativeActivFunc(output, outputActivFunc);
	    		delta = correction*derivative;
	    		layers.get(out_index).neurons.get(i).bias = delta;
	    		//de/dw(ij) (weights) - de/dw(i0) (bias)
	    		//gradient
	    		for(int j = 0; j < layers.get(out_index).neurons.get(i).weights.size(); j++) { // weights.length == sum of neurons in the previous layer
	    			double previous_output = layers.get(out_index-1).neurons.get(j).value;
	    			double partial_deriv = delta*previous_output;
	    			double temp = layers.get(out_index).neurons.get(i).weights.get(j) - (learning_rate*partial_deriv);
					layers.get(out_index).neurons.get(i).cache_weights.set(j,temp);
	    			layers.get(out_index).neurons.get(i).cache_bias = layers.get(out_index).neurons.get(i).bias - (learning_rate*delta);
	    		}
	    	}
	    	//Update all the subsequent hidden layers
	    	for(int i = out_index-1; i > 0; i--) {
	    		// For all neurons in that layers
	    		for(int j = 0; j < layers.get(i).neurons.size(); j++) {
	    			outputNeuron = layers.get(i).neurons.get(j).value; // u_i of the activation functions
	    			/*
	    			 * epeidh dn mporoume na paroume ta delta tou epomenou layer tou kathe neurwna
	    			 * kaloume thn sumGradient h opoia upologizei gia ton neurwna pou briskomaste
	    			 * ola ta delta kai ta barh pou exoun oloi oi neurwnes tou epomenou epipedou
	    			 * (px d(1)_1 = activFunc*[d(1)_2*w(11)_2+d(2)_2*w(21)_2])
	    			 * (sum_gradient = [d(1)_2*w(11)_2+d(2)_2*w(21)_2])
	    			 */
	    			sum = sumGradient(j,i+1);
	    			derivative = Statistics.derivativeActivFunc(outputNeuron,activFunc);
	    			deltaNeuron = sum*derivative;
	    			layers.get(i).neurons.get(j).bias = deltaNeuron;
	    			//gradient
	    			for(int k = 0; k < layers.get(i).neurons.get(j).weights.size(); k++) {
	    				double previous_output = layers.get(i-1).neurons.get(k).value;
	    				double partial_deriv = deltaNeuron*previous_output;
	    				double temp = layers.get(i).neurons.get(j).weights.get(k) - (learning_rate*partial_deriv);
	    				layers.get(i).neurons.get(j).cache_weights.set(k,temp);
	    				layers.get(i).neurons.get(j).cache_bias = layers.get(i).neurons.get(j).bias - (learning_rate*deltaNeuron);
	    			}
	    		}
	    	}
		}
		public static double calcError(int idxTrain) {
			double output=0,target=0;
			double error=0;
			int number_layers = layers.size();
			int out_index = number_layers-1;
			for(int k = 0; k < layers.get(out_index).neurons.size(); k++) {
    			output=layers.get(out_index).neurons.get(k).value; 
		        target=trainingData.get(idxTrain).expectedOutput.get(k);
		        //System.out.println("output=" + output + " | target=" + target);
		        error += Statistics.squaredError(output,target);
			}
			return error;
		}
		
		// This function is used to train being forward and backward.
	    public static void train(double learning_rate) {
	    	double previous_total_error=0;
	    	double MSE=0;
	    	int epoch=0;
	    	while(true) {
	    		MSE=0;
	    		for(int j = 0; j < trainingData.size(); j++) {
	    			forward(trainingData.get(j).data);
	    			backward(learning_rate,trainingData.get(j));
	    			if(j % B == 0) {
	    				update();
					}
	    			MSE += calcError(j);
		        }
	    		MSE = MSE/2;
	    		System.out.println("epoch error (MSE) = " + MSE );
	    		System.out.println("Finished epoch-" + epoch);
	    		System.out.println("-----------------------------------------");
	    		epoch++;
				if(epoch > epochs || Math.abs(previous_total_error - MSE)< 0.1){
	    			//System.out.println("epoch error (MSE) = " + MSE );
		    		//System.out.println("Finished epoch-" + epoch);
		    		//System.out.println("-----------------------------------------");
	                break;
	            }
	            previous_total_error = MSE;
	    	}
	    }
	    public static void update() {
	    	// update all the weights (one by one) and biases
	    	for(int i = 1; i< layers.size();i++) {
	    		for(int j = 0; j < layers.get(i).neurons.size();j++) {
	    			for(int k = 0; k < layers.get(i).neurons.get(j).weights.size(); k++) {
	    				layers.get(i).neurons.get(j).weights.set(k,layers.get(i).neurons.get(j).cache_weights.get(k));
	    			}
	    			layers.get(i).neurons.get(j).update_bias();
	    		}
	    	}
	    }
		public static double[] forward (ArrayList<Double> inputs) {
			int number_layers = layers.size();
	    	int out_index = number_layers-1;
	    	double[] networkOutput = new double[K];
	    	// First bring the inputs into the input layer layers[0]
			// replace null with the inputs(x,y values)
			layers.set(0, new Layer(inputs));
	    	
	    	// take all the layers except the H(0) and H(N+1) output layer
	        for(int i = 1; i < layers.size(); i++) { // for each layer
	        	for(int j = 0; j < layers.get(i).neurons.size(); j++) { // for each neuron
	        		double u_i = 0;
	        		for(int k = 0; k < layers.get(i-1).neurons.size(); k++) { // for every previous layer neuron
	        			u_i += layers.get(i-1).neurons.get(k).value * layers.get(i).neurons.get(j).weights.get(k);
	        		}
	        		u_i += layers.get(i).neurons.get(j).bias;
	        		if (i != out_index) { // we are in the hidden layers
	        			layers.get(i).neurons.get(j).value = Statistics.activFunc(u_i,activFunc);
	        		}
	        		else {	// we are in the output layer
	        			layers.get(i).neurons.get(j).value = Statistics.activFunc(u_i,outputActivFunc);
	            		networkOutput[j]=layers.get(i).neurons.get(j).value;
	            		
	        		}
	        	}
	        }
	        return networkOutput;
	    }
		
		public static void initializeLayers() {
			// H(0): input layer, takes null values
			// During forward passing we will initialize every value separately
			ArrayList<Double> a = new ArrayList<Double>();
			a.add(null);
			layers.add(new Layer(a)); 
			layers.add(new Layer(d,H1)); // H(1)
			layers.add(new Layer(H1,H2)); // H(2)
			layers.add(new Layer(H2,H3)); // H(3)
			layers.add(new Layer(H3,K)); // H(output)
		}
		public static void createData(String csvFileName, Integer N) throws IOException{
			String row;
			int attrNames = 0;
			int count = 0;
	        try(BufferedReader csvReader = new BufferedReader(new FileReader(csvFileName))) {
	            if((row = csvReader.readLine()) != null){
	            	attrNames =3;
	            }

	            while ((row = csvReader.readLine()) != null && count <N) {	// read every line
	                String[] data = row.split(",");
	                ArrayList<Double> inputs = new ArrayList<Double>();
	                ArrayList<Double> outputs = new ArrayList<Double>();
	                if(attrNames == data.length) { // check for false inputs (Nan values)
	                    for (int i = 0; i < attrNames-1; i++) {	// read every column
	                        double intputVal = Double.parseDouble(data[i]);
	                        inputs.add(intputVal);
	                    }
	                    double outputVal = Double.parseDouble(data[attrNames-1]);
	                    for(double j=1; j<attrNames+1;j++) {
	                    	if (outputVal == j) outputs.add(1.0);
	                    	else outputs.add(0.0);
	                    }
	                } else{
	                    throw new IOException("Incorrectly formatted file.");
	                }  
	                trainingData.add(new TrainData(inputs,outputs));
	                count++;
	            }
	        }
		}
        public static void createTest(String csvFileName, Integer N) throws IOException{
			String row;
			int attrNames = 0;
			int count = 0;
	        try(BufferedReader csvReader = new BufferedReader(new FileReader(csvFileName))) {
	            if((row = csvReader.readLine()) != null){
	            	attrNames =3;
	            }
	            
	            while ((row = csvReader.readLine()) != null && count <N) {	// read every line
	                String[] data = row.split(",");
	                ArrayList<Double> inputs = new ArrayList<Double>();
	                ArrayList<Double> outputs = new ArrayList<Double>();
	                if(attrNames == data.length) { 
	                    for (int i = 0; i < attrNames-1; i++) {	// read every column
	                        double intputVal = Double.parseDouble(data[i]);
	                        inputs.add(intputVal);
	                    }
	                    double outputVal = Double.parseDouble(data[attrNames-1]);
	                    for(int j=1; j<attrNames+1;j++) {
	                    	if (outputVal == j) correctTestOutputs.add(j); // keep the position of the correct answer 1,2,3
	                    }
	                } else{
	                    throw new IOException("Incorrectly formatted file.");
	                }
	                testData.add(new TrainData(inputs,outputs));
	                count++;
	            }
	        }
        }
        public static void test(int epoch) {
	    	int correctCateg=0;
	    	double[] output;
	    	System.out.println("---------------TEST--------------");
			for(int j = 0; j < testData.size(); j++) {
				output = forward(testData.get(j).data);
				int position = keepMaxOutput(output);
				keepSelectedOutputs.add(position);
				if (position == correctTestOutputs.get(j))
					correctCateg++;
	    	}
			double percentage = (correctCateg/(double)N)*100;
			System.out.println("correct predictions percentage = "+percentage+"%");
			System.out.println("number of correct categories = "+correctCateg);
	    }
        public static int keepMaxOutput(double[] output) {
        	double max=0;
        	int position=-1;
        	for(int i = 0; i < layers.get(4).neurons.size(); i++) {
    			if (output[i] > max) {
    				max = output[i];
    				position=i+1; // because our categories are 1,2,3
    			}
			}
        	return position;
        }
        public static void createCsvOutput(String outputFileName){
            try(BufferedWriter csvWriter = new BufferedWriter(new FileWriter(outputFileName))) {
                csvWriter.write("X");
                csvWriter.write(",");
                // for centroids columns
                csvWriter.write("Y");
                csvWriter.write(",");
                csvWriter.write("prediction");
                csvWriter.write("\n");

                for(int i = 0; i < testData.size(); i++){
                    for(int j=0; j<2; j++){
                        csvWriter.write(String.valueOf(testData.get(i).data.get(j)));
                        csvWriter.write(",");
                    }
                    csvWriter.write(String.valueOf(keepSelectedOutputs.get(i)));
                    csvWriter.write("\n");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

}
