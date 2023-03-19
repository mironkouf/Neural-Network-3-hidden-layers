// Nefeli Katsilerou AM:4385
// Myron Koufopoulos AM:4398
package exercise1;

import java.util.ArrayList;

public class Layer {
	public ArrayList<Neuron> neurons = new ArrayList<Neuron>();
	
	// Constructor for input layer for every neuron
	public Layer(ArrayList<Double> input) { 
		if (input.get(0) != null){ 
			for(int i = 0; i < input.size(); i++) {
				neurons.add(i,new Neuron(input.get(i)));
			}
		}		
	}
	
	// Constructor for hidden-output layer for every neuron
	public Layer(int inNeurons,int numberNeurons) {
		/*
		 * gia kathe neurwna sto layer pou briskomaste arxikopoioume ta barh
		 * twn neurwnwn tou prohgoumenou layer pou erxontai sto neurwna auto 
		 */
		for(int i = 0; i < numberNeurons; i++) {
			ArrayList<Double> weights = new ArrayList<Double>();
			ArrayList<Double> cache_weights = new ArrayList<Double>();
			double temp=0.0;
			for(int j = 0; j < inNeurons; j++) {
				weights.add(Statistics.RandomDouble(Neuron.minWeightValue, Neuron.maxWeightValue));
				cache_weights.add(temp);
			}
			double bias = Statistics.RandomDouble(-1, 1);
			neurons.add(i,new Neuron(weights,bias,cache_weights,temp));		
		}
	}
}
