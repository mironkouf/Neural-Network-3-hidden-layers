// Nefeli Katsilerou AM:4385
// Myron Koufopoulos AM:4398
package exercise1;

import java.util.ArrayList;

public class Neuron {
	
	static double minWeightValue;
	static double maxWeightValue;
	ArrayList<Double> weights;
	ArrayList<Double> cache_weights ;// for back-prop
    double bias;
    double cache_bias; // for back-prop
    double value = 0;
	
	// Constructor for the input neurons
	public Neuron(double value){
		this.value = value;
	}
	
	// Constructor for the hidden / output neurons
    public Neuron(ArrayList<Double> weights, double bias,ArrayList<Double> cache_weights,double cache_bias){
        this.weights = weights;
        this.bias = bias;
        this.cache_weights = cache_weights;
        this.cache_bias = cache_bias;
    }
	
	public static void setRangeWeights(double minWeight,double maxWeight) {
		minWeightValue = minWeight;
		maxWeightValue = maxWeight;
	}
	
	// Function used at the end of the backprop to switch the calculated value in the
    // cache weight in the weights
    public void update_weight() {
    	this.weights = this.cache_weights;
    }
    public void update_bias() {
    	this.bias = this.cache_bias;
    }
}
