// Nefeli Katsilerou AM:4385
// Myron Koufopoulos AM:4398
package exercise1;

public class Statistics {
	// Get a random numbers between min and max
    public static double RandomDouble(double min, double max) {
        double num = min + Math.random() * (max - min);
        return num;
    }
    // Used for the backpropagation
    public static double squaredError(double output,double target) {
    	return (Math.pow((target-output),2));
    }
    
    // Sigmoid function
    public static double Sigmoid(double x) {
        return (1/(1+Math.pow(Math.E, -x)));
    }
    
    // Relu function
    public static double Relu(double x) {
        if (x > 0) return x;
        return 0;
    }
    // Tahn function
    public static double Tanh(double x) {
    	double numerator = (Math.pow(Math.E, x) - Math.pow(Math.E, -x));
    	double denominator = (Math.pow(Math.E, x) + Math.pow(Math.E, -x));
        return numerator/denominator;
    }
    
    // Derivative of the sigmoid function
    public static double derivativeActivFunc(double x,int activFunc) {
        if (activFunc == 0) // sigmoid
        	return Sigmoid(x)*(1-Sigmoid(x)); //TODO
        else if (activFunc == 1) // tanh
        	return 1-(Tanh(x)*Tanh(x));
        else if (activFunc == 2) {	// relu
        	if (x > 0) return 1;
        	return 0;
        }else 
        	return 1;
    }
    
    public static double activFunc(double x,int activFunc) {
        if (activFunc == 0) // sigmoid
        	return Sigmoid(x);
        else if (activFunc == 1) // tanh
        	return Tanh(x);
        else if (activFunc == 2)	// relu
        	return Relu(x);
        else
        	return x;
    }
    /*public static double derivativeOutputFunc(double z,int activFunc) {
    	double ez;
    	ez = (Math.pow(Math.E, z));
    	return ez;
    }*/
}
