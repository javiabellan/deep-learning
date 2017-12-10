/*******************************************************************
IMPORTANTE:
Descargar dataset MNIST (formato CSV) y poner en la misma carpeta.
TRAIN: http://www.pjreddie.com/media/files/mnist_train.csv
TEST:  http://www.pjreddie.com/media/files/mnist_test.csv
Cada fila representa un numero con su correspondiente imagen en pixels
numero, pixel(1,1), pixel(1,2), pixel(1,3), ... , pixel(28,28)
*******************************************************************/


#include <stdio.h>
#include <stdlib.h> // For rand() system()
#include <string.h>
#include <math.h>

//////////////////////////////////// detalles estructura

/* KERAS
Conv2D(32, size=(3, 3))
Conv2D(64, (3, 3), activation='relu')
MaxPooling2D(pool_size=(2, 2))
Dropout(0.25)
Flatten()
Dense(128, activation='relu')
Dropout(0.5)
Dense(num_classes, activation='softmax')
*/

struct kernel
{
	int size;
	//int height;
	//int widht;
	//int padding = 0;
	//int stride  = 1;
};

enum layerType {Conv2D, Pooling2D, Flatten, Dense};
struct layer
{
	int numInput;
	int numOutput;
	double* weights; //double weights[numInput+1][numOutput];
	// Tipo: [Conv2D, Pooling2D, Flatten, Dense]
	// dropuot;
	// activationFunction = [sigmoid, relu]
};



//--------------------------------- DEFINE TUS CAPAS Y CREA TU MODELO

// Data
int    NumExamples; //max is 60000 (see numExamplesInitial)
double dataInput[numExamplesInitial][height][width];


double input = [channel][height][width]
double conv1 = [channel][height][width]
double conv2 = [channel][height][width]




layer1.type    = CONV2D
layer1.input   = [1, 28, 28]
layer1.kernel  = [1, 3, 3]
layer1.numKer  = 32
layer1.output  = [32, 26, 26]

layer2.type    = CONV2D
layer2.input   = [32, 26, 26]
layer2.kernel  = [32, 3, 3]
layer1.numKer  = 64
layer2.output  = [64, 24, 24]

layer3.type    = POOLING
layer3.input   = [64, 24, 24]
layer3.reduce  = 2
layer3.output  = [64, 12, 12]

layer4.type    = FLATTEN
layer4.input   = [64, 12, 12]
layer4.output  = 64*12*12 //= 9216

layer5.type    = DENSE
layer5.input   = 9216
layer5.weights = [9216+1][128]
layer5.output  = 128

layer6.type    = DENSE
layer6.input   = 128
layer6.weights = [128+1][10]
layer6.output  = 10




layer capa1 = Conv2D(dataSize, 32, 3) // 32 kernels de 3x3
layer capa2 = Conv2D(capa1, 64, 3)   // 64 kernels de 3x3
layer capa3 = Pooling2D(capa2, 2)    // pooling de 2x2
layer capa4 = Flatten(capa3)     // ~numInput
layer capa5 = Dense(capa4, 128)  // ~numHidden
layer salida= Dense(capa5, 10)   // ~numOutput


layer model[6] =
{
	capa1,
	capa2,
	capa3,
	capa4,
	capa5,
	salida
}

//---------------------------------

struct model
{
	int numLayer;
	struct layer* layers;
};

double* salida(double* entrada)
{
	capa1 = Conv2D(entrada, 32, 3) // 32 kernels de 3x3
	capa2 = Conv2D(capa1, 64, 3)   // 64 kernels de 3x3
	capa3 = Pooling2D(capa2, 2)    // pooling de 2x2
	capa4 = Flatten(capa3)     // ~numInput
	capa5 = Dense(capa4, 128)  // ~numHidden
	salida= Dense(capa5, 10)   // ~numOutput

	return salida
}



////////////////////////////////////  NEURONAL NETWORK CONSTANTS

// Data
const int numExamplesInitial = 60000;
const int height = 28
const int width  = 28

// CNN
const ker1_num = 32
const ker1_size = 3
const ker2_num = 64
const ker2_size = 3

// Size of each layer
const int numInput  = 784;// 7
const int numHidden = 15; // 128
const int numOutput = 10; // 10

// How fast I want to learn
// If I make more iterations, I should reduce the learning rate.
const int trainingIterations = 10; // More is better, but it takes longer
const double learnigRate = 0.5; // Range in [0,1]  (0.5 and 1 it's OK)

// Weights initialization
const double minWeight = -0.99;
const double maxWeight = 0.99;


////////////////////////////////////  NEURONAL NETWORK VARIABLES

// Dataset
int    NumExamples; //max is 60000 (see numExamplesInitial)
double dataInput[numExamplesInitial][height][width];
double dataOutput[numExamplesInitial][numOutput];

double dataInput[numExamplesInitial][numInput];
double dataOutput[numExamplesInitial][numOutput];

// Input layer
double* input; // dataInput pointer to one sample

double kers_1[ker1_num][ker1_size][ker1_size]
double kers_2[ker2_num][ker2_size][ker2_size]

// Second layer (neurons)
double weightIH[numInput+1][numHidden]; // threhold=1 + input
double hidden[numHidden];

// Third layer (neurons)
double weightHO[numHidden+1][numOutput]; // threhold=1 + hidden neurons
double output[numOutput];
double* target; // dataOutput pointer to one sample


////////////////////////////////////////////  METHODS DECLARATION

void   test();
void   train();
void   forwardExample(int);
void   backPropagation(int);
void   initializeWeights(double, double);
void   readCSV(char*, int);
void   readMNIST(char*, int);
void   printData();
void   printWeights();
int    outputNumber(double output[]);
double sigmoid();
double sigmoidPrime();
double randomInRange(double, double);


////////////////////////////////////////////////////////  METHODS

int main()
{
	system("clear");

	printf("Reading data...\n");
	readMNIST("mnist_train.csv", 60000); //readCSV("datos.csv", 5000);

	printf("Training...\n");
	train();

	printf("Reading test data...\n");
	readMNIST("mnist_test.csv", 10000);

	printf("Testing...\n");
	test();
	
	return 0;
}

void train()
{
	int correctClassified;
	double percentage;

	initializeWeights(minWeight, maxWeight);

	for(int i=0; i<trainingIterations; i++)
	{
		correctClassified = 0;
		percentage = 0;
		printWeights();

		// Iterate examples
		for(int e=0; e<NumExamples; e++)
		{
			// Do the the back-propagation training
			backPropagation(e);

			if(outputNumber(output) == outputNumber(target))
				correctClassified++;
		}
		percentage = (double)correctClassified/NumExamples*100;
		printf("Epoch %d:\tCorrect classified %d of %d\t(%.1f%%).\n",
				i, correctClassified, NumExamples, percentage);
	}
}

void test()
{
	int correctClassified = 0;
	double percentage = 0;

	// Iterate examples
	for(int e=0; e<NumExamples; e++)
	{
		// Run the network
		forwardExample(e);

		// Save number of corrects digits classified
		if(outputNumber(output) == outputNumber(target))
			correctClassified++;		    
	}

	percentage = (double)correctClassified/NumExamples*100;
	printf("Correct classified %d of %d", correctClassified, NumExamples);
	printf(" (%.1f%%).\n", percentage);
}



void forwardExample(int sample)
{
	double net;

	input  = dataInput[sample];
	target = dataOutput[sample];

	// Hidden layer computation
	for(int j=0; j<numHidden; j++)
	{	
		net = 0;	
		for(int i=0; i<numInput; i++)
		{
			net += input[i] * weightIH[i][j];
		}
		net += weightIH[numInput][j]; // Bias = 1 * another extra weight

		hidden[j] = sigmoid(net);
	}

	// Output layer computation
	for(int k=0; k<numOutput; k++)
	{
		net = 0;
		for(int j=0; j<numHidden; j++)
		{
			net += hidden[j] * weightHO[j][k];
		}
		net += weightHO[numHidden][k]; // Bias = 1 * another extra weight

		output[k] = sigmoid(net);
	}
}

void backPropagation(int e)
{
	//double Error = 0.0;
	double DeltaH[numHidden];
	double DeltaO[numOutput];


	/////////////////////////////////////////  OBTAIN THE LAYERS OF THE NETWORK

	// Compute example and save its output
	forwardExample(e);

	////////////////////////////////////////////////////////  OBTAIN THE DELTAS

	// Calculate delta error of the output layer
	for(int k=0; k<numOutput; k++)
	{
		//Error += 0.5 * pow(target[k]-output[k], 2);
		DeltaO[k] = (target[k] - output[k]) *
					output[k] * (1.0 - output[k]);
	}

	// Calculate delta error of the hidden layer (back-propagate errors)
	for(int j=0; j<numHidden; j++)
	{
		double Sum[numHidden];
		Sum[j] = 0.0;

		for(int k=0; k<numOutput; k++)
		{
			Sum[j] += weightHO[j][k] * DeltaO[k];
		}
		DeltaH[j] = Sum[j] * hidden[j] * (1.0 - hidden[j]);
	}

	///////////////////////////////////////////////////////  UPDATE THE WHEIGHS

	// Update weights WeightHO
	for(int k=0; k<numOutput; k++)
	{
		for(int j=0; j<numHidden; j++)
		{
			weightHO[j][k] += learnigRate * hidden[j] * DeltaO[k];
		}
		weightHO[numHidden][k] += learnigRate * DeltaO[k]; // Bias
	}

	// Update weights WeightIH
	for(int j=0; j<numHidden; j++)
	{		
		for(int i=0; i<numInput; i++)
		{
			weightIH[i][j] += learnigRate * input[i] * DeltaH[j];
		}
		weightIH[numInput][j] += learnigRate * DeltaH[j]; // Bias
	}
}

///////////////////////////////////////////////////////  LESS IMPORTANT METHODS

void initializeWeights(double minimum_number, double max_number)
{
	// Initialize weightIH
	for(int i=0; i<numInput+1; i++)
	{		
		for(int j=0; j<numHidden; j++)
		{
			weightIH[i][j] = randomInRange(minimum_number, max_number);
		}
	}

	// Initialize weightHO
	for(int j=0; j<numHidden+1; j++)
	{
		for(int k=0; k<numOutput; k++)
		{
			weightHO[j][k] = randomInRange(minimum_number, max_number);
		}
	}
}

double randomInRange(double minimum_number, double max_number)
{
	int precission = 10000;
	int minimum_number_int = minimum_number * precission;
	int max_number_int = max_number * precission;

	int myRand = (rand() % (max_number_int + 1 - minimum_number_int)) + minimum_number_int;

	return (double)myRand/(double)precission;
}

double sigmoid(double net)
{
	return 1.0/(1.0 + exp(-net));
}

double sigmoidPrime(double output)
{
	return output * (1.0 - output);
}

int outputNumber(double output[])
{
	double maxValue = 0;
	int index = -1;

	for(int k=0; k<numOutput; k++)
	{
		if(output[k]>maxValue)
		{
			maxValue = output[k];
			index = k;
		}
	}
	return index;
}


////////////////////////////////////////////////////////////////  READ CSV DATA

void readCSV(char* fileName, int lines)
{
	NumExamples = lines;

	FILE* myfile = fopen(fileName, "r"); // Reading mode

	if (myfile == NULL) printf("Error opening file");

	for(int e=0; e<NumExamples; e++)
	{
		// READ INPUT
		for(int i=0; i<numInput; i++)
		{
			fscanf(myfile,"%lf,",&dataInput[e][i]); // %lf = long float = double
		}

		// READ OUTPUT
		for(int i=0; i<numOutput; i++)
		{
			fscanf(myfile,"%lf,",&dataOutput[e][i]); // %d = digit = int
		}
	}
	fclose(myfile);
}

void readMNIST(char* fileName, int lines)
{
	NumExamples = lines;

	FILE *myfile;
	double pixel;
	int realDigit;

	myfile=fopen(fileName, "r");

	for(int e=0; e<NumExamples; e++)
	{
		// READ TARGET OUTPUT
		fscanf(myfile,"%d,",&realDigit);
		for(int i=0; i<10; i++)
		{
			dataOutput[e][i] = 0.0;
		}
		dataOutput[e][realDigit] = 1.0;
		//printf("Number: %d \n",realDigit);

		// READ INPUT
		for(int i=0; i<numInput; i++)
		{
			fscanf(myfile,"%lf,",&pixel);
			dataInput[e][i] = pixel/255;
			//printf("%.15f \n",pixel);
		}
	}
	fclose(myfile);
}

//////////////////////////////////////////////////////////  DEBUG PRINT METHODS

void printData()
{
	printf("Data:\n");
	
	for(int e=0; e<NumExamples; e++)
	{
		//  INPUT
		printf("[");
		for(int i=0; i<numInput; i++)
		{
			printf("%f, ",dataInput[e][i]);
		}
		printf("]  [");

		// READ OUTPUT
		for(int i=0; i<numOutput; i++)
		{
			printf("%f, ",dataOutput[e][i]);
		}
		printf("]\n");
	}

	printf("\n");
}

void printWeights()
{
	printf("double weightIH[%d][%d] = {\n", numInput+1, numHidden);
	for(int i=0; i<numInput+1; i++)
	{
		printf("\t{");
		for(int j=0; j<numHidden; j++)
		{
			printf("%f, ", weightIH[i][j]);
		}
		printf("},\n");
	}
	printf("};\n\n");

	printf("double weightHO[%d][%d] = {\n", numHidden+1, numOutput);
	for(int j=0; j<numHidden+1; j++)
	{
		printf("\t{");
		for(int k=0; k<numOutput; k++)
		{
			printf("%f, ", weightHO[j][k]);
		}
		printf("},\n");
	}
	printf("};\n\n");
}


/////////////////////////////////////////////////  SAMPLE OUTPUT OF THE PROGRAM
/*
Reading data...
Training...
Epoch 0:	Correct classified 52305 of 60000	(87.2%).
Epoch 1:	Correct classified 54571 of 60000	(91.0%).
Epoch 2:	Correct classified 55043 of 60000	(91.7%).
Epoch 3:	Correct classified 55173 of 60000	(92.0%).
Epoch 4:	Correct classified 55422 of 60000	(92.4%).
Epoch 5:	Correct classified 55552 of 60000	(92.6%).
Epoch 6:	Correct classified 55655 of 60000	(92.8%).
Epoch 7:	Correct classified 55723 of 60000	(92.9%).
Epoch 8:	Correct classified 55804 of 60000	(93.0%).
Epoch 9:	Correct classified 55892 of 60000	(93.2%).
Reading test data...
Testing...
Correct classified 9242 of 10000 (92.4%).
*/