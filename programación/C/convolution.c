#include<stdio.h>

#define inputSize  4
#define kernelSize 3
#define outputSise 2

int input[inputSize][inputSize] =
{
	{3, 3, 2, 1, 0},
	{0, 0, 1, 3, 1},
	{3, 1, 2, 2, 3},
	{2, 0, 0, 2, 2},
	{2, 0, 0, 0, 1}
};

int kernel[kernelSize][kernelSize] =
{
	{0, 1, 2},
	{2, 2, 0},
	{0, 1, 2}
};

int output[outputSise][outputSise];


void convolution2D()
{
	int O = I-K+1;
	int x, w, sum;

	for (int i=0; i<O; ++i) // iterate through output image in 2D
	{
		for (int j=0; j<O; ++j)
		{
			sum = 0;
			for (int ki=0; ki<K; ++ki) // iterate over kernel in 2D
			{
				for (int kj=0; kj<K; ++kj)
				{
					x = input[i + ki][j + kj];
					w = kernel[ki][kj];

					sum += x * w;
				}
			}
			output[i][j] = sum;
		}
	}
}


/* NOTE: 2D CONVOLUTION WITH CHANNELS

How is the convolution operation carried out
when multiple channels are present at the input layer?
For example, 3 channels: RGB

In such a case you have one 2D kernel per input channel.
So you perform each convolution (2D Input, 2D kernel) separately
and you sum the contributions which gives the final output feature map.

It's like a 3D kernel that moves in 2D.

*/
void convolution2Dchannels()
{
	int O = I-K+1;
	int x, w, sum;

	for (int i=0; i<O; ++i) // iterate through output image in 2D
	{
		for (int j=0; j<O; ++j)
		{
			sum = 0;
			for (int ki=0; ki<K; ++ki) // iterate over kernel in 2D
			{
				for (int kj=0; kj<K; ++kj)
				{
					for (int c=0; c<channels; ++c) // iterate over each channel
					{
						x = input[c][i + ki][j + kj];
						w = kernel[c][ki][kj];
						sum += x * w;
					}
				}
			}
			output[i][j] = sum;
		}
	}
}

void printOutput()
{
	printf("Output:\n");
	for (int i=0; i<O; ++i) // iterate through output image
	{
		printf("\t");

		for (int j=0; j<O; ++j)
		{
			printf("%d\t", output[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}


int main()
{
	convolution2D();
	printOutput();
	return 0;
}



