#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#define LAYER1SIZE 28*28
#define LAYER2SIZE 200
#define LAYER3SIZE 10
#define TRAININGDATASIZE 50 //50000
#define BATCHSIZE 50
#define TESTINGDATASIZE 10000 //10000
#define NUMTRAININGEPOCHS 1000 //10000
#define eta 1.0
#define TRAININGFILE "MNISTTrainingData.txt"
#define TESTINGFILE "MNISTTestData.txt"

float* vectorizeInt(int x);
float magnitudeSquaredOfDifference(float* vector1, float* vector2, int size);
float magnitudeSquared(float* vector, int size);
void normalize8BitValues(float* data, int size);
void normalizeArrayToPercentage(float* arr, int size);
int findIndexOfMaxItem(float* arr, int size);
float computeWeightedSum(float* wieghts, float* values, int size);
float sigmoidPrime(float x);
float sigmoid(float x);
float randomSmallFloat(float maxWeight);
float randomFloat01(void);


int main(){
    printf("PAREMATERS:\n    Hidden Layer Size = %d \n    Training Data Size = %d \n    eta = %f \n    Number of Epochs = %d\n", LAYER2SIZE, TRAININGDATASIZE, eta, NUMTRAININGEPOCHS);
	int a, b, c;
    int target;
    float layer1Values[LAYER1SIZE] = {};
    float layer2Values[LAYER2SIZE] = {};
    float layer3Values[LAYER3SIZE] = {};
    float layer2Biases[LAYER2SIZE] = {};
    float layer3Biases[LAYER3SIZE] = {};
	float layer21Weights[LAYER2SIZE][LAYER1SIZE];
	float layer32Weights[LAYER3SIZE][LAYER2SIZE];
    int arrayCounter;
    int L2Neuron, L3Neuron, iter, epoch, resetCounter;
    float beforeSigmoid;
    float totalEpochError;
    float averageEpochError;
    float gradientLayer32Weights[LAYER3SIZE][LAYER2SIZE];
    float gradientLayer3Biases[LAYER3SIZE];
    float gradientLayer21Weights[LAYER2SIZE][LAYER1SIZE];
    float gradientLayer2Biases[LAYER2SIZE];
    float dCostdSigmoid3, dSigmoid3dBeforeSigmoid3, dBeforeSigmoid3dWeight32;
    float dBeforeSigmoid3dSigmoid2, dSigmoid2dBeforeSigmoid2, dBeforeSigmoid2dWeight21;
    
	//Begin training
	for (b = 0; b < LAYER2SIZE; b++){
		for (a = 0; a < LAYER1SIZE; a++){
			layer21Weights[b][a] = randomSmallFloat(0.2);
		}
        layer2Biases[b] = randomSmallFloat(0.2);
	}
	for (c = 0; c < LAYER3SIZE; c++){
		for (b = 0; b < LAYER2SIZE; b++){
			layer32Weights[c][b] = randomSmallFloat(0.2);
		}
        layer3Biases[c] = randomSmallFloat(0.2);
	}
    
    FILE* trainingFile;
    trainingFile = fopen(TRAININGFILE, "r");
    if (trainingFile == NULL){
      perror("Cannot find file");
      exit(1);
    }
    int nxtChar;
	
    for (epoch = 0; epoch < NUMTRAININGEPOCHS; epoch++){
        memset(gradientLayer32Weights, 0, sizeof(gradientLayer32Weights[0][0]) * LAYER3SIZE * LAYER2SIZE);
        memset(gradientLayer3Biases, 0, sizeof(gradientLayer3Biases[0]) * LAYER3SIZE);
        memset(gradientLayer21Weights, 0, sizeof(gradientLayer21Weights[0][0]) * LAYER2SIZE * LAYER1SIZE);
        memset(gradientLayer2Biases, 0, sizeof(gradientLayer2Biases[0]) * LAYER2SIZE);
        totalEpochError = 0.0;
        
        int trainingDataPoint;
        for (trainingDataPoint = 0; trainingDataPoint < TRAININGDATASIZE; trainingDataPoint++){
            
            //Begin initializing data
            for (resetCounter = 0; resetCounter < LAYER1SIZE; resetCounter++){
                layer1Values[resetCounter] = 0.0;
            }
            nxtChar = fgetc(trainingFile);
            target = (int) nxtChar - '0';
            //printf("The target is %d\n", target);
            arrayCounter = 0;
            bool incrementCounter = false;
            while (1) {
                nxtChar = fgetc(trainingFile);
                if (nxtChar == 13 || nxtChar == 10 || nxtChar == EOF){
                    break;
                }
                else if (nxtChar > '9' || nxtChar < '0'){
                    if (incrementCounter){
                        arrayCounter ++;
                        incrementCounter = false;
                    }
                }
                else{
                    incrementCounter = true;
                    nxtChar = nxtChar - '0';
                    layer1Values[arrayCounter] = layer1Values[arrayCounter] * 10.0 + (float) nxtChar;
                    //printf("%f for val %d\n", layer1Values[arrayCounter], arrayCounter);
                }
            }

            normalize8BitValues(layer1Values, LAYER1SIZE);
            //End initializing data
            
            for (L2Neuron = 0; L2Neuron < LAYER2SIZE; L2Neuron++){
                beforeSigmoid = computeWeightedSum(layer21Weights[L2Neuron], layer1Values, LAYER1SIZE);
                beforeSigmoid += layer2Biases[L2Neuron];
                layer2Values[L2Neuron] = sigmoid(beforeSigmoid);
                //printf("Layer 2 Nueron : %d; After Sigmoid : %f\n", L2Neuron, layer2Values[L2Neuron]);
            }
            
            for (L3Neuron = 0; L3Neuron < LAYER3SIZE; L3Neuron++){
                beforeSigmoid = computeWeightedSum(layer32Weights[L3Neuron], layer2Values, LAYER2SIZE);
                beforeSigmoid += layer3Biases[L3Neuron];
                layer3Values[L3Neuron] = sigmoid(beforeSigmoid);
                //printf("Layer 3 Nueron : %d; After Sigmoid : %f\n", L3Neuron, layer3Values[L3Neuron]);
            }
            
            normalizeArrayToPercentage(layer3Values, LAYER3SIZE);
            //printf("Normalized Output, data point %d, target %d\n", trainingDataPoint, target);
            for (iter = 0; iter < LAYER3SIZE; iter++){
                //printf("%d : %f; ", iter, layer3Values[iter]);
            }
            
            totalEpochError += magnitudeSquaredOfDifference(vectorizeInt(target), layer3Values, LAYER3SIZE);
            
            for (c = 0; c < LAYER3SIZE; c++){
                if (c == target){
                    //dCostdSigmoid3 = 2 * (layer3Values[c] - 1) / LAYER3SIZE;
                    dCostdSigmoid3 = 2 * (layer3Values[c] - 1);
                }
                else{
                    //dCostdSigmoid3 = 2 * layer3Values[c] / LAYER3SIZE;
                    dCostdSigmoid3 = 2 * layer3Values[c];
                }
                dSigmoid3dBeforeSigmoid3 = sigmoidPrime(layer3Values[c]);
                for (b = 0; b < LAYER2SIZE; b++){
                    dBeforeSigmoid3dWeight32 = layer2Values[b];
                    gradientLayer32Weights[c][b] += dCostdSigmoid3 * dSigmoid3dBeforeSigmoid3 * dBeforeSigmoid3dWeight32;
                    
                    dBeforeSigmoid3dSigmoid2 = layer32Weights[c][b];
                    dSigmoid2dBeforeSigmoid2 = sigmoidPrime(layer2Values[b]);
                    for (a = 0; a < LAYER1SIZE; a++){
                        dBeforeSigmoid2dWeight21 = layer1Values[a];
                        gradientLayer21Weights[b][a] += dCostdSigmoid3 * dSigmoid3dBeforeSigmoid3 * dBeforeSigmoid3dSigmoid2 * dSigmoid2dBeforeSigmoid2 * dBeforeSigmoid2dWeight21;
                    }
                    gradientLayer2Biases[b] += dCostdSigmoid3 * dSigmoid3dBeforeSigmoid3 * dBeforeSigmoid3dSigmoid2 * dSigmoid2dBeforeSigmoid2;
                }
                gradientLayer3Biases[c] += dCostdSigmoid3 * dSigmoid3dBeforeSigmoid3;
            }
        
            //printf("\n-----------------END OF TRAINING SET %d--------------------\n", trainingDataPoint);
            
        }
        
        averageEpochError = totalEpochError / TRAININGDATASIZE;
        printf("Avg Epoch Error: %f\n", averageEpochError);
        
        //Begin update weights and biases
        
        for (c = 0; c < LAYER3SIZE; c++){
            for (b = 0; b < LAYER2SIZE; b++){
                layer32Weights[c][b] -= eta * gradientLayer32Weights[c][b] / (float) BATCHSIZE;
                //printf("Found gradient of %f for connection L2 %d to L3 %d\n", gradientLayer32Weights[c][b] / (float) BATCHSIZE, b, c);
            }
            layer3Biases[c] -= eta * gradientLayer3Biases[c] / (float) BATCHSIZE;
            //printf("Found gradient of %f for bias of L3 neuron %d\n", gradientLayer3Biases[c] / (float) BATCHSIZE, b);
        }
        
        for (b = 0; b < LAYER2SIZE; b++){
            for (a = 0; a < LAYER1SIZE; a++){
                layer21Weights[b][a] -= eta * gradientLayer21Weights[b][a] / (float) BATCHSIZE;
                //printf("Found gradient of %f for connection L1 %d to L2 %d\n", gradientLayer21Weights[b][a] / (float) BATCHSIZE, a, b);
            }
            layer2Biases[b] -= eta * gradientLayer2Biases[b] / (float) BATCHSIZE;
            //printf("Found gradient of %f for bias of L2 neuron %d\n", gradientLayer2Biases[b] / (float) BATCHSIZE, a);
        }
        
        //End update weights and biases
        
        printf("-----------END OF EPOCH %d-----------\n", epoch);
    }
    fclose(trainingFile);
    //End training
    
    printf("--------------------------BEGIN TESTING----------------------\n");
    
    //Begin testing
    FILE* testingFile;
    testingFile = fopen(TESTINGFILE, "r");
    if (testingFile == NULL){
      perror("Cannot find file");
      exit(1);
    }
    
    int totalCorrect = 0;
    int testingDataPoint;
    for (testingDataPoint = 0; testingDataPoint < TESTINGDATASIZE; testingDataPoint++){
            
        //Begin Initializing data
        for (resetCounter = 0; resetCounter < LAYER1SIZE; resetCounter++){
            layer1Values[resetCounter] = 0.0;
        }
        nxtChar = fgetc(testingFile);
        target = (int) nxtChar - '0';
        arrayCounter = 0;
        bool incrementCounter = false;
        while (1) {
            nxtChar = fgetc(testingFile);
            if (nxtChar == 13 || nxtChar == 10 || nxtChar == EOF){
                break;
            }
            else if (nxtChar > '9' || nxtChar < '0'){
                if (incrementCounter){
                    arrayCounter ++;
                    incrementCounter = false;
                }
            }
            else{
                incrementCounter = true;
                nxtChar = nxtChar - '0';
                layer1Values[arrayCounter] = layer1Values[arrayCounter] * 10.0 + (float) nxtChar;
                //printf("%f for val %d\n", layer1Values[arrayCounter], arrayCounter);
            }
        }
        /*
        printf("------------------END OF TESTING SET %d----------------------\n", testingDataPoint + 1);
        */
        
        normalize8BitValues(layer1Values, LAYER1SIZE);
        //End initializing data
        
        for (L2Neuron = 0; L2Neuron < LAYER2SIZE; L2Neuron++){
            beforeSigmoid = computeWeightedSum(layer21Weights[L2Neuron], layer1Values, LAYER1SIZE);
            beforeSigmoid += layer2Biases[L2Neuron];
            layer2Values[L2Neuron] = sigmoid(beforeSigmoid);
            //printf("Layer 2 Nueron : %d; After Sigmoid : %f\n", L2Neuron, layer2Values[L2Neuron]);
        }
        //printf("-----------------------------------\n");
            
        for (L3Neuron = 0; L3Neuron < LAYER3SIZE; L3Neuron++){
            beforeSigmoid = computeWeightedSum(layer32Weights[L3Neuron], layer2Values, LAYER2SIZE);
            beforeSigmoid += layer3Biases[L3Neuron];
            layer3Values[L3Neuron] = sigmoid(beforeSigmoid);
            //printf("Layer 3 Nueron : %d; After Sigmoid : %f\n", L3Neuron, layer3Values[L3Neuron]);
        }
            
        normalizeArrayToPercentage(layer3Values, LAYER3SIZE);
        printf("Target Value: %d \n Guess: %d, with %f percent confidence\n", target, findIndexOfMaxItem(layer3Values, LAYER3SIZE), 100. * layer3Values[findIndexOfMaxItem(layer3Values, LAYER3SIZE)]);
        for (iter = 0; iter < LAYER3SIZE; iter++){
            printf("%d : %f; ", iter, layer3Values[iter]);
        }

        if (target == findIndexOfMaxItem(layer3Values, LAYER3SIZE)){
            totalCorrect ++;
        }
        
        printf("\n-----------------END OF TESTING SET %d--------------------\n", testingDataPoint);
    }
    
    printf("\n TOTAL PERFORMANCE: %d out of %d correct, for an accuracy of %f percent\n", totalCorrect, TESTINGDATASIZE, 100.0 * (float) totalCorrect / (float) TESTINGDATASIZE);
    
    fclose(testingFile);
    
    //End testing
    
    return 0;
}

float* vectorizeInt(int x){
    static float vectorZero[10] = {1., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
    static float vectorOne[10] = {0., 1., 0., 0., 0., 0., 0., 0., 0., 0.};
    static float vectorTwo[10] = {0., 0., 1., 0., 0., 0., 0., 0., 0., 0.};
    static float vectorThree[10] = {0., 0., 0., 1., 0., 0., 0., 0., 0., 0.};
    static float vectorFour[10] = {0., 0., 0., 0., 1., 0., 0., 0., 0., 0.};
    static float vectorFive[10] = {0., 0., 0., 0., 0., 1., 0., 0., 0., 0.};
    static float vectorSix[10] = {0., 0., 0., 0., 0., 0., 1., 0., 0., 0.};
    static float vectorSeven[10] = {0., 0., 0., 0., 0., 0., 0., 1., 0., 0.};
    static float vectorEight[10] = {0., 0., 0., 0., 0., 0., 0., 0., 1., 0.};
    static float vectorNine[10] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 1.};
    
    if (x == 0){
        return vectorZero;
    }
    else if (x == 1){
        return vectorOne;
    }
    else if (x == 2){
        return vectorTwo;
    }
    else if (x == 3){
        return vectorThree;
    }
    else if (x == 4){
        return vectorFour;
    }
    else if (x == 5){
        return vectorFive;
    }
    else if (x == 6){
        return vectorSix;
    }
    else if (x == 7){
        return vectorSeven;
    }
    else if (x == 8){
        return vectorEight;
    }
    else if (x == 9){
        return vectorNine;
    }
    else{
        perror("Invalid input");
        exit(1);
    }
}

float magnitudeSquaredOfDifference(float* vector1, float* vector2, int size){
    float feed[size];
    int iter;
    for (iter = 0; iter < size; iter++){
        feed[iter] = *(vector1 + iter) - *(vector2 + iter);
    }
    return magnitudeSquared(feed, size);
}

float magnitudeSquared(float* vector, int size){
    float sum = 0;
    int iter;
    for (iter = 0; iter < size; iter++){
        sum += *(vector + iter) * *(vector + iter);
    }
    return sum;
}

void normalize8BitValues(float* data, int size){
    int i;
    for (i = 0; i < size; i++){
        data[i] = *(data + i) / 255.0;
    }
}

void normalizeArrayToPercentage(float* arr, int size){
    int i;
    float sum = 0;
    for (i = 0; i < size; i++){
        sum += *(arr + i);
    }
    for (i = 0; i < size; i++){
        *(arr + i) = *(arr + i) / sum;
    }
}

int findIndexOfMaxItem(float* arr, int size){
    float maxVal = *arr;
    int maxIndex = 0;
    int i;
    for (i = 1; i < size; i++){
        if (*(arr + i) > maxVal){
            maxVal = *(arr + i);
            maxIndex = i;
        }
    }
    return maxIndex;
}

float computeWeightedSum(float* wieghts, float* values, int size){
    int neuron;
    float sum = 0;
    for (neuron = 0; neuron < size; neuron++){
        sum += *(wieghts + neuron) * *(values + neuron);
    }
    return sum;
}

float sigmoid(float x){
    return (1.0 / (1.0 + exp(-1 * x)));
}

float sigmoidPrime(float x){
    return sigmoid(x) * (1 - sigmoid(x));
}

float randomSmallFloat(float maxWeight){
		return maxWeight * 2 * (((double) rand() / (double) RAND_MAX) - 0.5);
	}

float randomFloat01(void){
    return (double) rand() / (double) RAND_MAX;
}
