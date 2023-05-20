#include "kernels.h"
#include "iostream"
using namespace std;
/* 
 * Applies a 2d convolution on a 3D X using W: Z= W (conv) X + b
 * Tensor * X:		Input Tensor
 * Tensor * W:		Array of N weight Tensors (N == Z.size[0]) 
 * Tensor * Z:		Output Tensor 
 * Tensor * b:		Bias 
 */
void conv2d(Tensor * X, Tensor * W ,  Tensor * b, Tensor * Z)
{
    
   int Zc = 0, Zm = 0, Zn = 0;

    Zc = W->size[0]; // ASK: Is number of output channels same here as number of channels in wt and img
    Zn = X->size[1] - W->size[1] + 1;
    Zm = X->size[2] - W->size[2] + 1;
    int N = Z->size[0]; // number of filters
    int j = 0, k = 0;


    Tensor *currFilter = NULL; // TODO: check multiple filters handling

    for (size_t i = 0; i < (Z->size[0]*Z->size[1]*Z->size[2]); i++)
    {
        (*Z)[0][0][i] = 0;
    }
    


    for (size_t filters = 0; filters < N; filters++)
    {
        currFilter = &W[filters]; // ith weight tensor for ith output channel
        j = 0;
        k = 0;
        do
        {
            do
            {
                for (int c = 0; c < X->size[0]; c++)
                {
                    for (int p = 0; p < W->size[2]; p++)
                    {
                        for (int q = 0; q < W->size[1]; q++)
                        {
                            Z->data[filters][j][k] += (X->data[c][j + p][k + q]) * (currFilter->data[c][p][q]);
                        }
                    }
                }
                j++;
            } while (j < Zn);
            j = 0;
            k++;
        } while (k < Zm);
    }
    for (size_t i = 0; i < Z->size[0]; i++)
    {
        for (size_t j = 0; j < Z->size[1]; j++)
        {
            for (size_t k = 0; k < Z->size[2]; k++)
            {
                Z->data[i][j][k] += b->data[0][0][i];
            }
        }
    }
}


//printf("Kernel_size%d", W->size[0]);

/*
 * Applies a max pool layer on X (size = stride = 2)
 * Tensor * X:	input Tensor
 * Tensor * Z:	output Tensor
 */


void maxPool(Tensor * X, Tensor * Z)
{
    //size of the pooling ...
    //window = 2;
    //stride = 2
    //padding = 2

    int size  = 2;
    int stride = 2;
    int padding = 0;


    //calculate the size of the output tensor

    int out_size = (X->size[1] - size + 2*padding)/ stride + 1;
    
    //initialize output tensor
    //can be let in as security measurement. But Z -> size[1] == out_size and Z->size[2] == out_size should be TRUE.

    //Z -> size[0] = X -> size[0];
    //Z -> size[1] = out_size;
    //Z -> size[2] = out_size;

    //apply for maxpooling

    for(int i = 0; i < X->size[0]; i++){
        for(int j=0; j < out_size; j++){
            for(int k=0; k<out_size; k++){
                FLOAT max = (*X)[i][j*stride][k*stride];
                for(int l=0; l<size; l++){
                    for(int m = 0; m < size; m++){
                        FLOAT tmp = (*X)[i][j*stride + l][k*stride + m];
                        if(tmp>max){
                            max = tmp;
                        }
                    }
                }
                (*Z)[i][j][k] = max;
            }
        }
    }          
}


/*
 * Applies a Linear layer: z = Wx + b 
 * Flatten the input if required 
 * Tensor *	X: input Tensor
 * Tensor *	W: weight Matrix (in Tensor form)
 * Tensor *	B: bias array (in Tensor form)
 * Tensor *	Z: output array (in Tensor form)
 */

void Linear(Tensor * X, Tensor * W, Tensor * B, Tensor * Z)
{
    //flattening the input tensor X.
    /*Method
        1. firstly go along the height coordinate [][h][]
        2. secondly, change to the width coordinate [][][w]. Change for one width element. Start over with [][h][].
        3. thirdly, if all height-width combinations are made ... change the channel [c][][]. Proceed with [][h][] and [][][w]
        */

    
    int input_channels = X->size[0];
    int input_height = X->size[1];
    int input_width = X->size[2];
    int tensor_elements = input_height*input_width*input_channels;
    


    double sum = 0;
    //flattened input X-->x
    Tensor x(1,1, tensor_elements);
    //cout<<x.size[0] * x.size[1] * x.size[2]<<endl;
    
    //flattening the input tensor X.
    /*Method
        1. firstly go along the width coordinate [][][w]
        2. secondly, change to the height coordinate [][h][]. Change for one height element. Start over with [][][w].
        3. thirdly, if all height-width combinations are made ... change the channel [c][][]. Proceed with [][][w] and [][h][].
        */
    //channels
    
    //if(X->size[0]!=1 && X->size[1]!=1){

        int index = 0;

        for(int i=0; i<X->size[0]; i++){
            /*//height
            for(int j=0; j< X->size[1]; j++){
                //width
                for(int k=0; k< X->size[2]; k++){
                    //flattening X to x
            
                    x.data[0][0][k + input_height*j + input_height*input_width*i] = X->data[i][j][k];
                }
            } */
            //height
            for(int j=0; j< X->size[1]; j++){
                //width
                for(int k=0; k< X->size[2]; k++){
                    //flattening X to x
                    int c = 0;
                    //if ( i == 0)
                    //  { c= 0; } else{ c = X->size[0];}
                        //x.data[0][0][j + j*X->size[1] + ((j*k )+i)] = X->data[i][j][k];
                        //x.data[0][0][index++] = X->data[i][j][k];
                        x.data[0][0][k + j*X->size[1] + (X->size[1]*X->size[2])*i] = X->data[i][j][k];
                }
            }
        }
    //}
    /* else{
        //if X has shape 1x1xVALUE, then transfer the values from input X to the flattened input version x.
        for(int i = 0; i < X->size[2]; i++){
            x.data[0][0][i] = X->data[0][0][i];
        } 
    }*/

    //Compute the different scalar products w.Tx and add bias b to the individual scalar products w.Tx+b.
    Tensor* currW = NULL;
    //Amount of weights. For each w one scalar product. 
    index = 0;
    
    for (size_t c = 0; c < W->size[0]; c++)
    {
        /* code */
        //currW = &W[c];
    
        for(int i = 0; i<W->size[1];i++){
            //Each weight element in currently considered weight vector. Weight vector element W.
            for(int j = 0; j< W->size[2]; j++){
                //sum += currW->data[c][i][j]*x.data[0][0][index++];
                //sum += currW->data[c][i][j]*x.data[0][0][j];
                sum += W->data[c][i][j]*x.data[0][0][j];
            }
        

            sum += B->data[0][0][i];
            //printf("Size_of_sum: %lu", sizeof(sum));
            Z->data[0][0][i] = sum;
            sum = 0;
        }

    }
    /*printf("Size flattened x: %d\n", x.size[2]);
    printf("Weight channel: %d\n", W->size[0]);
    printf("Bias amount: %d\n", B->size[2]);*/
/*
    printf("Input X: %dx%dx%d\n",X->size[0], X->size[1], X->size[2]);
    printf("Flattened input x: %dx%dx%d\n",x.size[0], x.size[1], x.size[2]);
    printf("Weight W: %dx%dx%d\n",W->size[0], W->size[1], W->size[2]);
    printf("Bias B: %dx%dx%d\n",B->size[0], B->size[1], B->size[2]);
    printf("Output Z: %dx%dx%d\n",Z->size[0], Z->size[1], Z->size[2]);
  */  
    //printf("Weight*Input: %d", W->data*X->data);
}


/*
 * Applies the ReLU activation function: Z = ReLU(X)
 * Tensor * X: input Tensor
 * Tensor * Z: output Tensor
 */
void ReLU(Tensor *X , Tensor *Z)
{ 
    /*Tensor & X_ref = *X;
    for(int i=0; i < X_ref.size[0]; i++){*/
    for(int i=0; i < X ->size[0]; i++){
        for(int j=0; j < X ->size[1]; j++){
            for(int k=0; k < X->size[2]; k++){
                float tmp = (*X)[i][j][k];
                float val = 0;
                if(tmp > 0){
                    val = tmp;
                }
                (*Z)[i][j][k] = val;
            }
        }
    }
}


/*
 * Applies the Softmax activation function z = exp(x_i)/sum(exp(x_j))
 * This is a stable Softmax implementation
 * Tensor * X: input vector in Tensor form
 * Tensor * Z: output vector in Tensor form
 */
void Softmax(Tensor * X, Tensor * Z)
{
    /*First Part: Building the denominator sum ... sum(exp(x_j))*/
    float sum_exp = 0;
    for(int i=0; i < X ->size[0]; i++){
        for(int j=0; j < X ->size[1]; j++){
            for(int k=0; k < X->size[2]; k++){
                float tmp = (*X)[i][j][k];
                /*compute the euler exponent of the tensor value*/
                tmp = exp(tmp);
                /*building the sum ... denominator of the probability function*/
                sum_exp = sum_exp + tmp;
            }
        }
    }
    for(int i=0; i < X ->size[0]; i++){
        for(int j=0; j < X ->size[1]; j++){
            for(int k=0; k < X->size[2]; k++){
                float tmp = (*X)[i][j][k];
                float val = 0;
                /*compute the euler exponent of the tensor value*/
                tmp = exp(tmp);
                /*compute the probability z = exp(x_i)/sum(exp(x_j))*/
                val = tmp / sum_exp;
                (*Z)[i][j][k] = val;
            }
        }
    }

}