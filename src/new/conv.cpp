#include "conv.h"
#include <cmath>

#include <iostream>
using namespace std;

/*----------------------------- Helper Functions -------------------------------------*/
/*
 * This takes the input matrix c (*W_in)[c] flips it and stores it
 * in the real values of the complex Matrix W_out (1 channel)
 * Inputs:
 * Tensor	*	W_in:	Pointer to input Tensor (real)
 * C_Tensor	*	W_out:	Pointer to the output Tensor values stored in real part (complex) 
 * int 			c:		Channel of input to use
 */
void flip_Matrix(Tensor * W_in, C_Tensor * W_out, int c)
{
	////cout<<"a"<<endl;
	int k_width = W_in->size[1];
	////cout<<"b"<<endl;
	int upper_lim = k_width/2 + k_width%2;
	////cout<<"c"<<endl;
	int extra = -1 + k_width%2;
	////cout<<"d"<<endl;
	for(int i =-k_width/2; i < upper_lim; i++){
		for(int j =-k_width/2; j<upper_lim; j++){
			

			(*W_out)[0][i+k_width/2][j+k_width/2].real( 
				(*W_in)[c][-i + k_width/2 + extra][-j + k_width/2 + extra]);

			
		}
	}

	////cout<<"flip done"<<endl;
}

/* You can experiment with these as well */
const static FFT_STRUCT 	FFT_3 = {8,2,6};
const static FFT_STRUCT 	FFT_5 = {16,4,12};
const static FFT_STRUCT 	FFT_7 = {16,6,10};
const static FFT_STRUCT 	FFT_11= {32,10,22};
const FFT_STRUCT * getFFT(uint32_t k_size)
{
	const FFT_STRUCT * fft;
	switch(k_size){
		case 3:
			fft = &FFT_3;
			break;
		case 5:
			fft = &FFT_5;
			break;
		case 7:
			fft = &FFT_7;
			break;
		case 11:
			fft = &FFT_11;
			break;
		default:
			//printf("Kernel Size %d not supported by FFT\n",k_size);
			return NULL;
	}
	return fft;
}




/*-------------------------------- Winograd -------------------------------------------*/



void transposeMatrix( const float ** G, int rows, int cols, float ** transpose )
{
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			transpose[j][i] = G[i][j];
		}
	}
}


const WINOGRAD_STRUCT * getWino(uint32_t k_size)
{
	const WINOGRAD_STRUCT * wino;
	switch(k_size){
		case 3:
			wino = &Wino_F2_3; // Try Wino F4_3
			break;
		case 5:
			wino = &Wino_F4_5;
			break;
		case 7:
			wino = &Wino_F4_7;
			break;
		case 11:
			wino = &Wino_F4_11;
			break;
		default:
			//printf("Kernel Size %d not supported by Winograd \n",k_size);
			return NULL;
	}
	return wino;
}


/*-------------------------------- Winograd -------------------------------------------*/

/* 
 * Pre Transform the Weights
 * WINOGRAD_STRUCT 	*wino 	: Struct containing tile size, A^T, G, B^T
 * Tensor 			*W		: Untransformed Weight Tensor
 * int		output_channels	: Number of output channels
 * Return:		Tensor *	: New Tensor containing transformed Weights
 */
Tensor * winoWeights(Tensor * W, int output_channels)
{

	#if 1
	W->size[0]; //number of channels in kernel
	W->size[1]; //square dimensions
	W->size[2]; //square dimensions
	
	//printf("W is **********8: %d\n%d\n", W[0].size[1], W[0].size[2]);
	//exit(0);
	int g_rows = 0;// = sizeof(wino->G) / sizeof(wino->G[0]); 
	int g_col = 0; // = sizeof(wino->G[0]) / sizeof(float);
	
	// printf("winoWeights opened.\n");
	
	WINOGRAD_STRUCT* wino = NULL;
	Tensor *currFilter = NULL; // TODO: check multiple filters handling
	Tensor *transW = new Tensor[output_channels];
	Tensor *InterRes = new Tensor[output_channels];
	
	// printf("Array of Tensors init.\n");
	
	for (size_t filters = 0; filters < output_channels; filters++)
	{
		// printf("Determine dimensions matrix At, Bt, G\n");
		// printf("Size element: %d\n", W[filters].size[1]);
		//wino based on kernel size
		wino = getWino( W[filters].size[1]);
		// printf("End code\n");
		if(( wino->kernel_size == 3) && ( wino->out_size  == 2))
		{
			g_rows = 4;
			g_col = 3;
		}
		else if (( wino->kernel_size == 3) && ( wino->out_size  == 4))
		{
			g_rows = 6;
			g_col = 3;
		}
		else if (( wino->kernel_size == 5) && ( wino->out_size  == 4))
		{
			g_rows = 8;
			g_col = 5;
		}
		else if (( wino->kernel_size == 7) && ( wino->out_size  == 4))
		{
			g_rows = 10;
			g_col = 7;
		}
		else if (( wino->kernel_size == 11) && ( wino->out_size  == 4))
		{
			g_rows = 14;
			g_col = 11;
		}
		currFilter = &W[filters];
		// printf("Dimensions G: checked.\n");
		
		/** Tile information based on current kernel **/
		//printf("\n wino.Bt, filters, wino->kernel_size, wino->out_size, wino->tile_size, wino->tile_stride  %d %d %d %d %d\n",
		//	 filters, wino->kernel_size, wino->out_size, wino->tile_size, wino->tile_stride );

		/** Tile Information **/

		//printf("segfault in constructor---------");					
		//transW[filters] = Tensor(currFilter->size[0], g_rows, currFilter->size[2]);
		//for (size_t i = 0; i < output_channels; i++)
		//{
			
		//}
		//transW ... array of tensors. transW[filters].allocate ...Defines the dim boundaries of the filter-th tensor of transW
		transW[filters].allocate(currFilter->size[0], wino->tile_size,  wino->tile_size);
		InterRes[filters].allocate(currFilter->size[0], wino->tile_size, currFilter->size[1]);

		//Initialise all tensors to zero initially

		//Initialisation ends


		
	//	printf("segfault after constructor---------");
	//Gg ... loop is ok.
	//Computation scheme: compute all column-elements of an row for the output. Switch row only, if all column elements are computed.
		//Switch to next row: cursor jumps to new row in input (here: G). Kernel goes again through all columns row-wise.

		//loop-1 ... go into an channel. Same as feature map.
		for (size_t c = 0; c < currFilter->size[0]; c++)
		{
			// Matrix mul: Gg (g_rows, g_col) * ( size[1] == size[2])
			//G = 4,3  g =3,3 Gt = 3,4, 
			// ( 4,3)
			//loop-2 ... go into row of G. G-values*g_featuremap-values
			for (size_t i = 0; i < g_rows; i++)
			{
				for ( size_t j = 0; j < currFilter->size[1]; j++){

					//transW.data[filters][i][j] = 0;
					//loop-4 ... columns of G. Go along all columns of a row i.
					for (size_t k = 0; k < g_col; k++)
					{
						//TODO: Need intermediate matrix to store G.g
						//transW[filters].data[c][i][j] += wino->G[i][k] * currFilter->data[c][k][j];
						InterRes[filters].data[c][i][j] += wino->G[i][k] * currFilter->data[c][k][j];
					}
				}
			}
		}  

		//Gt for each G, which is selected based on kernel size
		float** Gt = new float*[g_col];
    	for (int i = 0; i < g_col; i++) 
		{
        	Gt[i] = new float[g_rows];
    	}
		transposeMatrix(wino->G, g_rows, g_col, Gt);
	
		//(previous reuslt) * Gt
		for (size_t c = 0; c < InterRes[filters].size[0]; c++)
		{
			// Matrix mul: Gg (g_rows, g_col) * ( size[1] == size[2])
			//G = 4,3  g =3,3 Gt = 3,4
			//transW = (4,3) * Gt = (3,4)
			for (size_t i = 0; i < InterRes[filters].size[1]; i++)
			{
				//Number of cols of Gt = g_rows of G
				for ( size_t j = 0; j < InterRes[filters].size[1]; j++){
					//transW.data[filters][i][j] = 0;
					for (size_t k = 0; k < g_col; k++)
					{ //Gt[j][k]
						transW[filters].data[c][i][j] += InterRes[filters].data[c][i][k] * Gt[k][j];
					}
				}
			}
		}
		//delete [] InterRes;
		// Free the memory for each row
		for (int i = 0; i < g_col; i++) {
		    delete[] Gt[i];
		}
		// Free the memory for the array of row pointers
		delete[] Gt;
	}
//G = 4,3  g =3,3 Gt = 3,4

	// Initialize all elements of transW to 0
	#if 0
	for (size_t filters = 0; filters < output_channels; filters++)
	{
		currFilter = &W[filters];
		//transW.resize(output_channels, g_rows, currFilter->size[1]);
		//transW[filters] = new Tensor(currFilter->size[0], g_rows, currFilter->size[1]);

		for (size_t c = 0; c < currFilter->size[0]; c++)
		{
			for (size_t i = 0; i < g_rows; i++)
	    	{
	    	    for (size_t j = 0; j < currFilter->size[1]; j++)
	    	    {
	    	        transW[filters]->data[c][i][j] = 0.0; // Or the desired initial value
	    	    }
	    	}
		}
	}
	#endif
	

	delete [] InterRes;
	//printf("DONEEEEEEEEEEE");
	return transW;
	#endif
}



//#########################################################
//winoTile ... B.T*d*B. Transformation input tiles.
//#########################################################



Tensor * winoTile(Tensor * W, int output_channels, WINOGRAD_STRUCT* wino, int row_matrix, int column_matrix, int select)
{
	#if 0
	W->size[0]; //number of channels in kernel
	W->size[1]; //square dimensions
	W->size[2]; //square dimensions
	
	//printf("W is **********8: %d\n%d\n", W[0].size[1], W[0].size[2]);
	//exit(0);
	int g_rows = 0;// = sizeof(wino->G) / sizeof(wino->G[0]); 
	int g_col = 0; // = sizeof(wino->G[0]) / sizeof(float);
	
	// printf("winoWeights opened.\n");
	
	WINOGRAD_STRUCT* wino = NULL;
	Tensor *currFilter = NULL; // TODO: check multiple filters handling
	Tensor *transW = new Tensor[output_channels];
	Tensor *InterRes = new Tensor[output_channels];
	
	// printf("Array of Tensors init.\n");
	
	for (size_t filters = 0; filters < output_channels; filters++)
	{
		// printf("Determine dimensions matrix At, Bt, G\n");
		// printf("Size element: %d\n", W[filters].size[1]);
		//wino based on kernel size
		wino = getWino( W[filters].size[1]);
		// printf("End code\n");
		if(( wino->kernel_size == 3) && ( wino->out_size  == 2))
		{
			g_rows = 4;
			g_col = 3;
		}
		else if (( wino->kernel_size == 3) && ( wino->out_size  == 4))
		{
			g_rows = 6;
			g_col = 3;
		}
		else if (( wino->kernel_size == 5) && ( wino->out_size  == 4))
		{
			g_rows = 8;
			g_col = 5;
		}
		else if (( wino->kernel_size == 7) && ( wino->out_size  == 4))
		{
			g_rows = 10;
			g_col = 7;
		}
		else if (( wino->kernel_size == 11) && ( wino->out_size  == 4))
		{
			g_rows = 14;
			g_col = 11;
		}
		currFilter = &W[filters];
		// printf("Dimensions G: checked.\n");
		
		/** Tile information based on current kernel **/
		//printf("\n wino.Bt, filters, wino->kernel_size, wino->out_size, wino->tile_size, wino->tile_stride  %d %d %d %d %d\n",
		//	 filters, wino->kernel_size, wino->out_size, wino->tile_size, wino->tile_stride );

		/** Tile Information **/

		//printf("segfault in constructor---------");					
		//transW[filters] = Tensor(currFilter->size[0], g_rows, currFilter->size[2]);
		//for (size_t i = 0; i < output_channels; i++)
		//{
			
		//}
		//transW ... array of tensors. transW[filters].allocate ...Defines the dim boundaries of the filter-th tensor of transW
		transW[filters].allocate(currFilter->size[0], wino->tile_size,  wino->tile_size);
		InterRes[filters].allocate(currFilter->size[0], wino->tile_size, currFilter->size[1]);

		//Initialise all tensors to zero initially

		//Initialisation ends


		
	//	printf("segfault after constructor---------");
	//Gg ... loop is ok.
	//Computation scheme: compute all column-elements of an row for the output. Switch row only, if all column elements are computed.
		//Switch to next row: cursor jumps to new row in input (here: G). Kernel goes again through all columns row-wise.

		//loop-1 ... go into an channel. Same as feature map.
		for (size_t c = 0; c < currFilter->size[0]; c++)
		{
			// Matrix mul: Gg (g_rows, g_col) * ( size[1] == size[2])
			//G = 4,3  g =3,3 Gt = 3,4, 
			// ( 4,3)
			//loop-2 ... go into row of G. G-values*g_featuremap-values
			for (size_t i = 0; i < g_rows; i++)
			{
				for ( size_t j = 0; j < currFilter->size[1]; j++){

					//transW.data[filters][i][j] = 0;
					//loop-4 ... columns of G. Go along all columns of a row i.
					for (size_t k = 0; k < g_col; k++)
					{
						//TODO: Need intermediate matrix to store G.g
						//transW[filters].data[c][i][j] += wino->G[i][k] * currFilter->data[c][k][j];
						InterRes[filters].data[c][i][j] += wino->G[i][k] * currFilter->data[c][k][j];
					}
				}
			}
		}  

		//Gt for each G, which is selected based on kernel size
		float** Gt = new float*[g_col];
    	for (int i = 0; i < g_col; i++) 
		{
        	Gt[i] = new float[g_rows];
    	}
		transposeMatrix(wino->G, g_rows, g_col, Gt);
	
		//(previous reuslt) * Gt
		for (size_t c = 0; c < InterRes[filters].size[0]; c++)
		{
			// Matrix mul: Gg (g_rows, g_col) * ( size[1] == size[2])
			//G = 4,3  g =3,3 Gt = 3,4
			//transW = (4,3) * Gt = (3,4)
			for (size_t i = 0; i < InterRes[filters].size[1]; i++)
			{
				//Number of cols of Gt = g_rows of G
				for ( size_t j = 0; j < InterRes[filters].size[1]; j++){
					//transW.data[filters][i][j] = 0;
					for (size_t k = 0; k < g_col; k++)
					{ //Gt[j][k]
						transW[filters].data[c][i][j] += InterRes[filters].data[c][i][k] * Gt[k][j];
					}
				}
			}
		}
	}
	#endif 
	#if 1
	// printf("winoWeights opened.\n");
	Tensor *currFilter = NULL; // TODO: check multiple filters handling
	Tensor *transW = new Tensor[output_channels];
	Tensor *InterRes = new Tensor[output_channels];
	
	//######################################################################
	//Transform the different tensors (denoted by filters) of W individually
	//######################################################################
	
	for (size_t filters = 0; filters < output_channels; filters++)
	{
		// printf("Determine dimensions matrix At, Bt, G\n");
		// printf("Size element: %d\n", W[filters].size[1]);
		
		// printf("End code\n");
		
		

		//printf("bt_rows: %d\n", bt_rows);
		currFilter = &W[filters];
		// printf("Dimensions G: checked.\n");
	
			

		//transW ... array of tensors. transW[filters].allocate ...Defines the dim boundaries of the filter-th tensor of transW
		transW[filters].allocate(currFilter->size[0], row_matrix,  row_matrix);
		InterRes[filters].allocate(currFilter->size[0], row_matrix, currFilter->size[1]);


		//Gg ... loop is ok.
		//Computation scheme: compute all column-elements of an row for the output. Switch row only, if all column elements are computed.
		//Switch to next row: cursor jumps to new row in input (here: G). Kernel goes again through all columns row-wise.

		//########################################################################################
		//Gg / B.Td / A.Tm ... Individual feature map of chosen Tensor gets seperatly transformed
		//########################################################################################

		//loop-1 ... go into an channel. Same as feature map.
		for (size_t c = 0; c < currFilter->size[0]; c++)
		{
	
			//loop-2 ... go into row of G. G-values*g_featuremap-values
			for (size_t i = 0; i < row_matrix; i++)
			{
				for ( size_t j = 0; j < currFilter->size[1]; j++){

					//transW.data[filters][i][j] = 0;
					//loop-4 ... columns of G. Go along all columns of a row i.
					for (size_t k = 0; k < column_matrix; k++)
					{
						if(select == 0)
							InterRes[filters].data[c][i][j] += wino->Bt[i][k] * currFilter->data[c][k][j];
						else
							InterRes[filters].data[c][i][j] += wino->At[i][k] * currFilter->data[c][k][j];

					}
				}
			}
		}  

		//###############################################################
		//G.T, B, A ... Compute transposed of the Winograd-Transf. matrix
		//################################################################

		//Gt for each G, which is selected based on kernel size
		float** Gt = new float*[column_matrix]; //column_matrix, row_matrix
    	for (int i = 0; i < column_matrix; i++) 
		{
        	Gt[i] = new float[row_matrix];
    	}
		if(select == 0)
			transposeMatrix(wino->Bt, row_matrix, column_matrix, Gt);
		else
			transposeMatrix(wino->At, row_matrix, column_matrix, Gt);

		//##################################################################
		//(Gg)*G.T / (B.Td)*B / (A.Tm)*A ... compute Winograd Transformation
		//##################################################################


		//(previous result) * Gt
		for (size_t c = 0; c < InterRes[filters].size[0]; c++)
		{
			for (size_t i = 0; i < InterRes[filters].size[1]; i++)
			{
				//Number of cols of Gt = g_rows of G
				for ( size_t j = 0; j < InterRes[filters].size[1]; j++){
					//transW.data[filters][i][j] = 0;
					for (size_t k = 0; k < column_matrix; k++)
					{
						transW[filters].data[c][i][j] += InterRes[filters].data[c][i][k] * Gt[k][j];
					}
				}
			}
		}
		//delete [] InterRes;
		// Free the memory for each row
		for (int i = 0; i < column_matrix; i++) {
		    delete[] Gt[i];
		}
		// Free the memory for the array of row pointers
		delete[] Gt;
	}

	// Initialize all elements of transW to 0
	#if 0
	for (size_t filters = 0; filters < output_channels; filters++)
	{
		currFilter = &W[filters];
		//transW.resize(output_channels, g_rows, currFilter->size[1]);
		//transW[filters] = new Tensor(currFilter->size[0], g_rows, currFilter->size[1]);

		for (size_t c = 0; c < currFilter->size[0]; c++)
		{
			for (size_t i = 0; i < g_rows; i++)
	    	{
	    	    for (size_t j = 0; j < currFilter->size[1]; j++)
	    	    {
	    	        transW[filters]->data[c][i][j] = 0.0; // Or the desired initial value
	    	    }
	    	}
		}
	}
	#endif
	
	//free memory space
	delete [] InterRes;
	
	//printf("DONEEEEEEEEEEE");
	return transW;
	#endif 

	//return transW;
}

float check_decimal(float value)
{
    //result ... initialize output function. 
		//correct #tile_for_feature_map
	float result = 0;
	//fractionalPart ... gives the value of the fractional part of the float.
    float fractionalPart = std::fmod(value, 1.0);

	if(fractionalPart){
		result = fractionalPart + value;
	}
	else{
		result = value;
	}
    //return ... returns the fractional part
    return result;
}


/*
 * Convolution using inputs and converted Weights
 * Tensor 			*U_wino	: Transformed Weight Tensor
 * Tensor			*B		: Bias 
 * Tensor			*Z		: Output Tensor
 * int 			k_size		: Width and Height of weight kernel
 */
void convWinograd(Tensor * X, Tensor * U_wino , Tensor * B, Tensor * Z, int k_size)
{
//##############################################
	//Init and definition of variables
	//##############################################

	//#1 transform input tile
		//&transformed_tile = winoWeights(untransformed_tile)
	//#2 elementwise multiplication
		//result = transformed_tile x U_wino
	//#3 m = sum(result) ... get summary over whole feature extraction by kernel. Transformed space
	//#4 untransformed A.TmA
		//&A.TmA = winoWeight(m) 
	
	//wino ... At, Bt, G, ...
		//k_size ... size of weights (there individual feature maps).
			//Mind: all output Z feature maps should have the same dimensions. Therefore the weight dim over all weights must be the same.
			//k_size ... usable for all weights (tensor) of given U_wino array of tensors U_wino.  

	WINOGRAD_STRUCT* wino = getWino(k_size);


	//int numRows = inputMatrix.size();
    //int numCols = inputMatrix[0].size();

    // Compute the number of tiles
		//numTilesRows_unchecked ... first estimation number of tiles. Problem: partly catched tiles are resembeled only as decimal places.
		//(static_cast<float>(a) + static_cast<float>(b))/ static_cast<float>(c)
    float numTilesRows_unchecked = ((static_cast<float>(X->size[1]) - static_cast<float>(wino->tile_size)) / static_cast<float>(wino->tile_stride)) +1;
    float numTilesCols_unchecked = ((static_cast<float>(X->size[2]) - static_cast<float>(wino->tile_size)) / static_cast<float>(wino->tile_stride)) +1;

	//numTilesRows ... total amount of tiles. #tile
		//improvement to numTilesRows_unchecked ... only partly considered tiles by the area of the input feature map gets fully considered now.
	float numTilesRows = check_decimal(numTilesRows_unchecked);
	float numTilesCols = check_decimal(numTilesCols_unchecked);
	
	int output_size = 1;
	//DEBUGGING HELP

	// printf("\n");
	// printf("X->size[1]: %d\n", X->size[1]);
	// printf("l ... input tile size: %d\n", wino->tile_size);
	// printf("ls ... stride input tile: %d\n", wino->tile_stride);
	
	// printf("numTilesRows_unchecked: %f\n", numTilesRows_unchecked);
	// printf("numTilesCols_unchecked : %f\n", numTilesCols_unchecked);

	// printf("numTilesRows: %d\n", numTilesRows);
	// printf("numTilesCols: %d\n", numTilesCols);
	// printf("\n");

	
	// printf("Number of Tiles for Row: %d\n", numTilesRows);
	// printf("Number of Tiles for Cols: %d\n", numTilesCols);
	

	int at_rows = 0;
	int at_col = 0;

	int g_rows  =0;
	int g_col = 0;

	int bt_rows = 0;
	int bt_col = 0;

	if(( wino->kernel_size == 3) && ( wino->out_size  == 2))
	{
		at_rows = 2;
		at_col = 4;
			
		g_rows = 4;
		g_col = 3;

		bt_rows = 4;
		bt_col = 4;
	}
	else if (( wino->kernel_size == 3) && ( wino->out_size  == 4))
	{
		at_rows = 4;
		at_col = 6;
		
		g_rows = 6;
		g_col = 3;

		bt_rows = 6;
		bt_col = 6;
	}
	else if (( wino->kernel_size == 5) && ( wino->out_size  == 4))
	{
		at_rows = 4;
		at_col = 8;

		g_rows = 8;
		g_col = 5;

		bt_rows = 8;
		bt_col = 8;
	}
	else if (( wino->kernel_size == 7) && ( wino->out_size  == 4))
	{
		at_rows = 4;
		at_col = 10;

		g_rows = 10;
		g_col = 7;

		bt_rows = 10;
		bt_col = 10;
	}
	else if (( wino->kernel_size == 11) && ( wino->out_size  == 4))
	{
		at_rows = 4;
		at_col = 14;

		g_rows = 14;
		g_col = 11;

		bt_rows = 14;
		bt_col = 14;
	}

	// printf("bt_rows: %d\n", bt_rows);
	
	//tile ... tensor. Contains the space for tiles of one position over all feature maps
	Tensor *tile = new Tensor(X->size[0], wino->tile_size, wino->tile_size);
	// printf("All variables defined\n");


	// Loop over the tiles
 
	//#############################################
	//extract the tiles from the feature maps 
	//#############################################

	for (int tileRow = 0; tileRow < numTilesRows; tileRow++) {
		for (int tileCol = 0; tileCol < numTilesCols; tileCol++) {
			

			//#################################################################
			//determine starting point (startRow, startCol) of current tile 
			//#################################################################
			
			// Compute the starting and ending indices for the current tile
			//startRow ... gets #1 element coordinates (pixel-bases) of current tile
				//Subnote 1: coordinate of input feature map
				//Subnote 2: tileRow are tile-stride-based coordinates for #1 element of current tile
				//Subnote 3: tile_stride is used to get the #1 element tile. tile_size is used to get all other elements of the tile, which is positioned at the determined position.
			int startRow = tileRow * wino->tile_stride;
			int endRow = startRow + wino->tile_size;
			int startCol = tileCol * wino->tile_stride;
			int endCol = startCol + wino->tile_size;
			// Check: do the area of the current tile stratches over the border of the current input feature map?
				//Subnote 1: endRow, endCol ... both determined out of startRow, startCol. startRow, startCol both elements feature map. endRow, endCol do not must be elements feature map
				//Subnote 2: if endRow, endCol reach over boundaries - the value gets set to the biggest dim value e.g., X->size[1] for Row.
			if (endRow > X->size[1]) endRow = X->size[1];
			if (endCol > X->size[2]) endCol = X->size[2];


			//############################################################################################################
			//extract the tiles from the feature maps + for one starting point (startRow, startCol) over all feature maps 
			//############################################################################################################
			
			// Loop over the feature maps
			for (int featureMap = 0; featureMap < X->size[0]; featureMap++) {

				//##########################################################
				//copy values from input feature map into tile ... or set 0
				//##########################################################

					//##############
					//normal values
					//##############

				// Process the current tile for the current feature map
					//Subnote 1: tile and input have same amount feature maps
					//Subnote 2: #1 go into feature map. #2 copy from inpute into tile.
					//Subnote 3: (startRow, startCol) ... starting point of each copy-operation (per feature map)
				for (int row = startRow; row < endRow; row++) {
					for (int col = startCol; col < endCol; col++) {
						//row = [startRow, endRow], col = [startCol, endCol]
							//Subnote 1: startRow, endRow, row ... all are coordinate based on feature image.
							//Subnote 2: tile coordinates smaller input coordinate X. Therefore row-startRow, col-startCol
							//Subnote 3: row >= startRow, col >=startCol
						tile->data[featureMap][row-startRow][col-startCol] = X->data[featureMap][row][col];
					}
				}
				
					//##############
					//0 values
					//##############

				//execute only, if #1 and #2 definition endRow and endCol are unqual (!=)
					//Subnote 1: loop fills remaining unfilled coordinates with 0
					//Subnote 2: tile is a tensor with a defined amount of rows and columns.
					//Subnote 3: if endRow != startRow+tile_size, some coordinates will be not filled after the first loop.
				if((endRow != (startRow + wino->tile_size))|| (endCol != (startCol + wino->tile_size))){
					for (int row = 0; row < wino->tile_size; row++) {
						for (int col = 0; col < wino->tile_size; col++) {
							//Filling unfilled tile feature map areas with 0
								//Subnote 1: endRow gets defined twice
								//Subnote 2: #1 endRow = startRow + tile_size
								//Subnote 3: #2 endRow = X->size[1] ... if #1 endRow > X->size[1]
							if (row >= (endRow-startRow) || col >= (endRow-startRow)) {
								tile->data[featureMap][row][col] = 0;
							}
						}
					}
				}
			}

			//#################################
			//Transform input tile
			//#################################
			
			
			Tensor * tile_trans = winoTile(tile, 1, wino, bt_rows, bt_col, 0);
			

			//#####################################
			//Compute m ... transformed output tile
			//#####################################

			//numFeatureMaps ... amount feature maps of transformed tiles
			//numRows ... amount rows transformed tiles
			//numCols ... amount coloumns transformed tiles.
				//Subnote 1: tile_trans is an array of tensors.
				//Subnote 2: tile_trans contains the transformed tile. 
				//Subnote 3: tile_trans contains data "(featuremap, startRow, startCol)"
			int numFeatureMaps =  tile_trans->size[0];
			int numRows = tile_trans->size[1];
			int numCols = tile_trans->size[2];
			
			//m ... array of tensors.
				//Subnote 1: after defintion ... contains for each weight (element W) the multiplication result
			
			Tensor *m = new Tensor[Z->size[0]];
			for(int i = 0; i < Z->size[0]; i++){
				m[i].allocate(numFeatureMaps, numRows, numCols);
			}
			
			//CurrWeight ... Z->size[0] x different weights in U_wino. For each weight-Tensor own iteration. 
				//Subnote 1: pick one weight tensor
			for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){
				// Perform element-wise multiplication for each feature map
					//Subnote 1: pick current feature map. Element-wise multiplication.
				for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
					for (int row = 0; row < numRows; row++) {
						for (int col = 0; col < numCols; col++) {
							// Perform element-wise multiplication between corresponding feature maps
								//Subnote 1: CurrWeight ... amount of different weights considered. Amount feature maps output.
								//Subnote 2: featureMap ... amount feature maps input.
							m[CurrWeight].data[featureMap][row][col] = tile_trans->data[featureMap][row][col] * U_wino[CurrWeight].data[featureMap][row][col];
						}
					}
				}
			}
			delete [] tile_trans;
			
			//#########################################################################################
			//Compute the sums over all individual weights. Store sums as feature maps in tensore m_sum
			//#########################################################################################

			//m_sum ... Tensor with one feature map. All feature maps of Tensor m get element wise summed together. 
			Tensor m_sum(Z->size[0], numRows, numCols);
			//loop ... compute the sum over all layers
			for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){
				for (int row = 0; row < numRows; row++) {
					for (int col = 0; col < numCols; col++) {
						for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
							// Accumulate the sum for each location
							m_sum.data[CurrWeight][row][col] += m[CurrWeight].data[featureMap][row][col];
						}
					}
				}
			}

			delete [] m;

			//####################################
			//Untransformed output tile m -> ATmA
			//####################################

				//#####################################################
				//Predefinition array of tensors (input transformation)
				//#####################################################


				//##############################
				//m -> A.TmA ... transformation
				//#############################

			//ATma ... untransformed output tiles. An Array of Tensor with one Tensor. The one Tensor contains only one feature map.
				//m_trans ... transformed output tiles
				//1 ... output_size. There is only one Tensor in the Array of Tensors.
				//wino ... contains At and kernel_size
				//at_rows ... rows of At
				//at_col ... columns of At
				//1 ... select. Select in wino matrix At to use. Wino contains several matrices - no confusion.
			//Tensor * ATmA = winoTile(m_trans, 1, wino, at_rows, at_col, 1);
			Tensor * ATmA = winoTile(&m_sum, 1, wino, at_rows, at_col, 1);
			//delete [] m_trans;
			// printf("U_wino last element ... amount feature maps: %d\n", U_wino[Z->size[0]-1].size[0]);

			//#########################
			//Compute output tensor Z
			//#########################

			//To-Do: add the "0-adding in the kernel dismissing"

			int startRow_fm = tileRow*wino->out_size;
			int startCol_fm = tileCol*wino->out_size;

			int numRowsZ = Z->size[1];
			int numColsZ = Z->size[2]; 

			for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){
				//loop ... defining output tensor Z. 
				for (int row = 0; row < wino->out_size && startRow_fm + row < numRowsZ; row++) {
					for (int col = 0; col < wino->out_size && startCol_fm + col < numColsZ ; col++) {
						
						// Calculate the indices in the feature map
						int featureMapRow = startRow_fm + row;
						int featureMapCol = startCol_fm + col;

						// Assign the value from the tile to the corresponding position in the feature map
						//Z->data[CurrWeight][featureMapRow][featureMapCol] ... output tensor Z
						//ATmA[0].data[0][row][col] ... Array of Tensors ATma contains only one Tensor ATmA[0] with only one fm ATmA.data[0][y][x]

						Z->data[CurrWeight][featureMapRow][featureMapCol] = ATmA[0].data[CurrWeight][row][col];
					}
				}	
			}
			delete [] ATmA;
		}	
	}
	delete tile;
	//#############################################
	//Add B to output Z (Winograd = WX, not WX + B)
	//#############################################

	//loop ... add bias B to the CONV.
	//channels output
	for(int o=0; o < Z->size[0]; o++){
		//height of channel in Z
		for(int p = 0; p < Z->size[1]; p++){
		//width in channel in Z
			for(int q=0; q < Z->size[2]; q++){
				Z->data[o][p][q] += (B->data[0][0][o]);
			}	
		}
	}
	//printf("B added.\n");
}
	








/*-------------------------------- FFT  -----------------------------------------------*/
/*
 * Pre Transform the weights depending on the tile size
 * FFT_STRUCT 		*fft	: Struct containing tile size (N), overlap and stride
 * Tensor 			*W		: Untransformed Weight Tensor
 * int		output_channels	: Number of output channels
 * Return:		C_Tensor *	: New Tensor containing transformed Weights
 */

C_Tensor * fftWeights(Tensor * W, int output_channels)
{

	//fft ... compute the tile size, stride, oberlap
		//Subnote 1: all weight tensors are applied on the same input.
		//Subnote 2: all possess the same amount of channels.
		//Subnote 3: all weight tensors do possess same kernel size. All output feature maps must be equal by size.
	const FFT_STRUCT* fft = NULL;
	fft  = getFFT(W[0].size[1]);
	//printf("Hello1\n");
	//currFilter ... address currently observed weight
	Tensor* currFilter = NULL;
	//printf("Hello2\n");
	//temp_bfrPad ... results flipping and padding
		//Subnote 1: content is flipped and padded weight. 
		//Subnote 2: still in t-space. Input into FFT. For transforming into f-space.
	//printf("Hello3\n");
	C_Tensor* temp_afterPad_bfrFFT = NULL;
	temp_afterPad_bfrFFT = new C_Tensor();
	//printf("Hello4\n");
	temp_afterPad_bfrFFT->allocate(1, fft->tile_size, fft->tile_size);

	//cout<<"Tile Size: "<<fft->tile_size<<endl;

	//printf("Hello5\n");
	//temp2 ... Pointer on Tensor.
		//Subnote 1: successor of temp_bfrPad
		//Subnote 2: temp_bfrPad contains one 2D fm of input weight (one Tensor). Flipped and Padded. Not transformed.
		//Subnote 3: temp2 ... contains FFT version of temp_bfrPad
	C_Tensor* temp_FFT = new C_Tensor;
	temp_FFT->allocate(1, fft->tile_size, fft->tile_size);

	C_Tensor* U_fft = new C_Tensor[output_channels];

	//C_Tensor* currFilter_padded = new C_Tensor[output_channels];


	for (int filters = 0; filters < output_channels; filters++)
	{
		//currFilter ... get address current weight.	
			//Subnote 1: weight is unflipped and unweighted
		
		currFilter = &W[filters];
		//printf("Hello\n.");
		//U_fft ... array of Tensor Output of function
			//Subnote 1: all elements possess the same dimensions.
			//Subnote 2: weights get #1 flipped, #2 padded
			//Subnote 3: flipping in original 3D shape, padding from weight to tile shape.
		U_fft[filters].allocate(W[0].size[0], fft->tile_size, fft->tile_size);
		//printf("Hello\n.");
		//loop ... #1 Flipping + Padding, #2 FFT
			//Subnote 1: takes as input original weight tensor (1 weight tensor)
			//Subnote 2: works feature map wise.
			//Subnote 3: individual fm gets flipped, padded, FFT and then put into output. Then only next fm weight.
		for (int c = 0; c < W[0].size[0]; c++ )
		{
			/***************** FLIPPING AND PADDING OF WEIGHTS *********************/

			flip_Matrix(currFilter, temp_afterPad_bfrFFT, c );
			
			/***************** FFT OF WEIGHTS *********************/

			// Perform 2D FFT and store result in U_fft
			fft2d(temp_afterPad_bfrFFT, temp_FFT);
			for (size_t j = 0; j < fft->tile_size; j++)
			{
				for (size_t k = 0; k < fft->tile_size; k++)
				{
					U_fft[filters].data[c][j][k] = temp_FFT->data[0][j][k];
					
				}

			}

		}

	}
	
	delete temp_FFT;
	delete temp_afterPad_bfrFFT;
    return U_fft;
}

// //######################################
// //convFFT: ECE and SATHVIK
// //######################################

// void convFFT(Tensor * X, C_Tensor * U_fft, Tensor * B, 
//         Tensor * Z, int k_size)
// {

// 	/*for(int i=0; i<B->size[0]; i++){
// 		for(int j=0; j<B->size[1]; j++){
// 			for(int k=0; k<B->size[2]; k++){
// 				cout<<B->data[i][j][k]<<" ";
// 			}
// 			cout<<endl;
// 		}
// 		cout<<endl;
// 	}*/
// 	//exit(0);
// 	// Pick tile size
//     const FFT_STRUCT *fft = getFFT(k_size);
//     int tile_size = fft->tile_size; // Size of tile including overlap (N)
// 	int overlap = fft->overlap; // Size of overlap = w.x - 1 (M-1)
// 	int tile_stride = fft->tile_stride; // Size of tile without overlap (N-(M-1))

// 	int num_input_channels = X->size[0];
//     int input_height = X->size[1];
//     int input_width = X->size[2];

//     int num_output_channels = Z->size[0];
//     int output_height = Z->size[1];
//     int output_width = Z->size[2];

// 	int pad_amount_row = 0;
// 	int pad_amount_column = 0;
				
// 	//create input tiles:
// 	float numTilesRows_unchecked = ((static_cast<float>(X->size[1])) / static_cast<float>(tile_stride));
//     float numTilesCols_unchecked = ((static_cast<float>(X->size[2])) / static_cast<float>(tile_stride));

// 	int numTilesRows = check_decimal(numTilesRows_unchecked);
// 	int numTilesCols = check_decimal(numTilesCols_unchecked);

	
// 	//cout<<"overlap :"<<overlap<<endl;
// 	////cout<<"tile size: "<<tile_size<<endl;
// 	////cout<<"input h w: "<<X->size[1]<<endl;
// 	////cout<<"# Tiles Rows: "<<numTilesRows<<endl;
// 	////cout<<"# Tiles Cols: "<<numTilesCols<<endl;

// 	for(int i=0; i<Z->size[0]; i++){
// 		for(int j=0; j<Z->size[1]; j++){
// 			for(int k=0; k<Z->size[2]; k++){
// 				Z->data[i][j][k] = 0.0;
// 			}
// 		}
// 	}

// 	C_Tensor *tile = new C_Tensor(X->size[0], tile_size, tile_size);

// 	//CurrWeight ... Z->size[0] x different weights in U_fft. For each weight-Tensor own iteration. 
// 	for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){	//for each weight 3d

// 		//for each tile
// 		for (int tileRow = 0; tileRow < numTilesRows; tileRow++) {
// 			for (int tileCol = 0; tileCol < numTilesCols; tileCol++) {

// 				for(int i=0; i<tile->size[0]; i++){
// 					for(int j=0; j<tile->size[1]; j++){
// 						for(int k=0; k<tile->size[2]; k++){
// 							tile->data[i][j][k] = 0.0;
// 						}
// 					}
// 				}
			
// 				int startRow = tileRow * fft->tile_stride;
// 				int endRow = startRow + fft->tile_size;
// 				int startCol = tileCol * fft->tile_stride;
// 				int endCol = startCol + fft->tile_size;
			
// 				if (endRow > X->size[1]) endRow = X->size[1];
// 				if (endCol > X->size[2]) endCol = X->size[2];

// 				/*//cout<<"Start Row: "<<startRow<<endl;
// 				//cout<<"Start Col: "<<startCol<<endl;
// 				//cout<<"End Row: "<<endRow<<endl;
// 				//cout<<"End Col: "<<endCol<<endl;*/

// 				//for each input channel
// 				for (int featureMap = 0; featureMap < X->size[0]; featureMap++) {
// 					for (int row = startRow; row < endRow; row++) {
// 						for (int col = startCol; col < endCol; col++) {
// 							tile->data[featureMap][row-startRow][col-startCol] = X->data[featureMap][row][col];
// 						}
// 					}
					
// 				}

// 				/*if (CurrWeight==1){
// 				cout<<"original data: "<<endl;
// 				for(int featureMap=0; featureMap<X->size[0]; featureMap++){
// 				for (int row = startRow; row < endRow; row++) {
// 					for (int col = startCol; col < endCol; col++) {
// 						cout<<X->data[featureMap][row][col]<<" ";
// 					}
// 					cout<<endl;
// 				}
// 				cout<<endl;
// 				}

// 				cout<<"tile2 after padding: "<<endl;
// 					for(int featureMap=0; featureMap<X->size[0]; featureMap++){
// 					for (int row = 0; row < fft->tile_size; row++) {
// 						for (int col = 0; col < fft->tile_size; col++) {
// 							cout<<tile->data[featureMap][row][col]<<" ";
// 						}
// 						cout<<endl;
// 					}
// 					cout<<endl;
// 					}
// 				}*/

// 				int input_size = X->size[0]; 
// 				C_Tensor *tile_fft = new C_Tensor[1];
// 				C_Tensor *temp_fft = new C_Tensor[1];
// 				C_Tensor *tile2d = new C_Tensor[1];

// 				tile2d[0].allocate(1, fft->tile_size, fft->tile_size);
// 				temp_fft[0].allocate(1, fft->tile_size, fft->tile_size);
// 				tile_fft[0].allocate(X->size[0], fft->tile_size, fft->tile_size);

// 				for(int c=0; c<X->size[0]; c++){			
// 					for(int j=0; j<fft->tile_size; j++){
// 						for(int k=0; k<fft->tile_size; k++){
// 							tile2d[0].data[0][j][k] = tile[0].data[c][j][k];
// 						}
// 					}

// 					/*if (CurrWeight==1){
// 					cout<<"tile2d channel: "<<c<<endl;
// 					for(int j=0; j<fft->tile_size; j++){
// 						for(int k=0; k<fft->tile_size; k++){
// 							cout<<tile2d[0].data[0][j][k]<<" ";
// 						}
// 						cout<<endl;
// 					}
// 					cout<<endl;
// 					}*/

// 					fft2d(&tile2d[0], &temp_fft[0]);

// 					/*#if 1
// 					if (CurrWeight==1){
// 					cout<<"fft channel: "<<c<<endl;
// 					for(int j=0; j<fft->tile_size; j++){
// 						for(int k=0; k<fft->tile_size; k++){
// 							cout<<temp_fft[0].data[0][j][k]<<" ";
// 						}
// 						cout<<endl;
// 					}
// 					cout<<endl;
// 					}*/

// 					for(int j=0; j<fft->tile_size; j++){
// 						for(int k=0; k<fft->tile_size; k++){
// 							tile_fft[0].data[c][j][k] = temp_fft[0].data[0][j][k];

// 						}
// 					}

// 					/*if (CurrWeight==1){
// 					cout<<"tile fft channel: "<<c<<endl;
// 					for(int j=0; j<fft->tile_size; j++){
// 						for(int k=0; k<fft->tile_size; k++){
// 							cout<<tile_fft[0].data[c][j][k]<<" ";
// 						}
// 						cout<<endl;
// 					}
// 					cout<<endl;
// 					}*/
// 				}

// 				#if 0
// 				//cout<<"Tile fft for all channels: "<<endl;
// 				for(int c = 0; c<X->size[0]; c++){
// 					//cout<<"Channel: "<<c<<endl;
// 					for(int j=0; j<fft->tile_size; j++){
// 						for(int k=0; k<fft->tile_size; k++){
// 							//cout<<tile_fft[0].data[c][j][k]<<" ";
// 						}
// 					}
// 					//cout<<endl;
// 				}
// 				//cout<<endl;
// 				#endif 

// 				delete [] temp_fft;
// 				delete [] tile2d;

// 				int numFeatureMaps =  tile_fft->size[0];
// 				int numRows = tile_fft->size[1];
// 				int numCols = tile_fft->size[2];
				
// 				//m ... result element-wise multiplication. Transformed m
// 				C_Tensor m(numFeatureMaps, numRows, numCols);

// 				// Perform element-wise multiplication for each feature map
// 				for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
// 					for (int row = 0; row < numRows; row++) {
// 						for (int col = 0; col < numCols; col++) {
// 							// Perform element-wise multiplication between corresponding feature maps
// 							m.data[featureMap][row][col] = tile_fft->data[featureMap][row][col] * U_fft[CurrWeight].data[featureMap][row][col];
// 						}
// 					}
// 				}

// 				for(int c = 0; c<numFeatureMaps; c++){

// 					/*if (CurrWeight==1){
// 					cout<<"element wise mult channel: "<<c<<endl;
// 					for(int j=0; j<fft->tile_size; j++){
// 						for(int k=0; k<fft->tile_size; k++){
// 							cout<<m.data[c][j][k]<<" ";
// 						}
// 						cout<<endl;
// 					}
// 					cout<<endl;
// 					}

// 					#if 0
// 					//cout<<"U_fft channel: "<<c<<endl;
// 					for (int row = 0; row < numRows; row++) {
// 							for (int col = 0; col < numCols; col++) {
// 								//cout<<U_fft->data[c][row][col]<<" ";
// 							}
// 							//cout<<endl;
// 					}
// 					//cout<<endl;
// 					#endif 

// 					#if 0
// 					//cout<<"Multiplication result channel: "<<c<<endl;
// 					for (int row = 0; row < numRows; row++) {
// 							for (int col = 0; col < numCols; col++) {
// 								//cout<<m.data[c][row][col]<<" ";
// 							}
// 							//cout<<endl;
// 					}
// 					//cout<<endl;
// 				#endif*/
				
// 				}

// 				C_Tensor m_sum(1, numRows, numCols);
// 				C_Tensor ifft_sum(1, numRows, numCols);


// 				for (int row = 0; row < numRows; row++) {
// 					for (int col = 0; col < numCols; col++) {
// 						for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
// 							// Accumulate the sum for each location
// 							m_sum.data[0][row][col] += m.data[featureMap][row][col];
// 						}
// 					}
// 				}

// 				#if 0
// 				//cout<<"summation result: "<<endl;
// 				for (int row = 0; row < numRows; row++) {
// 					//cout<<"[ ";
// 						for (int col = 0; col < numCols; col++) {
// 							//cout<<m_sum.data[0][row][col]<<" ";
// 						}
// 						//cout<<"] ";	
// 						//cout<<endl;
// 				}
// 				//cout<<endl;
// 				#endif

// 				ifft2d(&m_sum, &ifft_sum);

// 				#if 0
// 				//cout<<"ifft result: "<<endl;
// 				for (int row = 0; row < numRows; row++) {
// 					//cout<<"[ ";
// 						for (int col = 0; col < numCols; col++) {
// 							//cout<<ifft_sum.data[0][row][col]<<" ";
// 						}
// 						//cout<<"] ";
// 						//cout<<endl;
// 				}
// 				//cout<<endl;
// 				#endif
				
// 				C_Tensor * output_tile = new C_Tensor[1];
// 				pad_amount_row = - endRow + startRow + fft->tile_size;
// 				pad_amount_column = - endCol + startCol + fft->tile_size;
				
// 				int row_size = ifft_sum.size[1] - pad_amount_row - overlap;
// 				int column_size = ifft_sum.size[2] - pad_amount_column  - overlap;
	
// 				output_tile[0].allocate(1, row_size, column_size);

// 				int r = 0, c = 0;
				
// 				if(tile_size-pad_amount_column-overlap<=0) cout<<"aaaaaaa"<<endl;
// 				if(tile_size-pad_amount_row-overlap<=0) cout<<"bbbbb"<<endl;

// 				for(int rows = overlap; rows < (ifft_sum.size[1] - pad_amount_row); rows++){
// 						c = 0;
// 					for(int cols = overlap; cols < (ifft_sum.size[2] - pad_amount_column); cols++){
// 							output_tile[0].data[0][r][c].real(ifft_sum.data[0][rows][cols].real());
// 						c++;

// 					}
// 						r++;
// 				}

// 				#if 0
// 				//cout<<"output tile: "<<endl;
// 				for (int row = 0; row < r; row++) {
// 					//cout<<"[ ";
// 						for (int col = 0; col < c; col++) {
// 							//cout<<output_tile[0].data[0][row][col]<<" ";
// 						}
// 						//cout<<"] ";
// 						//cout<<endl;
// 				}
// 				//cout<<endl;
// 				#endif

// 				for(int i=0; i<output_tile->size[1]; i++){
// 					for(int j=0; j<output_tile->size[2]; j++){
// 						Z->data[CurrWeight][tileRow*(fft->tile_size-fft->overlap)+i][tileCol*(fft->tile_size-fft->overlap)+j] = output_tile[0].data[0][i][j].real()+B->data[0][0][CurrWeight];
// 					}
// 				}

// 				#if 0
// 				//cout<<"all z for one weight:"<<endl;	
// 				for(int i=0; i<Z->size[1]; i++){
// 					for(int j=0; j<Z->size[2]; j++){
// 						//cout<<Z->data[CurrWeight][i][j]<<" ";
// 					}
// 					//cout<<endl;
// 				}
// 				#endif
// 				delete [] tile_fft;
// 				delete [] output_tile;
// 			}
// 		}
// 	}

// 	////cout<<"Z: "<<endl;
// 	/*for(int c=0; c<Z->size[0];c++){
// 		for(int i=0; i<Z->size[1]; i++){
// 			for(int j=0; j<Z->size[2]; j++){
// 				Z->data[c][i][j] += B->data[0][0][0];
// 				////cout<<Z->data[c][i][j]<<" ";
// 			}
// 			//cout<<endl;
// 		}

// 		//cout<<endl;
// 	}*/

// 	delete tile;
// }




//###########################################
//convFFT: PHILIPP
//###########################################



/*
 * Convolution using inputs and converted Weights
 * FFT_STRUCT *fft : Struct containing tile size (N), overlap and stride
 * C_Tensor *U_fft : Complex Transformed Weight Tensor
 * Tensor *B : Bias 
 * Tensor *Z : Output Tensor
 * int k_size : Width and Height of weight kernel
 */





void convFFT(Tensor * X, C_Tensor * U_fft, Tensor * B, 
        Tensor * Z, int k_size)
{

	// Pick tile size
    const FFT_STRUCT *fft = getFFT(k_size);
    int tile_size = fft->tile_size; // Size of tile including overlap (N)
	int overlap = fft->overlap; // Size of overlap = w.x - 1 (M-1)
	int tile_stride = fft->tile_stride; // Size of tile without overlap (N-(M-1))

	int num_input_channels = X->size[0];
    int input_height = X->size[1];
    int input_width = X->size[2];

    int num_output_channels = Z->size[0];
    int output_height = Z->size[1];
    int output_width = Z->size[2];

	int pad_amount_row = 0;
	int pad_amount_column = 0;
				
	//create input tiles:
	float numTilesRows_unchecked = ((static_cast<float>(X->size[1])) / static_cast<float>(tile_stride));
    float numTilesCols_unchecked = ((static_cast<float>(X->size[2])) / static_cast<float>(tile_stride));

	int numTilesRows = check_decimal(numTilesRows_unchecked);
	int numTilesCols = check_decimal(numTilesCols_unchecked);


	C_Tensor *tile = new C_Tensor(X->size[0], tile_size, tile_size);
	//m ... result element-wise multiplication. Transformed m
		//Subnote 1: for current tile all weights get applied. 
		//Subnote 2: For all weights all output feature maps get computed.
		//Subnote 3: all results get stored in m
	C_Tensor *m = new C_Tensor[Z->size[0]];
	
	for(int CurrWeight= 0; CurrWeight < Z->size[0]; CurrWeight++){
		m[CurrWeight].allocate(X->size[0], fft->tile_size, fft->tile_size);
	}

	//CurrWeight ... Z->size[0] x different weights in U_fft. For each weight-Tensor own iteration. 
	//for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){	//for each weight 3d

	//for each tile
	for (int tileRow = 0; tileRow < numTilesRows; tileRow++) {
		for (int tileCol = 0; tileCol < numTilesCols; tileCol++) {
			
			//############################
			//Definition of current tile
			//############################

			//Subnote 1: tile is copy from the corresponding location in the image
			//Subnote 2: tile contains all fm of the input image.

			

				//############################
				//Init: 0 values. 
				//############################

					//Subnote 1: fill elements of the tile with 0.
					//Subnote 2: after all elements filled with 0: overwrite with actual copied values from the input.
					//Subnote 3: possible improvement ... no 2 overwrittings in one definition step.
			//loop ... add values over all feature maps of the tile (certain position in input image).
			for(int i=0; i<tile->size[0]; i++){
				for(int j=0; j<tile->size[1]; j++){
					for(int k=0; k<tile->size[2]; k++){
						tile->data[i][j][k] = 0.0;
					}
				}
			}

			int startRow = tileRow * fft->tile_stride;
			int endRow = startRow + fft->tile_size;
			int startCol = tileCol * fft->tile_stride;
			int endCol = startCol + fft->tile_size;

			if (endRow > X->size[1]) endRow = X->size[1];
			if (endCol > X->size[2]) endCol = X->size[2];

				//############################
				//normal values 
				//############################

			//for each input channel
			for (int featureMap = 0; featureMap < X->size[0]; featureMap++) {
				for (int row = startRow; row < endRow; row++) {
					for (int col = startCol; col < endCol; col++) {
						tile->data[featureMap][row-startRow][col-startCol] = X->data[featureMap][row][col];
					}
				}
			}

			//##########################
			//FFT: transform tile. T->F.
			//##########################
			
			//Define needed elements
			int input_size = X->size[0]; 
			
			C_Tensor *tile2d = new C_Tensor();
			C_Tensor *temp_fft = new C_Tensor();
			C_Tensor *tile_fft = new C_Tensor();
			
			tile2d->allocate(1, fft->tile_size, fft->tile_size);
			temp_fft->allocate(1, fft->tile_size, fft->tile_size);
			tile_fft->allocate(X->size[0], fft->tile_size, fft->tile_size);
			
			//loop ... copy current fm c from tile into tile2d
				//Subnote 1: fm is not transformed. Is an element of time domain
				//Subnote 2: tile2d ... takes feature map and leads it to transformation.

			for(int c=0; c<X->size[0]; c++){			
				for(int j=0; j<fft->tile_size; j++){
					for(int k=0; k<fft->tile_size; k++){
						tile2d->data[0][j][k] = tile->data[c][j][k];
					}
				}
			
				//FFT
					//Subnote 1: fft2d ... FFT only for one feature map at a time
					//Subnote 2: temp_fft ... contains transformed feature map.
				fft2d(tile2d, temp_fft);
				
				//title_fft ... contains the whole transformed tile. 
					//Subnote 1: end result ... whole tile is over all feature maps transformed.

				for(int j=0; j<fft->tile_size; j++){
					for(int k=0; k<fft->tile_size; k++){
						
						tile_fft->data[c][j][k] = temp_fft->data[0][j][k];

					}
				}
			}
			
			//deallocate allocated memory space
			delete temp_fft;
			delete tile2d;


			int numFeatureMaps =  tile_fft->size[0];
			int numRows = tile_fft->size[1];
			int numCols = tile_fft->size[2];

	
			// C_Tensor *m = new C_Tensor[Z->size[0]];
			// for(int CurrWeight= 0; CurrWeight < Z->size[0]; CurrWeight++){
			// 	m[CurrWeight].allocate(X->size[0], fft->tile_size, fft->tile_size);
			// }
			//#########
			//result m
			//#########
		

			for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){
				for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
					for (int row = 0; row < numRows; row++) {
						for (int col = 0; col < numCols; col++) {

							m[CurrWeight].data[featureMap][row][col] = tile_fft->data[featureMap][row][col]*U_fft[CurrWeight].data[featureMap][row][col];
						}
					}
				}
			}
			delete tile_fft;
			
				//############
				//sum together
				//############

				//Subnote 1: m_sum ... contains for each weight a summary.
				//Subnote 2: each feature map of m_sum is a summary for one feature map.

			//m_sum ... contains summed output layers.
				//Subnote 1: each feature map corresponds to one summed output layer
				//Subnote 2: one summed output layer is the sum over all output feature maps for one weight 
			//ifft_sum ... contains all feature maps of m_sum after inverted FFT.
			C_Tensor m_sum(Z->size[0], numRows, numCols);
			C_Tensor ifft_sum(Z->size[0], numRows, numCols);

			//loop ... compute the summary of all individual outputs.
				//Subnote 1: each output consists out of several feature maps.
				//Subnote 2: sum all feature maps for one output together.
				//Subnote 3: output m_sum contains for each weight one feature map
			for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){
				for (int row = 0; row < numRows; row++) {
					for (int col = 0; col < numCols; col++) {
						for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
							// Accumulate the sum for each location
							m_sum.data[CurrWeight][row][col] += m[CurrWeight].data[featureMap][row][col];
						}
					}
				}
			}
			
			//##############
			//IFFT
			//##############
				//Subnote 1: back transformation of 
			
			C_Tensor *ifft_2d = new C_Tensor();
			ifft_2d->allocate(1, fft->tile_size, fft->tile_size);
			
			C_Tensor *temp_ifft_2d = new C_Tensor();
			temp_ifft_2d->allocate(1, fft->tile_size, fft->tile_size);
			
			//loop ... transform m_sum (FFT) from frequency-domain into time-domain
				//Subnote 1: Inverted FFT is IFFT
			for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){
				//loop ... copies one feature map from m_sum into Tensor ifft_2d
				for (int row = 0; row < numRows; row++) {
					for (int col = 0; col < numCols; col++) {
						ifft_2d->data[0][row][col] = m_sum.data[CurrWeight][row][col];
					}
				}
				//IFFT ... on one feature map of m_sum 
				ifft2d(ifft_2d, temp_ifft_2d);
				//loop ... put transformed feature map in 3D tensor for transformed output.
				for (int row = 0; row < numRows; row++) {
					for (int col = 0; col < numCols; col++) {
						ifft_sum.data[CurrWeight][row][col] = temp_ifft_2d->data[0][row][col];
					}
				}
			}
			delete ifft_2d;
			delete temp_ifft_2d;
			
			//##############################
			//Compute effective output tile
			//#############################

			
			//C_Tensor * output_tile = new C_Tensor[1];
			//pad_amount_row ... computes of local 0-padding for tile. Considering row.
			//pad_amount_column ... same case as pad_amount_row. Considers column.
			pad_amount_row = - endRow + startRow + fft->tile_size;
			pad_amount_column = - endCol + startCol + fft->tile_size;
			
			//row_size ... row dimension of the currently considered part of current output ifft_sum
			//column_size ... column dimensions of the currently considered part of current output ifft_sum
				//Subnote 1: just dimensions of the currently considered part. No location data where this part in ifft_sum fm is located
				//Subnote 2: over all feature maps of ifft_sum the data gets extracted from the starting point of the found area.
			int row_size = ifft_sum.size[1] - pad_amount_row - overlap;
			int column_size = ifft_sum.size[2] - pad_amount_column  - overlap;

			// output_tile[0].allocate(1, row_size, column_size);

			// int r = 0, c = 0;
			
			
			//loop ... take the non-overlapped values out of the 
			// for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){
			// 	for (int row = 0; row < m_sum.size[1] && startRow_fm + row < numRowsZ; row++) {
			// 		for (int col = 0; col < wino->out_size && startCol_fm + col < numColsZ ; col++) {
						
			// }

			// output_row ... current start row in the output Z for the current tile
			// output_col ... current start col in the output Z for the current tile.

			int output_row = fft->tile_stride*tileRow;
			int output_col = fft->tile_stride*tileCol;
			
			for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){
				for(int i=0; (i<row_size); i++){
					for(int j=0; (j<column_size); j++){
						//computing Z.
						Z->data[CurrWeight][tileRow*(fft->tile_stride)+i][tileCol*(fft->tile_stride)+j] = ifft_sum.data[CurrWeight][i+overlap][j+overlap].real()+B->data[0][0][CurrWeight];
					}
				}
			}
			
			//delete [] tile_fft;
			//delete [] output_tile;
		}
	}
	delete [] m;
	delete tile;


	//#############################################
	//Add B to output Z (Winograd = WX, not WX + B)
	//#############################################

	//loop ... add bias B to the CONV.

}

//############################################################
//
//############################################################

/*--------------------------------------- Basic ------------------------------------------*/
/* Copy your basic function in here! */

void convBasic(Tensor * X, Tensor * W ,  Tensor * b, Tensor * Z)
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
