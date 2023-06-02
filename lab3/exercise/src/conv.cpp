#include "conv.h"
#include <cmath>

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
	int k_width = W_in->size[1];
	int upper_lim = k_width/2 + k_width%2;
	int extra = -1 + k_width%2;
	for(int i =-k_width/2; i < upper_lim; i++){
		for(int j =-k_width/2; j<upper_lim; j++){
			(*W_out)[0][i+k_width/2][j+k_width/2].real( 
				(*W_in)[c][-i + k_width/2 + extra][-j + k_width/2 + extra]);
		}
	}
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
			printf("Kernel Size %d not supported by FFT\n",k_size);
			return NULL;
	}
	return fft;
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
			printf("Kernel Size %d not supported by Winograd \n",k_size);
			return NULL;
	}
	return wino;
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
		    //delete[] Gt[i];
		}
		// Free the memory for the array of row pointers
		//delete[] Gt;
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
		    //delete[] Gt[i];
		}
		// Free the memory for the array of row pointers
		//delete[] Gt;
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
	//delete [] InterRes;
	
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
			
			//tile_untrans ... array of tensor. Contains all input feature map tiles.
				//Subnote 1: Until now happend: 
					//#1 ... defining used weights by choosing array of tensor W
					//#2 ... defining currently considered starting point (startRow, startCol) of current tile 
					//#3 ... defining area of the current tile. Based on (startRow, startCol) and tile-size
					//#4 ... copy all values in the area of the input into the tile. All over all feature maps of the input into corresponding feature maps of the tile.
				//Subnote 2: Transformation function winoTile wants Array of Tensors as input. 
				//Subnote 3: Tile only tensor in array. output_size = 1.
			int output_size = 1; 
			Tensor *tile_untrans = new Tensor[output_size];
			for(int i =0; i < output_size; i++){
				tile_untrans[i].allocate(X->size[0], wino->tile_size, wino->tile_size);
			}
			//put defined, copied, untransformed input tile "tile" into array of tensors tile_untrans
			//tile_trans ... contains transformed input tiles. Tiles from all input feature maps. All tiles come from the same coordinate location.
			tile_untrans = tile;
			
			Tensor * tile_trans = winoTile(tile_untrans, 1, wino, bt_rows, bt_col, 0);
			//delete [] tile_untrans;

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

			//m_trans ... array of Tensors. Contains the m_sum Tensor. Input for the transformation ATmA.
			Tensor * m_trans = new Tensor[1];
			for(int i =0; i < output_size; i++){
				//bt_rows ... dim(BTdB) = (bt_rows, bt_rows). Due to o., dim(m) = dim(BtdB)
					//Subnote 1: m_sum contains in each feature map the sum (over all fm input) for one weight (element W)
				m_trans[i].allocate(Z->size[0], bt_rows, bt_rows);
			}
			//Tensor m_sum is the only element in the Array of Tensors m_trans
			m_trans[0] = m_sum;


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
			Tensor * ATmA = winoTile(m_trans, 1, wino, at_rows, at_col, 1);
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
			//delete [] m_trans;
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
}


/*
 * Convolution using inputs and converted Weights
 * FFT_STRUCT 		*fft	: Struct containing tile size (N), overlap and stride
 * C_Tensor			*U_fft:	: Complex Transformed Weight Tensor
 * Tensor			*B		: Bias 
 * Tensor			*Z		: Output Tensor
 * int 			k_size		: Width and Height of weight kernel
 */
void convFFT(Tensor * X, C_Tensor * U_fft, Tensor * B, 
		Tensor * Z, int k_size)
{

}


/*--------------------------------------- Basic ------------------------------------------*/
/* Copy your basic function in here! */
void convBasic(Tensor * X, Tensor * W ,  Tensor * b, Tensor * Z)
{
}
