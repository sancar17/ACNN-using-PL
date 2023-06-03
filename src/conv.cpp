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
	//cout<<"a"<<endl;
	int k_width = W_in->size[1];
	//cout<<"b"<<endl;
	int upper_lim = k_width/2 + k_width%2;
	//cout<<"c"<<endl;
	int extra = -1 + k_width%2;
	//cout<<"d"<<endl;
	for(int i =-k_width/2; i < upper_lim; i++){
		for(int j =-k_width/2; j<upper_lim; j++){
			//cout<<"e"<<endl;
			//cout<<"i: "<<i<<" j: "<<j<<endl;
			//cout<<"i+k_width/2: "<<i+k_width/2<<endl;
			//cout<<(*W_out)[0]<<endl;
			//cout<<(*W_out)[0][i+k_width/2]<<endl;
			//cout<<(*W_out)[0][i+k_width/2][j+k_width/2]<<endl;

			(*W_out)[0][i+k_width/2][j+k_width/2].real( 
				(*W_in)[c][-i + k_width/2 + extra][-j + k_width/2 + extra]);

			
		}
	}

	//cout<<"flip done"<<endl;
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
	W->size[0]; //number of channels in kernel
	W->size[1]; //square dimensions
	W->size[2]; //square dimensions
	
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
		transW[filters].allocate(currFilter->size[0], g_rows,  g_rows);
		InterRes[filters].allocate(currFilter->size[0], g_rows, currFilter->size[1]);

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
				for ( size_t j = 0; j < g_rows; j++){
					//transW.data[filters][i][j] = 0;
					for (size_t k = 0; k < InterRes[filters].size[2]; k++)
					{
						transW[filters].data[c][i][j] += InterRes[filters].data[c][i][k] * Gt[k][j];
					}
				}
			}
		}
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
	
	//printf("\nFinal Dim of tensor: %d %d %d\n", transW.size[0], transW.size[1], transW.size[2]);

	
	//printf("DONEEEEEEEEEEE");
	return transW;
}

//#########################################################
//winoTile ... B.T*d*B. Transformation input tiles.
//#########################################################



Tensor * winoTile(Tensor * W, int output_channels, WINOGRAD_STRUCT* wino, int row_matrix, int column_matrix, int select)
{
	
	// printf("winoWeights opened.\n");
	Tensor *currFilter = NULL; // TODO: check multiple filters handling
	Tensor *transW = new Tensor[output_channels];
	Tensor *InterRes = new Tensor[output_channels];
	
	// printf("Array of Tensors init.\n");
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

		//loop-1 ... go into an channel. Same as feature map.
		for (size_t c = 0; c < currFilter->size[0]; c++)
		{
	
			//loop-2 ... go into row of G. G-values*g_featuremap-values
			for (size_t i = 0; i < row_matrix; i++)
			{
				for ( size_t j = 0; j < currFilter->size[1]; j++){

					//transW.data[filters][i][j] = 0;
					//loop-4 ... columns of G. Go along all columns of a row i.
					for (size_t k = 0; k < row_matrix; k++)
					{
						if(select == 0)
							InterRes[filters].data[c][i][j] += wino->G[i][k] * currFilter->data[c][k][j];
						else
							InterRes[filters].data[c][i][j] += wino->At[i][k] * currFilter->data[c][k][j];

					}
				}
			}
		}  

		//Gt for each G, which is selected based on kernel size
		float** Gt = new float*[column_matrix]; //column_matrix, row_matrix
    	for (int i = 0; i < column_matrix; i++) 
		{
        	Gt[i] = new float[row_matrix];
    	}
		if(select == 0)
			transposeMatrix(wino->G, row_matrix, column_matrix, Gt);
		else
			transposeMatrix(wino->At, row_matrix, column_matrix, Gt);

	
		//(previous result) * Gt
		for (size_t c = 0; c < InterRes[filters].size[0]; c++)
		{

			for (size_t i = 0; i < InterRes[filters].size[1]; i++)
			{
				//Number of cols of Gt = g_rows of G
				for ( size_t j = 0; j < row_matrix; j++){
					//transW.data[filters][i][j] = 0;
					for (size_t k = 0; k < InterRes[filters].size[2]; k++)
					{
						transW[filters].data[c][i][j] += InterRes[filters].data[c][i][k] * Gt[k][j];
					}
				}
			}
		}
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
	
	//printf("\nFinal Dim of tensor: %d %d %d\n", transW.size[0], transW.size[1], transW.size[2]);

	
	//printf("DONEEEEEEEEEEE");
	return transW;
}

int check_decimal(float value)
{
    //result ... initialize output function. 
		//correct #tile_for_feature_map
	int result = 0;
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
	//Iniit and definition of variables
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
	int numTilesRows = check_decimal(numTilesRows_unchecked);
	int numTilesCols = check_decimal(numTilesCols_unchecked);
	
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
	
	//#############################################
	//choose weights to be used (current iteration)
	//#############################################
	
	
	//CurrWeight ... Z->size[0] x different weights in U_wino. For each weight-Tensor own iteration. 
	for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){
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
				
				//numFeatureMaps ... amount feature maps of transformed tiles
				//numRows ... amount rows transformed tiles
				//numCols ... amount coloumns transformed tiles.
				int numFeatureMaps =  tile_trans->size[0];
				int numRows = tile_trans->size[1];
				int numCols = tile_trans->size[2];
				
				//m ... result element-wise multiplication. Transformed m
				Tensor m(numFeatureMaps, numRows, numCols);

				// Perform element-wise multiplication for each feature map
				for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
					for (int row = 0; row < numRows; row++) {
						for (int col = 0; col < numCols; col++) {
							// Perform element-wise multiplication between corresponding feature maps
							m.data[featureMap][row][col] = tile_trans->data[featureMap][row][col] * U_wino[CurrWeight].data[featureMap][row][col];
						}
					}
				}


				//############################################
				//Add output tile feature maps for one weight
				//############################################

				//m_sum ... Tensor with one feature map. All feature maps of Tensor m get element wise summed together. 
				Tensor m_sum(1, numRows, numCols);
				//loop ... compute the sum over all layers
				for (int row = 0; row < numRows; row++) {
					for (int col = 0; col < numCols; col++) {
						for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
							// Accumulate the sum for each location
							m_sum[0][row][col] += m.data[featureMap][row][col];
						}
					}
				}

				//############################################
				//Untransformed output tile ATmA
				//############################################

				//m_trans ... array of Tensors. Contains the m Tensor. Input for the transformation ATmA.
				Tensor * m_trans = new Tensor[1];
				for(int i =0; i < output_size; i++){
					//bt_rows ... dim(BTdB) = (bt_rows, bt_rows). Due to o., dim(m) = dim(BtdB)
					m_trans[i].allocate(X->size[0], bt_rows, bt_rows);
				}
				//Tensor m_sum is the only element in the Array of Tensors m_trans
				m_trans[0] = m_sum;

				//ATma ... untransformed output tiles. An Array of Tensor with one Tensor. The one Tensor contains only one feature map.
					//m_trans ... transformed output tiles
					//1 ... output_size. There is only one Tensor in the Array of Tensors.
					//wino ... contains At and kernel_size
					//at_rows ... rows of At
					//at_col ... columns of At
					//1 ... select. Select in wino matrix At to use. Wino contains several matrices - no confusion.
				Tensor * ATmA = winoTile(m_trans, 1, wino, at_rows, at_col, 1);

				// printf("Z ... dim Z: %d\n", Z->size[0]);
				// printf("Z ... dim Y: %d\n", Z->size[1]);
				// printf("Z ... dim Y: %d\n", Z->size[2]);
				//number of tensors in U_wino

				int numTensors = 0;

				// Iterate through the array until a null pointer is encountered
				// printf("U_wino last element ... amount feature maps: %d\n", U_wino[Z->size[0]-1].size[0]);

				//############################################
				//Defining output tensor Z
				//############################################

				//To-Do: add the "0-adding in the kernel dismissing"

				int startRow_fm = tileRow*wino->out_size;
				int startCol_fm = tileCol*wino->out_size;

				int numRowsZ = Z->size[1];
				int numColsZ = Z->size[2]; 

				//loop ... defining output tensor Z. 
				for (int row = 0; row < wino->out_size && startRow_fm + row < numRowsZ; row++) {
					for (int col = 0; col < wino->out_size && startCol_fm + col < numColsZ ; col++) {
						
						// Calculate the indices in the feature map
						int featureMapRow = startRow_fm + row;
						int featureMapCol = startCol_fm + col;

						// Assign the value from the tile to the corresponding position in the feature map
						//Z->data[CurrWeight][featureMapRow][featureMapCol] ... output tensor Z
						//ATmA[0].data[0][row][col] ... Array of Tensors ATma contains only one Tensor ATmA[0] with only one fm ATmA.data[0][y][x]

						Z->data[CurrWeight][featureMapRow][featureMapCol] = ATmA[0].data[0][row][col];
					}
				}		
			}	
		}
	}
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
	printf("B added.\n");

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
	
	cout<<"weights started"<<endl;
	//cout<<"w "<< W->size[0] <<" "<< W->size[1]<<" "<<W->size[2]<<endl;
	const FFT_STRUCT* fft = getFFT(W->size[1]);

	fft->overlap; fft->tile_size; fft->tile_stride;
	
	Tensor* currFilter = NULL;
	C_Tensor* temp = new C_Tensor[output_channels];
	C_Tensor* temp2 = new C_Tensor[output_channels];
	C_Tensor* U_fft = new C_Tensor[output_channels];
	Tensor* currFilter_padded = new Tensor[output_channels];

	
	
	for (size_t filters = 0; filters < output_channels; filters++)
	{
		//cout<<"output channels: "<< output_channels<<endl;
		//cout<<	"filters :" << filters << endl;
		currFilter  = &W[filters];
		//cout<<"1"<<endl;

		/*cout<<"original: "<<endl;
		for(int i = 0; i<W[filters].size[0]; i++){
			for(int j = 0; j<W[filters].size[1]; j++){
				for(int k = 0; k<W[filters].size[2]; k++){
					cout<<W[filters].data[i][j][k]<<" ";
				}
				cout<<endl;
			}
			cout<<endl;
		}*/

		/***************** PADDING OF WEIGHTS *********************/

		//pad the weights such that its total length ==  tile_size
		int pad_size =  fft->tile_size  - currFilter->size[1];
		//cout<<"2"<<endl;
		currFilter_padded[filters].allocate(currFilter->size[0], currFilter->size[1] + pad_size,\
			currFilter->size[2] + pad_size);
		//cout<<"3"<<endl;
	
		// Copy the content from W to currFilter_padded
        for (size_t i = 0; i < currFilter->size[0]; i++)
        {
            for (size_t j = 0; j < currFilter->size[1]; j++)
            {
                for (size_t k = 0; k < currFilter->size[2]; k++)
                {
					//cout<<"i: "<< i << " j: "<<j<<" k: "<<k<<endl;
					//cout<<"4"<<endl;
                    currFilter_padded[filters].data[i][j][k] = currFilter->data[i][j][k];
					//cout<<"original versions: "<<currFilter_padded[filters].data[i][j][k]<<endl;
					//cout<<"5"<<endl;
                }
            }
        }

		//cout<<"6"<<endl;
        
        // Pad the extra size with zeros
        /*for (size_t i = 0; i < currFilter->size[0]; i++)
        {
            for (size_t j = currFilter->size[1]; j < currFilter_padded[filters].size[1]; j++)
            {
                for (size_t k = currFilter->size[2]; k < currFilter_padded[filters].size[2]; k++)
                {
					///cout<<"7"<<endl;
					//cout<<"i: "<< i << " j: "<<j<<" k: "<<k<<endl;

                    currFilter_padded[filters].data[i][j][k] = 0.0;
					//cout<<"8"<<endl;
                }
            }
        }*/

		for (size_t i = 0; i < currFilter->size[0]; i++)
        {
            for (size_t j = 0; j < currFilter_padded[filters].size[1]; j++)
            {
                for (size_t k = 0; k < currFilter_padded[filters].size[2]; k++)
                {
					//cout<<"i: "<< i << " j: "<<j<<" k: "<<k<<endl;
					//cout<<"4"<<endl;
					//cout<<"after padding: "<<currFilter_padded[filters].data[i][j][k]<<endl;
					//cout<<"5"<<endl;
                }
            }
        }


		// Flip the matrix and store it in temp tensor
		U_fft[filters].allocate(currFilter->size[0], fft->tile_size, fft->tile_size);
		temp[filters].allocate(1, fft->tile_size, fft->tile_size);
		temp2[filters].allocate(1, fft->tile_size, fft->tile_size);

		for (int c=0; c<W->size[0]; c++){

			flip_Matrix( &currFilter_padded[filters], &temp[filters], c );
			// Perform 2D FFT and store result in U_fft
		}

		cout<<"after flipping: ";
		for (size_t i = 0; i < currFilter->size[0]; i++)
        {
            for (size_t j = 0; j < currFilter_padded[filters].size[1]; j++)
            {
                for (size_t k = 0; k < currFilter_padded[filters].size[2]; k++)
                {
					//cout<<"i: "<< i << " j: "<<j<<" k: "<<k<<endl;
					//cout<<"4"<<endl;
					//cout<<temp[filters].data[i][j][k]<<" ";
					//cout<<"5"<<endl;
                }
				//cout<<endl;
            }
			//cout<<endl;
        }

		cout<<"flipped"<<endl;
		fft2d( &temp[filters], &temp2[filters]);
		cout<<"fft done"<<endl;

		cout<<"After fft: "<<endl;
		for (int p=0; p<W->size[0]; p++){
            for (size_t j = 0; j < fft->tile_size; j++)
            {
                for (size_t k = 0; k < fft->tile_size; k++)
                {
					//cout<<"i: "<< p << " j: "<<j<<" k: "<<k<<endl;
					
                    U_fft[filters].data[p][j][k] = temp2[filters].data[0][j][k];
					//cout<<U_fft[filters].data[p][j][k]<<" ";
                }

				//cout<<endl;
            }

			//cout<<endl;
        }
	}

		//cout<<"2d performed"<<endl;
	
	cout<<"weights ended"<<endl;
	//exit(0);
	delete [] temp;
	delete [] temp2;
	delete[] currFilter_padded;
	//exit(0);
    return U_fft;

}



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
	int pad_amount = 0; //start_tile + tile_size - end_tile
    int tile_size = fft->tile_size; // Size of tile including overlap (N)
	int overlap = fft->overlap; // Size of overlap = w.x - 1 (M-1)
	int tile_stride = fft->tile_stride; // Size of tile without overlap (N-(M-1))

	cout<<"tile size: "<<tile_size<<endl;
	cout<<"overlap: "<<overlap<<endl;
	cout<<"tile stride: "<<tile_stride<<endl;

	int num_input_channels = X->size[0];
    int input_height = X->size[1];
    int input_width = X->size[2];

	//cout<<"num input channels: "<<num_input_channels<<endl;
	//cout<<"input hxw: "<<input_height<<" "<<input_width<<endl;

    int num_output_channels = Z->size[0];
    int output_height = Z->size[1];
    int output_width = Z->size[2];

	//cout<<"num output channels: "<<num_output_channels<<endl;
	//cout<<"output hxw: "<<output_height<<" "<<output_width<<endl;

	//create input tiles:
	float numTilesRows_unchecked = ((static_cast<float>(X->size[1]) - static_cast<float>(tile_size)) / static_cast<float>(tile_stride)) +1;
    float numTilesCols_unchecked = ((static_cast<float>(X->size[2]) - static_cast<float>(tile_size)) / static_cast<float>(tile_stride)) +1;

	//numTilesRows ... total amount of tiles. #tile
	//improvement to numTilesRows_unchecked ... only partly considered tiles by the area of the input feature map gets fully considered now.
	int numTilesRows = check_decimal(numTilesRows_unchecked);
	int numTilesCols = check_decimal(numTilesCols_unchecked);

	//cout<<"num tile rows: "<<numTilesRows<<endl;
	//cout<<"num tile cols: "<<numTilesCols<<endl;

	C_Tensor *tile = new C_Tensor(X->size[0], tile_size, tile_size);

	//CurrWeight ... Z->size[0] x different weights in U_fft. For each weight-Tensor own iteration. 
	for(int CurrWeight = 0; CurrWeight < Z->size[0]; CurrWeight++){
		// Loop over the tiles
			//cout<<"cur weight"<<CurrWeight<<endl;

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
				int startRow = tileRow * fft->tile_stride;
				int endRow = startRow + fft->tile_size;
				int startCol = tileCol * fft->tile_stride;
				int endCol = startCol + fft->tile_size;
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
					
					if((endRow != (startRow + fft->tile_size))|| (endCol != (startCol + fft->tile_size))){
						pad_amount = - endRow + startRow + fft->tile_size;
						for (int row = 0; row < fft->tile_size; row++) {
							for (int col = 0; col < fft->tile_size; col++) {
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
				int input_size = X->size[0]; 
				C_Tensor *tile_fft = new C_Tensor[input_size];
				C_Tensor *temp_fft = new C_Tensor[1];
				C_Tensor *tile2d = new C_Tensor[1];

				tile2d[0].allocate(1, fft->tile_size, fft->tile_size);
				temp_fft[0].allocate(1, fft->tile_size, fft->tile_size);

				//cout<<"start fft on tiles:"<<endl;
				for(int c=0; c<X->size[0]; c++){
					
					tile_fft[c].allocate(X->size[0], fft->tile_size, fft->tile_size);
					
					
					for(int j=0; j<fft->tile_size; j++){
						for(int k=0; k<fft->tile_size; k++){
							tile2d[0].data[0][j][k] = tile[0].data[c][j][k];
				//cout<<"3"<<endl;
						}
					}

					//cout<<" 1"<<endl;

					fft2d(&tile2d[0], &temp_fft[0]);
					//cout<<" 2"<<endl;
					
					for(int i=0; i<input_size; i++){
						for(int j=0; j<fft->tile_size; j++){
							for(int k=0; k<fft->tile_size; k++){
								tile_fft[c].data[i][j][k] = temp_fft[0].data[0][j][k];
					//cout<<"3"<<endl;
							}
						}
					}
				}

				delete temp_fft;
				delete tile2d;
				

	
				//cout<<"performed fft on tiles"<<endl;

				//numFeatureMaps ... amount feature maps of transformed tiles
				//numRows ... amount rows transformed tiles
				//numCols ... amount coloumns transformed tiles.
				int numFeatureMaps =  tile_fft->size[0];
				int numRows = tile_fft->size[1];
				int numCols = tile_fft->size[2];
				
				//m ... result element-wise multiplication. Transformed m
				//C_Tensor m(numFeatureMaps, numRows, numCols);
				C_Tensor *m = new C_Tensor[1];
				m[0].allocate(numFeatureMaps, numRows, numCols);

				// Perform element-wise multiplication for each feature map
				for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
					for (int row = 0; row < numRows; row++) {
						for (int col = 0; col < numCols; col++) {
							// Perform element-wise multiplication between corresponding feature maps
							m->data[featureMap][row][col].real(tile_fft->data[featureMap][row][col].real() * U_fft[CurrWeight].data[featureMap][row][col].real());
						}
					}
				}

				//############################################
				//Add output tile feature maps for one weight
				//############################################

				//m_sum ... Tensor with one feature map. All feature maps of Tensor m get element wise summed together. 
				C_Tensor *m_sum = new C_Tensor[1];
				m_sum[0].allocate(1, numRows, numCols);
				C_Tensor *ifft_sum = new C_Tensor[1];
				ifft_sum[0].allocate(1, numRows, numCols);
				//C_Tensor m_sum(1, numRows, numCols);
				//C_Tensor ifft_sum(1, numRows, numCols);
				//cout<<"sum parts started"<<endl;
				//loop ... compute the sum over all layers
				for (int row = 0; row < numRows; row++) {
					for (int col = 0; col < numCols; col++) {
						for (int featureMap = 0; featureMap < numFeatureMaps; featureMap++) {
							// Accumulate the sum for each location
							m_sum->data[0][row][col] += m->data[featureMap][row][col];
							//inverse fft
							
						}
					}
				}
				//cout<<"sum parts ended"<<endl;

				ifft2d(&m_sum[0], &ifft_sum[0]);
				cout<<"AAAA: "<<"ifft_sum data: "<<ifft_sum->data[0][0][0]<<endl;
				//cout<<"ifft performed"<<endl;

				C_Tensor * output_tile = new C_Tensor[1];
				int row_size = endRow - pad_amount - startRow - overlap + 1;
				int column_size = endCol - pad_amount - startCol - overlap + 1;
				cout<<" sizes: "<<row_size<<" "<<column_size<<endl;
				output_tile[0].allocate(1, row_size, column_size);

				//remove parts and add bias
				cout<<"remove parts started"<<endl;
				cout<<pad_amount<<endl;
				for(int rows = startRow+overlap; rows<=endRow - pad_amount; rows++){
					for(int cols = startCol + overlap; cols<=endCol - pad_amount; cols++){
						cout<<"rows cols: "<<rows<<" "<<cols<<" "<<endl;
						cout<<"rows cols of output tile: "<<rows-startRow-overlap<<" "<<cols-startCol-overlap<<" "<<endl;
						cout<<"ifft size: "<<ifft_sum->size[1]<<" "<< ifft_sum->size[2]<<endl;
						//cout<<"data ifft: "<<ifft_sum->data[0][rows][cols]<<endl;
						//cout<<"output tile: "<<output_tile[0].data[0][rows-startRow-overlap][cols-startCol-overlap]<<endl;
						if(rows==8 && cols==2)
							cout<<"ifft_sum data: "<<ifft_sum->data[0][rows][cols]<<endl;
						output_tile[0].data[0][rows-startRow-overlap][cols-startCol-overlap].real(ifft_sum->data[0][rows][cols].real());

					}
				}
				cout<<"remove parts ended"<<endl;

				for(int i=0; i<output_tile->size[1]; i++){
					for(int j=0; j>output_tile->size[2]; j++){
						Z->data[CurrWeight][tile_size*tileRow+i][tile_size*tileCol+j] = output_tile[0].data[0][i][j].real();
					}
				}
				
				//delete [] tile_fft;
				//delete [] output_tile;
				
			}
		}
	}

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

	delete tile;
	
	}


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
