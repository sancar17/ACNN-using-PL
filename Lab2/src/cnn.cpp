#include "cnn.h"

namespace ml {

CNN::CNN(std::vector<CNN_layer_struct> in_layers)
{
	uint32_t insize = 0;
	layers = in_layers;
	// The tricky part is allocating the proper tensors
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct & lay = layers[i];
		// lay->X is the input lay->Z is the output
		switch(lay.type){
			case Layer_Type::ReLU: case Layer_Type::Softmax:
				lay.output_size[0] = layers[i - 1].output_size[0];
				lay.output_size[1] = layers[i - 1].output_size[1];
				lay.output_size[2] = layers[i - 1].output_size[2];
				if(lay.in_place)
					lay.Z = layers[i - 1].Z;
				else
					lay.Z = new Tensor(layers[i-1].output_size[0],lay.output_size[1],lay.output_size[2]);
				break;
			case Layer_Type::Pool:
				lay.Z = new Tensor(lay.output_size[0],lay.output_size[1],lay.output_size[2]);
				break;
			case Layer_Type::Conv:
				lay.Z = new Tensor(lay.output_size[0],lay.output_size[1],lay.output_size[2]);
				lay.W = new Tensor[lay.output_size[0]]();
				for(int i =0 ; i < lay.output_size[0]; i++){
					lay.W[i].allocate(lay.input_channels,lay.kernel_width,lay.kernel_width);
				}
				lay.B = new Tensor(1,1,lay.output_size[0]);
				break;
			case Layer_Type::Linear:
				insize = layers[i-1].output_size[0] * layers[i-1].output_size[1] * layers[i-1].output_size[2];
				lay.Z = new Tensor(1,1,lay.output_size[2]);
				lay.W = new Tensor(1,lay.output_size[2],insize);
				lay.B = new Tensor(1,1,lay.output_size[2]);
				break;
			default:
				throw std::runtime_error("Layer not implemented !\n");
		}
	}
}



CNN::~CNN()
{
	for(int i = 0; i < layers.size(); i++){
		CNN_layer_struct & lay = layers[i];
		// lay->X is the input lay->Z is the output
		switch(lay.type){
			case Layer_Type::ReLU:
				if(!(lay.in_place))
					delete lay.Z;
				break;
			case Layer_Type::Softmax:
			case Layer_Type::Pool:
				delete lay.Z;
				break;
			case Layer_Type::Conv:
				delete lay.Z;
				delete [] lay.W;
				delete lay.B;
				break;
			case Layer_Type::Linear:
				delete lay.Z;
				delete lay.W;
				delete lay.B;
				break;
			default:
				printf("Rogue unimplemented layer found during deallocation !\n");
		}
	}
}



/* Implement Inference here !*/
Tensor * CNN::inference(Tensor * input)
{
	Tensor * X = input;
	//printf("%s", "a \n");

	for(int i = 0; i < layers.size(); i++){
		//printf("%d", i);
		CNN_layer_struct & lay = layers[i];

		switch(lay.type){
			
			case Layer_Type::ReLU:
				switch(lay.in_place){
					case true:
						//printf("%s", "relu in place \n");
						ReLU(X, X);
						break;
					case false:
						//printf("%s", "relu not in place \n");
						ReLU(X, lay.Z);
						break;
				}
				break;

			case Layer_Type::Softmax:
				//printf("%s", "softmax \n");
				Softmax(X, lay.Z);
				break;

			case Layer_Type::Pool:
				//printf("%s", "pool \n");
				maxPool(X, lay.Z);
				break;

			case Layer_Type::Conv:
				if(lay.pad!=0){
				//printf("%s", "will perform padding\n");
				X = padTensor(X, lay.pad);
				//printf("%s", "performed\n");
				}
				
				//printf("%s", "conv \n");
				conv2d(X, lay.W, lay.B, lay.Z);
				break;

			case Layer_Type::Linear:
				//printf("%s", "linear \n");
				Linear(X, lay.W, lay.B, lay.Z);
				break;

			default:
				throw std::runtime_error("Layer not implemented !\n");
		}

		X = lay.Z;
	}

	//printf("%s", "return \n");

	return X;
}

}