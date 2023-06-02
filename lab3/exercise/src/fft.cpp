#include "fft.h"

void C_Tensor::allocate(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x)
{
	size[0] = dim_z;
	size[1] = dim_y;
	size[2] = dim_x;
	size_t dim_zy = size[1]*size[0];
	size_t dim_zyx = dim_zy * size[2]; 
	data = new C_FLOAT **[size[0]];
	C_FLOAT ** tmp_y = new C_FLOAT*[dim_zy];
	C_FLOAT * tmp_x = new C_FLOAT[dim_zyx];
	for(uint32_t i = 0; i < size[0]*size[1]; i++){
		tmp_y[i] = &(tmp_x[i * size[2]]);
	}
	for(uint32_t i = 0; i < size[0]; i++){
		data[i] = &(tmp_y[i * size[1]]);
	}
}

C_Tensor::C_Tensor(uint32_t dim_z, uint32_t dim_y, uint32_t dim_x)
{
	allocate(dim_z,dim_y,dim_x);
}

C_Tensor::~C_Tensor()
{
	if(data != NULL){
		if(data[0] != NULL){
			if(data[0][0] != NULL)
				delete [] data[0][0];
			delete [] data[0];
		}
		delete [] data;
	}
}

void fft(C_FLOAT * x_in, C_FLOAT * X_out, int N)
{
	//recursive implementation
	//X[k] = even[k] + twiddle factor * odd[k]

	if(N == 1) {
        // base of the recursion
        X_out[0] = x_in[0];
    } else {
        // Allocate temporary arrays
        C_FLOAT* x_even = new C_FLOAT[N/2];
        C_FLOAT* x_odd = new C_FLOAT[N/2];
        C_FLOAT* X_even = new C_FLOAT[N/2];
        C_FLOAT* X_odd = new C_FLOAT[N/2];

        // Split even and odd elements of the input
        for(int i = 0; i < N/2; i++) {
            x_even[i] = x_in[2*i];
            x_odd[i] = x_in[2*i + 1];
        }

        // Recursive FFT call for even and odd inputs
        fft(x_even, X_even, N/2);
        fft(x_odd, X_odd, N/2);

        //X[k] = even[k] + twiddle factor * odd[k]
        for(int k = 0; k < N/2; k++) {

            C_FLOAT twiddle = std::exp(C_FLOAT(0, -2.0 * M_PI / N * k));
			C_FLOAT t = twiddle * X_odd[k];
            X_out[k] = X_even[k] + t;  // For the first half
            X_out[k + N/2] = X_even[k] - t;  // For the second half
        }

        // Deallocate memory
        delete[] x_even;
        delete[] x_odd;
        delete[] X_even;
        delete[] X_odd;
    }
}

void ifft(C_FLOAT * x_in, C_FLOAT * X_out, int N)
{
	//Almost same as fft
	//Divide by N after the transformation
	//Twiddle factor has positive j
	//Use recursion again

	if(N == 1) {
        // base of the recursion
        X_out[0] = x_in[0];
    } else {
        // Allocate temporary arrays
        C_FLOAT* x_even = new C_FLOAT[N/2];
        C_FLOAT* x_odd = new C_FLOAT[N/2];
        C_FLOAT* X_even = new C_FLOAT[N/2];
        C_FLOAT* X_odd = new C_FLOAT[N/2];

        // Split even and odd elements of the input
        for(int i = 0; i < N/2; i++) {
            x_even[i] = x_in[2*i];
            x_odd[i] = x_in[2*i + 1];
        }

        // Recursive IFFT call for even and odd inputs
        fft(x_even, X_even, N/2);
        fft(x_odd, X_odd, N/2);

    	//X[k] = (even[k] + twiddle factor * odd[k])/N
        for(int k = 0; k < N/2; k++) {

            C_FLOAT twiddle = std::exp(C_FLOAT(0, 2.0 * M_PI / N * k));
			C_FLOAT t = twiddle * X_odd[k];
            X_out[k] = (X_even[k] + t) / float(N);  // For the first half
            X_out[k + N/2] = (X_even[k] - t) / float(N);  // For the second half
        }

        // Deallocate memory
        delete[] x_even;
        delete[] x_odd;
        delete[] X_even;
        delete[] X_odd;
    }
}

//For 2D FFT and IFFT we will use the FFT and IFFT functions from above
// We will apply FFT and IFFT column-wise and then row-wise

void fft2d(C_Tensor * x_in, C_Tensor * X_f)
{
	int dim_z = x_in->size[0];
    int dim_y = x_in->size[1];
    int dim_x = x_in->size[2];

    // Perform FFT row-wise
    for (int z = 0; z < dim_z; z++) {
        for (int y = 0; y < dim_y; y++) {
            fft(x_in->data[z][y], X_f->data[z][y], dim_x);
        }
    }

    // Perform FFT column-wise
    C_FLOAT *col_in = new C_FLOAT[dim_y];
    C_FLOAT *col_out = new C_FLOAT[dim_y];
    for (int z = 0; z < dim_z; z++) {
        for (int x = 0; x < dim_x; x++) {
            for (int y = 0; y < dim_y; y++) {
                col_in[y] = X_f->data[z][y][x];
            }
            fft(col_in, col_out, dim_y);
            for (int y = 0; y < dim_y; y++) {
                X_f->data[z][y][x] = col_out[y];
            }
        }
    }

    delete[] col_in;
    delete[] col_out;
}

//Same as FFT, just use IFFT instead
void ifft2d(C_Tensor * X_f , C_Tensor * x_out)
{
	int dim_z = X_f->size[0];
    int dim_y = X_f->size[1];
    int dim_x = X_f->size[2];

    // Perform IFFT row-wise
    for (int z = 0; z < dim_z; z++) {
        for (int y = 0; y < dim_y; y++) {
            ifft(X_f->data[z][y], x_out->data[z][y], dim_x);
        }
    }

    // Perform IFFT column-wise
    C_FLOAT *col_in = new C_FLOAT[dim_y];
    C_FLOAT *col_out = new C_FLOAT[dim_y];
    for (int z = 0; z < dim_z; z++) {
        for (int x = 0; x < dim_x; x++) {
            for (int y = 0; y < dim_y; y++) {
                col_in[y] = x_out->data[z][y][x];
            }
            ifft(col_in, col_out, dim_y);
            for (int y = 0; y < dim_y; y++) {
                x_out->data[z][y][x] = col_out[y];
            }
        }
    }

    delete[] col_in;
    delete[] col_out;
}


