
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//��������������� ������� ��� ������������� CUDA ��� ������������� ������������ ��������.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//�������, ������� ������ ��������� GPU
__global__ void addKernel(int *c, const int *a, const int *b)
{
	//� �������� ���������� ������� ������� ������������ ������ ���� � �����
    int i = threadIdx.x;	//�������� ������ ���� � �����
    c[i] = a[i] + b[i];		//���������� �������� ��� ���������������� ������� ���������� ��������
}

int main()
{
	//������������� ��������: ��������� ������ ���������� �������� �������� � � b 
    const int arraySize = 5;							//���������� ����� ��������
    const int a[arraySize] = { 1, 2, 3, 4, 5 };			//���� ������� �
    const int b[arraySize] = { 10, 20, 30, 40, 50 };	//� b
    int c[arraySize] = { 0 };							//� �������������� ������

    // �������� �������� � ������������ ������� (�� GPU) - ���������� ��������������� �������
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	//����� ���������� ����������
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

	// 5.5 ����� ����������

    // cudaDeviceReset ������ ���� ������ ����� ������� �� ���������, ����� ����������� �������������� � �����������
	// ���������� ������ traces
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// ��������������� ������� ��� ������������� CUDA ��� ������������� ������������ ��������.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;
	// 0. ��������������� ����������

    // �������, � ����� GPU ��������. � ������-��� ������� ����� �������� �������� ������
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	//1. ���� �������� ����������� ���������� ������ �� ����������

    // ��������� ������� ��� ���� �������� (��� �������, ���� ��������)
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int)); //�������� size(� ����� ������ ������ �������� arraySize) ������ ������ �� ���������� (���) � ���������� ��������� �� ���������� ������ � dev_c
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	//�� �� ����� ��� ������� �
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	//�� �� ����� ��� ������� b
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	//2. ���� �������� ������ �� ����� ������ � ������ ����������

    // ����������� �������� �������� � ������� �� ������ ����� � ������ ���
	//cudaMemcpy() �������� size * sizeof(int) ������ �� ������, �� ������� ��������� � � ������, �� ������� ��������� dev_a, ����������� � ����� �� ����������
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);	
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	//�� �� ����� ��� ������� b
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	// 3. ���� ��������� ���� �� ���������� //4 ���������� ��������� ���� *����� �� �����, ������ ����� ��������� �� ������* 

    // ���� ��������� ���� �� ��� � ����� ������� ��� ������� ��������
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);	//����� ������������ ����� � ��������� __global__ ������� ����, ���
	//1 - ������ ����� � ������, size - ������ ������� ����� (� ����� ������ arraySize = 5)

    // �������� �� ������� ������ ��� ������� ����
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize ������� ���������� ������ ���� � ���������� ��� ������, ������������ �� ����� �������.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	// 5. ���� �������� ���������� �� ������ ���������� � ���� ������

    // ����������� ������� �� ������ GPU � ������ �����
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);	//�������� �������� - ����� ��� ����������� ����������� �� ���������� � ����
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
Error: //� ������ ����������� ������ �� ������ �� ������ ���� ��� ���������� ��� �� ��� ����� (� ���������� �� �����������)
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
