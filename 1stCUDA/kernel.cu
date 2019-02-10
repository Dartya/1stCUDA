
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//Вспомогательная функция для использования CUDA для параллельного суммирования векторов.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//функция, которую должен выполнять GPU
__global__ void addKernel(int *c, const int *a, const int *b)
{
	//в качестве переменной индекса массива используется индекс нити в блоке
    int i = threadIdx.x;	//получаем индекс нити в блоке
    c[i] = a[i] + b[i];		//производим операцию над соответствующими индексу элементами массивов
}

int main()
{
	//инициализация массивов: программа должна складывать элементы векторов а и b 
    const int arraySize = 5;							//определяем длину массивов
    const int a[arraySize] = { 1, 2, 3, 4, 5 };			//сами массивы а
    const int b[arraySize] = { 10, 20, 30, 40, 50 };	//и b
    int c[arraySize] = { 0 };							//и результирующий массив

    // Сложение векторов в параллельных потоках (на GPU) - выполнение вспомогательной функции
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	//вывод результата вычислений
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

	// 5.5 РЕСЕТ УСТРОЙСТВА

    // cudaDeviceReset должен быть вызван перед выходом из программы, чтобы инструменты профилирования и трассировки
	// отображали полные traces
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Вспомогательная функция для использования CUDA для параллельного суммирования векторов.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;
	// 0. ПРЕДВАРИТЕЛЬНАЯ ПОДГОТОВКА

    // Выбрать, с каким GPU работать. В мульти-ГПУ системе нужно изменить параметр метода
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

	//1. ХОСТ ВЫДЕЛЯЕТ НЕОБХОДИМОЕ КОЛИЧЕСТВО ПАМЯТИ НА УСТРОЙСТВЕ

    // Выделение буферов для трех массивов (два входных, один выходной)
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int)); //выделяет size(в нашем случае размер массивов arraySize) байтов памяти на устройстве (ГПУ) и возвращает указатель на выделенную память в dev_c
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	//то же самое для массива а
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	//то же самое для массива b
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	//2. ХОСТ КОПИРУЕТ ДАННЫЕ ИЗ СВОЕЙ ПАМЯТИ В ПАМЯТЬ УСТРОЙСТВА

    // Копирование исходных векторов с данными из памяти хоста в буферы ГПУ
	//cudaMemcpy() копирует size * sizeof(int) байтов из памяти, на которую указывает а в память, на которую указывает dev_a, направление с хоста на устройство
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);	
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	//то же самое для массива b
    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	// 3. ХОСТ ЗАПУСКАЕТ ЯДРО НА УСТРОЙСТВЕ //4 УСТРОЙСТВО ИСПОЛНЯЕТ ЯДРО *этого не видим, только можем проверить на ошибки* 

    // Хост запускает ядро на ГПУ с одним потоком для каждого элемента
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);	//вызов определенной ранее с атрибутом __global__ функции ядра, где
	//1 - размер сетки в блоках, size - размер каждого блока (в нашем случае arraySize = 5)

    // Проверка на наличие ошибок при запуске ядер
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize ожидает завершения работы ядра и возвращает все ошибки, обнаруженные во время запуска.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	// 5. ХОСТ КОПИРУЕТ РЕЗУЛЬТАТЫ ИЗ ПАМЯТИ УСТРОЙСТВА В СВОЮ ПАМЯТЬ

    // Копирование вектора из буфера GPU в память хоста
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);	//обращаем внимание - здесь уже направление копирования из устройства в хост
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
Error: //в случае обнаружения ошибок на каждом из этапов выше код отправляет нас на эту метку (с ассемблера не пользовался)
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
