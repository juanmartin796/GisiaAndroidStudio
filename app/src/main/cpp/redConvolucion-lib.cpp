//
// Created by juanm on 20/10/2018.
//
#include <jni.h>
#include <string>
#include <android/NeuralNetworks.h>
#include <float.h>
#include <android/log.h>
#include <sstream>
#include <iomanip>
#include <fcntl.h>
#include <android/log.h>

#include <android/sharedmem.h>
#include <sys/mman.h>
#include <android/bitmap.h>
#include <sys/stat.h>

extern "C"
JNIEXPORT void JNICALL
Java_gisia_martin_com_perceptron_MainActivity_redConvolucion(JNIEnv *env, jobject instance) {
    
    ANeuralNetworksModel* model = NULL;
    ANeuralNetworksModel_create(&model);

    ANeuralNetworksOperandType inputConvolucion, filterConvolucion, biasCovolucion, paddingConvolucion,
            walkWidhtConvolucion, walkHeightConvolucion, activation, outputConvolucion,
            filterWidht, filterHeight, outputPooling,
            paddingPooling, walkWidhtPooling, walkHeightPooling;

    inputConvolucion.type= ANEURALNETWORKS_TENSOR_FLOAT32;
    inputConvolucion.scale = 0.f;
    inputConvolucion.zeroPoint = 0;
    inputConvolucion.dimensionCount = 4;
    uint32_t dimsInput[4] = {1,5,5,1}; //batches, height, width, depth_in
    inputConvolucion.dimensions = dimsInput;

    filterConvolucion.type= ANEURALNETWORKS_TENSOR_FLOAT32;
    filterConvolucion.scale = 0.f;
    filterConvolucion.zeroPoint = 0;
    filterConvolucion.dimensionCount = 4;
    uint32_t dimsFilter[4] = {1,2,2,1}; //depth_out, filter_height, filter_width, depth_in
    filterConvolucion.dimensions = dimsFilter;

    biasCovolucion.type= ANEURALNETWORKS_TENSOR_FLOAT32;
    biasCovolucion.scale = 0.f;
    biasCovolucion.zeroPoint = 0;
    biasCovolucion.dimensionCount = 1;
    uint32_t dimsBias[1] = {1}; //depth_out
    biasCovolucion.dimensions = dimsBias;

    paddingConvolucion.type = ANEURALNETWORKS_INT32;
    paddingConvolucion.scale = 0.f;
    paddingConvolucion.zeroPoint = 0;
    paddingConvolucion.dimensionCount = 0;
    paddingConvolucion.dimensions = nullptr;

    walkWidhtConvolucion.type = ANEURALNETWORKS_INT32;
    walkWidhtConvolucion.scale = 0.f;
    walkWidhtConvolucion.zeroPoint = 0;
    walkWidhtConvolucion.dimensionCount = 0;
    walkWidhtConvolucion.dimensions = nullptr;

    walkHeightConvolucion.type = ANEURALNETWORKS_INT32;
    walkHeightConvolucion.scale = 0.f;
    walkHeightConvolucion.zeroPoint = 0;
    walkHeightConvolucion.dimensionCount = 0;
    walkHeightConvolucion.dimensions = nullptr;

    activation.type = ANEURALNETWORKS_INT32;
    activation.scale = 0.f;
    activation.zeroPoint = 0;
    activation.dimensionCount = 0;
    activation.dimensions = nullptr;

    outputConvolucion.type= ANEURALNETWORKS_TENSOR_FLOAT32;
    outputConvolucion.scale = 0.f;
    outputConvolucion.zeroPoint = 0;
    outputConvolucion.dimensionCount = 4;
    uint32_t dimsOutputConvolution[4] = {1,4,4,1}; //batches, out_height, out_width, depth_out
    outputConvolucion.dimensions = dimsOutputConvolution;

    filterWidht.type = ANEURALNETWORKS_INT32;
    filterWidht.scale = 0.f;
    filterWidht.zeroPoint = 0;
    filterWidht.dimensionCount = 0;
    filterWidht.dimensions = nullptr;

    filterHeight.type = ANEURALNETWORKS_INT32;
    filterHeight.scale = 0.f;
    filterHeight.zeroPoint = 0;
    filterHeight.dimensionCount = 0;
    filterHeight.dimensions = nullptr;

    outputPooling.type= ANEURALNETWORKS_TENSOR_FLOAT32;
    outputPooling.scale = 0.f;
    outputPooling.zeroPoint = 0;
    outputPooling.dimensionCount = 4;
    uint32_t dimsOutputPooling[4] = {1,2,2,1}; //batches, out_height, out_width, depth_out
    outputPooling.dimensions = dimsOutputPooling;



    paddingPooling.type = ANEURALNETWORKS_INT32;
    paddingPooling.scale = 0.f;
    paddingPooling.zeroPoint = 0;
    paddingPooling.dimensionCount = 0;
    paddingPooling.dimensions = nullptr;

    walkWidhtPooling.type = ANEURALNETWORKS_INT32;
    walkWidhtPooling.scale = 0.f;
    walkWidhtPooling.zeroPoint = 0;
    walkWidhtPooling.dimensionCount = 0;
    walkWidhtPooling.dimensions = nullptr;

    walkHeightPooling.type = ANEURALNETWORKS_INT32;
    walkHeightPooling.scale = 0.f;
    walkHeightPooling.zeroPoint = 0;
    walkHeightPooling.dimensionCount = 0;
    walkHeightPooling.dimensions = nullptr;




    ANeuralNetworksModel_addOperand(model, &inputConvolucion);  // operand 0
    ANeuralNetworksModel_addOperand(model, &filterConvolucion);  // operand 1
    ANeuralNetworksModel_addOperand(model, &biasCovolucion);  // operand 2
    ANeuralNetworksModel_addOperand(model, &paddingConvolucion);  // operand 3
    ANeuralNetworksModel_addOperand(model, &walkWidhtConvolucion); // operand 4
    ANeuralNetworksModel_addOperand(model, &walkHeightConvolucion);  // operand 5
    ANeuralNetworksModel_addOperand(model, &activation); // operand 6
    ANeuralNetworksModel_addOperand(model, &outputConvolucion); // operand 7

    ANeuralNetworksModel_addOperand(model, &filterWidht); // operand 8
    ANeuralNetworksModel_addOperand(model, &filterHeight);  // operand 9
    ANeuralNetworksModel_addOperand(model, &outputPooling); // operand 10

    ANeuralNetworksModel_addOperand(model, &paddingPooling);  // operand 11
    ANeuralNetworksModel_addOperand(model, &walkWidhtPooling); // operand 12
    ANeuralNetworksModel_addOperand(model, &walkHeightPooling);  // operand 13



    //Filter values
    float filterValues[4]={0,1,1,0};
    ANeuralNetworksModel_setOperandValue(model, 1, &filterValues, sizeof(filterValues));
    //Bias values
    float biasValues[1] = {0};
    ANeuralNetworksModel_setOperandValue(model, 2, &biasValues, sizeof(biasValues));


    //Padding
    int32_t paddingValue = ANEURALNETWORKS_PADDING_VALID;
    ANeuralNetworksModel_setOperandValue(model, 3, &paddingValue, sizeof(paddingValue));
    //walkWidhtConvolucion
    int32_t walkWidhtValue = 1;
    ANeuralNetworksModel_setOperandValue(model, 4, &walkWidhtValue, sizeof(walkWidhtValue));
    //walkHeightConvolucion
    int32_t walkHeightValue = 1;
    ANeuralNetworksModel_setOperandValue(model, 5, &walkHeightValue, sizeof(walkHeightValue));
    //Funcion activacion
    int32_t noneValue = ANEURALNETWORKS_FUSED_NONE;
    ANeuralNetworksModel_setOperandValue(model, 6, &noneValue, sizeof(noneValue));


    int32_t filterWidhtValues[1]= {2};
    ANeuralNetworksModel_setOperandValue(model, 8, &filterWidhtValues, sizeof(filterWidhtValues));

    int32_t filterHeightValues[1] = {2};
    ANeuralNetworksModel_setOperandValue(model, 9, &filterHeightValues, sizeof(filterHeightValues));


    //Padding
    int32_t paddingValuePooling = ANEURALNETWORKS_PADDING_VALID;
    ANeuralNetworksModel_setOperandValue(model, 11, &paddingValuePooling, sizeof(paddingValuePooling));

    //walkWidht
    int32_t walkWidhtValuePooling[1] = {2};
    ANeuralNetworksModel_setOperandValue(model, 12, &walkWidhtValuePooling, sizeof(walkWidhtValuePooling));

    //walkHeight
    int32_t walkHeightValuePooling[1] = {2};
    ANeuralNetworksModel_setOperandValue(model, 13, &walkHeightValuePooling, sizeof(walkHeightValuePooling));


    //Operacion de convolucion
    uint32_t addInputIndexesConvolucion[7] = {0,1,2,3,4,5,6};
    uint32_t addOutputIndexesConvolucion[1] = {7};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_CONV_2D, 7, addInputIndexesConvolucion, 1, addOutputIndexesConvolucion);

    //Operacion de pooling
    uint32_t addInputIndexesPooling[7] = {7,11,12,13,8,9,6};
    uint32_t addOutputIndexesPooling[1] = {10};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MAX_POOL_2D, 7, addInputIndexesPooling, 1, addOutputIndexesPooling);





    // Our model has one inputConvolucion (0) and one outputConvolucion (6).
    uint32_t modelInputIndexes[1] = {0};
    uint32_t modelOutputIndexes[1] = {10};
    ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, modelInputIndexes, 1 ,modelOutputIndexes);


    ANeuralNetworksModel_finish(model);
    // Compile the model.
    ANeuralNetworksCompilation* compilation;
    ANeuralNetworksCompilation_create(model, &compilation);
    // Ask to optimize for low power consumption.
    ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_LOW_POWER);
    ANeuralNetworksCompilation_finish(compilation);
    // Run the compiled model against a set of inputs.
    ANeuralNetworksExecution* run1 = NULL;
    ANeuralNetworksExecution_create(compilation, &run1);

    ANeuralNetworksMemory* mem1 = NULL;
    //int fd = open("/sdcard/94cdfb1955d225502630fb0f110eaf4c.jpg", O_RDONLY);
    int fd = open("/sdcard/IMAGENESCUERO/e.bmp", O_RDONLY);
    struct stat sb;
    fstat(fd, &sb);
    off_t buffer_size_bytes_ = sb.st_size;
    ANeuralNetworksMemory_createFromFd(buffer_size_bytes_, PROT_READ, fd, 0, &mem1);

    float inputValues[1][5][5][1] = {23,34,32,23,43,43,23,12,12,13,2,3,4,5,2,34,5,345,34,5345,34,5,345,34,5};
    //ANeuralNetworksExecution_setInput(run1, 0, NULL, bitmapByte, sizeof(bitmapByte));
    //ANeuralNetworksExecution_setInput(run1, 0, NULL, jd, 100);
    ANeuralNetworksExecution_setInputFromMemory(run1, 0, NULL, mem1, 0, buffer_size_bytes_);

    // Set the outputConvolucion.
    float myOutput[1][2][2][1];
    ANeuralNetworksExecution_setOutput(run1, 0, NULL, myOutput, sizeof(myOutput));



    // Starts the work. The work proceeds asynchronously.
    ANeuralNetworksEvent* run1_end = NULL;
    ANeuralNetworksExecution_startCompute(run1, &run1_end);
    // For our example, we have no other work to do and will just wait for the completion.
    ANeuralNetworksEvent_wait(run1_end);
    ANeuralNetworksEvent_free(run1_end);
    ANeuralNetworksExecution_free(run1);
    __android_log_print(ANDROID_LOG_VERBOSE, "RedConvolucion", "Termino la convolucion", 1);
}
