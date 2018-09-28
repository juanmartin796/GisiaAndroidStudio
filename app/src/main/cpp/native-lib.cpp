#include <jni.h>
#include <string>
#include <android/NeuralNetworks.h>
#include <float.h>
#include <android/log.h>
#include <sstream>
#include <iomanip>
#include <fcntl.h>
extern "C" JNIEXPORT jstring

JNICALL
Java_gisia_martin_com_perceptron_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT jfloat
JNICALL
Java_gisia_martin_com_perceptron_MainActivity_model(JNIEnv *env, jobject instance) {
    ANeuralNetworksModel* model = NULL;
    ANeuralNetworksModel_create(&model);

    uint32_t dims[1] = {3};
    // In our example, all our tensors are matrices of dimension [3][4].
    ANeuralNetworksOperandType entradas, pesos, sal, activacion;
    entradas.type= ANEURALNETWORKS_TENSOR_FLOAT32;
    entradas.scale = 0.f;
    entradas.zeroPoint = 0;
    entradas.dimensionCount = 1;
    entradas.dimensions = dims;

    pesos.type= ANEURALNETWORKS_TENSOR_FLOAT32;
    pesos.scale = 0.f;
    pesos.zeroPoint = 0;
    pesos.dimensionCount= 1;
    pesos.dimensions = dims;

    /*sal.type= ANEURALNETWORKS_FLOAT32;
    sal.scale = 0.f;
    sal.zeroPoint = 0;
    sal.dimensionCount = 0;
    sal.dimensions = nullptr;*/
    sal.type= ANEURALNETWORKS_TENSOR_FLOAT32;
    sal.scale = 0.f;
    sal.zeroPoint = 0;
    sal.dimensionCount = 1;
    sal.dimensions = dims;

    activacion.type = ANEURALNETWORKS_INT32;
    activacion.scale = 0.f;
    activacion.zeroPoint = 0;
    activacion.dimensionCount = 0;
    activacion.dimensions = nullptr;






    // Now we add the seven operands, in the same order defined in the diagram.
    ANeuralNetworksModel_addOperand(model, &entradas);  // operand 0
    ANeuralNetworksModel_addOperand(model, &pesos);  // operand 1
    ANeuralNetworksModel_addOperand(model, &activacion);  // operand 2
    ANeuralNetworksModel_addOperand(model, &sal); // operand 3

    /*uint32_t ent[3] = {1,1,1};
    const void *bufferEnt = ent;
    ANeuralNetworksModel_setOperandValue(model, 0, bufferEnt, 3);*/


    //uint32_t pes[3] = {2,1,1};
    float pes[3] = {-1.5,1.0,1.0};
    const void *bufferPesos = pes;
    ANeuralNetworksModel_setOperandValue(model, 1, &pes, sizeof(pes));

    // We set the values of the activation operands, in our example operands 2 and 5.
    int32_t noneValue = ANEURALNETWORKS_FUSED_NONE;
    ANeuralNetworksModel_setOperandValue(model, 2, &noneValue, sizeof(noneValue));


    // We have two operations in our example.
    // The first consumes operands 1, 0, 2, and produces operand 4.
    uint32_t addInputIndexes[3] = {0, 1,2};
    uint32_t addOutputIndexes[1] = {3};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_MUL, 3, addInputIndexes, 1, addOutputIndexes);

    // Our model has one input (0) and one output (6).
    uint32_t modelInputIndexes[1] = {0};
    uint32_t modelOutputIndexes[1] = {3};
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
    // Set the single input to our sample model. Since it is small, we won’t use a memory buffer.
    float myInput[3] = {1.0,1,1};
    ANeuralNetworksExecution_setInput(run1, 0, NULL, myInput, sizeof(myInput));
    // Set the output.
    float myOutput[1];
    ANeuralNetworksExecution_setOutput(run1, 0, NULL, myOutput, sizeof(myOutput));
    // Starts the work. The work proceeds asynchronously.
    ANeuralNetworksEvent* run1_end = NULL;
    ANeuralNetworksExecution_startCompute(run1, &run1_end);
    // For our example, we have no other work to do and will just wait for the completion.
    ANeuralNetworksEvent_wait(run1_end);
    ANeuralNetworksEvent_free(run1_end);
    ANeuralNetworksExecution_free(run1);


    float result = 0;
    for (int i = 0; i < sizeof(myOutput); ++i) {
        result = result + myOutput[i];
    }
    return result;
}
