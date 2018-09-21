#include <jni.h>
#include <string>
#include <android/NeuralNetworks.h>
#include <float.h>
extern "C" JNIEXPORT jstring

JNICALL
Java_gisia_martin_com_perceptron_MainActivity_stringFromJNI(
        JNIEnv *env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT void
JNICALL
Java_gisia_martin_com_perceptron_MainActivity_model(JNIEnv *env, jobject instance) {
    ANeuralNetworksModel* model = NULL;
    ANeuralNetworksModel_create(&model);//

    uint32_t dims[1] = {1};
    // In our example, all our tensors are matrices of dimension [3][4].
    ANeuralNetworksOperandType input1, input2, input3,sal;
    input1.type= ANEURALNETWORKS_FLOAT32;
    input1.scale = 0.f;
    input1.zeroPoint = 0;
    input1.dimensionCount = 1;
    input1.dimensions = dims;

    input2.type= ANEURALNETWORKS_FLOAT32;
    input2.scale = 0.f;
    input2.zeroPoint = 0;
    input2.dimensionCount = 1;
    input2.dimensions = dims;

    input3.type= ANEURALNETWORKS_INT32;
    input3.scale = 0.f;
    input3.zeroPoint = 0;
    input3.dimensionCount = 0;
    input3.dimensions = NULL;

    sal.type= ANEURALNETWORKS_FLOAT32;
    sal.scale = 0.f;
    sal.zeroPoint = 0;
    sal.dimensionCount = 1;
    sal.dimensions = dims;



    // Now we add the seven operands, in the same order defined in the diagram.
    ANeuralNetworksModel_addOperand(model, &input1);  // operand 0
    ANeuralNetworksModel_addOperand(model, &input2);  // operand 1
    ANeuralNetworksModel_addOperand(model, &input3);  // operand 1
    ANeuralNetworksModel_addOperand(model, &sal); // operand 2

    // We set the values of the activation operands, in our example operands 2 and 5.
    int32_t noneValue = ANEURALNETWORKS_FUSED_NONE;
    ANeuralNetworksModel_setOperandValue(model, 2, &noneValue, sizeof(noneValue));

    // We have two operations in our example.
    // The first consumes operands 1, 0, 2, and produces operand 4.
    uint32_t addInputIndexes[3] = {0, 1,2};
    uint32_t addOutputIndexes[1] = {3};
    ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_ADD, 3, addInputIndexes, 1, addOutputIndexes);

    // Our model has one input (0) and one output (6).
    uint32_t modelInputIndexes[2] = {0,1};
    uint32_t modelOutputIndexes[1] = {3};
    ANeuralNetworksModel_identifyInputsAndOutputs(model, 2, modelInputIndexes, 1 ,modelOutputIndexes);

    ANeuralNetworksModel_finish(model);




    // Compile the model.
    ANeuralNetworksCompilation* compilation;
    ANeuralNetworksCompilation_create(model, &compilation);

    // Ask to optimize for low power consumption.
    ANeuralNetworksCompilation_setPreference(compilation, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    ANeuralNetworksCompilation_finish(compilation);


    // Run the compiled model against a set of inputs.
    ANeuralNetworksExecution* run1 = NULL;
    ANeuralNetworksExecution_create(compilation, &run1);
    // Set the single input to our sample model. Since it is small, we won’t use a memory buffer.
    float myInput[3] = {1.0,1.0,1.0};
    ANeuralNetworksExecution_setInput(run1, 0, NULL, myInput, sizeof(myInput));
    ANeuralNetworksExecution_setInput(run1, 1, NULL, myInput, sizeof(myInput));
    //ANeuralNetworksExecution_setInput(run1, 2, NULL, myInput, sizeof(myInput));
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



}
