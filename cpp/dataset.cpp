#include "dataset.h"

Dataset::Dataset(Matrix inputs, Matrix targets, double percentage)
{
    //inputs sample no. must be = target sample no.
    this->inputs = inputs;
    this-> targets = targets;
    this->percentage = percentage;
    size_t Percentage = static_cast<size_t> (percentage);
    std::array<size_t, 2> inputSize = inputs.getSize();
    size_t trainSize = (inputSize[1]*Percentage)/100;     //number of samples taken as train.
    std::vector<std::vector<double>> trainBuffer;
    std::vector<std::vector<double>> inputData{inputs.getMatData()};
    for (size_t i = 0; i < inputSize[0]; i++)   //creating train_inputs
    {
        std::vector<double> temp;
        for (size_t j = 0; j < trainSize; j++)
            temp.push_back(inputData[i][j]);
        trainBuffer.push_back(temp);  
    }
    Matrix mm1{trainBuffer};
    train_inputs = mm1;
    
    std::vector<std::vector<double>> testBuffer;    //creating test_inputs
    for (size_t i = 0; i < inputSize[0]; i++)
    {
        std::vector<double> temp;
        for (size_t j = trainSize; j < inputSize[1]; j++)
            temp.push_back(inputData[i][j]);
        testBuffer.push_back(temp);  
    }
    Matrix mm2{testBuffer};
    test_inputs = mm2;
    
    std::array<size_t, 2> targetSize = targets.getSize();  //creating train_targets
    trainSize = (targetSize[1]*Percentage)/100;
    std::vector<std::vector<double>> trainTargetBuffer;
    std::vector<std::vector<double>> targetData{targets.getMatData()};
    for (size_t i = 0; i < targetSize[0]; i++)
    {
        std::vector<double> temp;
        for (size_t j = 0; j < trainSize; j++)
            temp.push_back(targetData[i][j]);
        trainTargetBuffer.push_back(temp);  
    }
    Matrix mm3{trainTargetBuffer};
    train_targets = mm3;

    std::vector<std::vector<double>> testTargetBuffer;   //creating test_target
    for (size_t i = 0; i < targetSize[0]; i++)
    {
        std::vector<double> temp;
        for (size_t j = trainSize; j < targetSize[1]; j++)
            temp.push_back(targetData[i][j]);
        testTargetBuffer.push_back(temp);  
    }
    Matrix mm4{testTargetBuffer};
    test_targets = mm4;

    no_of_samples = inputData[0].size();  //number of columns of input matrix.
    input_dim = inputData.size();         //number of rows of input matrix.
    target_dim = targetData.size();       //number of rows of target matrix.
}

Dataset::Dataset(Matrix inputs, Matrix targets, Matrix train_inputs, 
                        Matrix train_targets, Matrix test_inputs, Matrix test_targets)
{
    this->inputs = inputs;
    this->targets = targets;
    this->train_inputs = train_inputs;
    this->train_targets = train_targets;
    this->test_inputs = test_inputs;
    this->test_targets = test_targets;
    std::array<size_t, 2> size{inputs.getSize()};
    no_of_samples = size[1];
    input_dim = size[0];
    std::array<size_t, 2> size2{targets.getSize()};
    target_dim = size2[0];
}

Dataset::Dataset()
{
    percentage = 0;
    no_of_samples = 0;
    input_dim = 0;
    target_dim = 0;
}

size_t Dataset::getNoOfSamples()
{
    return no_of_samples;
}

size_t Dataset::getNoOfTrainSamples()
{
    std::array<size_t, 2> trainSize{train_inputs.getSize()};
    return trainSize[1];
}

size_t Dataset::getNoOfTestSamples()
{
    std::array<size_t, 2> testSize{test_inputs.getSize()};
    return testSize[1];
}

size_t Dataset::getInputDim()
{
    return input_dim;
}

size_t Dataset::getTargetDim()
{
    return target_dim;
}

Matrix Dataset::getInputs()
{
    return inputs;
}

Matrix Dataset::getTargets()
{
    return targets;
}

Matrix Dataset::getTrainInputs()
{
    return train_inputs;
}

Matrix Dataset::getTrainTargets()
{
    return train_targets;
}

Matrix Dataset::getTestInputs()
{
    return test_inputs;
}

Matrix Dataset::getTestTargets()
{
    return test_targets;
}

void Dataset::shuffle()
{
    inputs = targets.randShuffle2(inputs);
    train_inputs.randShuffle();
    train_targets.randShuffle();
    test_inputs.randShuffle();
    test_targets.randShuffle();
}

void Dataset::show()
{
    size_t trainN = static_cast<int>(no_of_samples*percentage/100.0);
    std::cout<<"Dataset: \n";
    std::cout<<std::setw(5)<<""<<"No of samples: "<<no_of_samples<<std::endl;
    std::cout<<std::setw(5)<<""<<"Train samples: "<<trainN<<std::endl;
    std::cout<<std::setw(5)<<""<<"Test samples: "<<no_of_samples-trainN<<std::endl;
    std::cout<<std::setw(5)<<""<<"Input dimension: "<<input_dim<<std::endl;
    std::cout<<std::setw(5)<<""<<"Target dimension: "<<target_dim<<std::endl;
}

size_t Dataset::show1()
{
    size_t trainN = static_cast<int>(no_of_samples*percentage/100.0);
    std::cout<<"Dataset: \n";
    std::cout<<std::setw(5)<<""<<"No of samples: "<<no_of_samples<<std::endl;
    std::cout<<std::setw(5)<<""<<"Train samples: "<<trainN<<std::endl;
    std::cout<<std::setw(5)<<""<<"Test samples: "<<no_of_samples-trainN<<std::endl;
    std::cout<<std::setw(5)<<""<<"Input dimension: "<<input_dim<<std::endl;
    std::cout<<std::setw(5)<<""<<"Target dimension: ";
    return target_dim;
}

std::vector<Matrix> Dataset::operator [] (size_t j)       
{
    std::vector<Matrix> v;
    std::vector<std::vector<double>> temp;
    std::vector<std::vector<double>> matData;
    std::array<size_t, 2> size;
    size = inputs.getSize();
    matData = inputs.getMatData();
    for (size_t i = 0; i < size[0]; i++)
    {
        std::vector<double> temp2;
        temp2.push_back(matData[i][j]);
        temp.push_back(temp2);
    } 
    Matrix I{temp};
    v.push_back(I);

    std::vector<std::vector<double>> Temp;
    std::vector<std::vector<double>> MatData;
    std::array<size_t, 2> Size;
    Size = targets.getSize();
    MatData = targets.getMatData();
    for (size_t i = 0; i < Size[0]; i++)
    {
        std::vector<double> temp2;
        temp2.push_back(MatData[i][j]);
        Temp.push_back(temp2);
    }
    Matrix T{Temp};
    v.push_back(T);
    return v;
}

Dataset Dataset::operator+(const Dataset& dataset)
{
    std::vector<std::vector<double>> data1,data2,data;
    data1 = inputs.getMatData();
    data2 = dataset.inputs.getMatData();
    std::array<size_t, 2> isize1, isize2;
    isize1 = inputs.getSize();
    isize2 = dataset.inputs.getSize();
    //if( isize1[0] == isize2[0] )        //creating new input
    for (size_t i = 0; i < isize1[0]; i++)
    {
        std::vector<double> temp;
        for (size_t j = 0; j < isize1[1]; j++)
            temp.push_back(data1[i][j]);
        for (size_t j = 0; j < isize2[1]; j++)
            temp.push_back(data2[i][j]);
        data.push_back(temp);    
    }
    Matrix Input(data);

    std::vector<std::vector<double>> data11,data22,data0;   
    data11 = targets.getMatData();
    data22 = dataset.targets.getMatData();
    std::array<size_t, 2> tsize1, tsize2;
    tsize1 = targets.getSize();
    tsize2 = dataset.targets.getSize();
    //if( tsize1[0] == tsize2[0] )       //creating new target
    for (size_t i = 0; i < tsize1[0]; i++)
    {
        std::vector<double> temp;
        for (size_t j = 0; j < tsize1[1]; j++)
            temp.push_back(data11[i][j]);
        for (size_t j = 0; j < tsize2[1]; j++)
            temp.push_back(data22[i][j]);
        data0.push_back(temp);    
    }
    Matrix Target(data0);

    Matrix RInput{Input};
    Matrix RTarget = RInput.randShuffle2(Target);  //shuffles Input and Target the same order.
    std::vector<std::vector<double>> RInputData{RInput.getMatData()};
    std::vector<std::vector<double>> trainBuffer;
    size_t trainSize = static_cast<size_t> ( ( isize1[1] + isize2[1] ) * 0.7);
    for (size_t i = 0; i < isize1[0]; i++)   //creating train_inputs
    {
        std::vector<double> temp;
        for (size_t j = 0; j < trainSize; j++)
            temp.push_back(RInputData[i][j]);
        trainBuffer.push_back(temp);  
    }
    Matrix Train_Input{trainBuffer};

    std::vector<std::vector<double>> testBuffer;
    for (size_t i = 0; i < isize1[0]; i++)   //creating test_inputs
    {
        std::vector<double> temp;
        for (size_t j = trainSize; j < ( isize1[1] + isize2[1] ); j++)
            temp.push_back(RInputData[i][j]);
        testBuffer.push_back(temp);  
    }
    Matrix Test_Input{testBuffer};

    std::vector<std::vector<double>> RTargetData{RTarget.getMatData()};
    std::vector<std::vector<double>> trainBuffer2;
    size_t trainSize2 = static_cast<size_t> ( ( tsize1[1] + tsize2[1] ) * 0.7);
    for (size_t i = 0; i < tsize1[0]; i++)   //creating train_targets
    {
        std::vector<double> temp;
        for (size_t j = 0; j < trainSize2; j++)
            temp.push_back(RTargetData[i][j]);
        trainBuffer2.push_back(temp);  
    }
    Matrix Train_Target{trainBuffer2};

    std::vector<std::vector<double>> testBuffer2;
    for (size_t i = 0; i < tsize1[0]; i++)   //creating test_targets
    {
        std::vector<double> temp;
        for (size_t j = trainSize2; j < ( tsize1[1] + tsize2[1] ); j++)
            temp.push_back(RTargetData[i][j]);
        testBuffer2.push_back(temp);  
    }
    Matrix Test_Target{testBuffer2};
    return Dataset(Input, Target, Train_Input, Train_Target, Test_Input, Test_Target);
}

std::ostream &operator<<(std::ostream &os, Dataset &d)
{ 
    os << d.show1();
    return os;
}










