#include "aphw3.h"



Dataset loadFuncDataset(const char* filename)
{
    std::ifstream file{ filename };
    std::vector<std::vector<double>> Data;
    char temp_char{','};
    double temp_double{};
    size_t counter{};            //number of columns.
    while(temp_char == ',')
    {
        file >> temp_double;
        file >> temp_char;
        counter++;
    }
    file.close();
    file.open(filename);
    char delimiter;
    std::vector<double> features(counter, 1);
    while (!file.eof()) 
    {
        for (size_t i = 0; i < counter; i++)
            if (i < counter - 1)
                file >> features[i] >> delimiter;
            else
                file >> features[counter - 1];
        Data.push_back(features);
    }
    Data.pop_back();
    Matrix m{Data};
	
    std::array<size_t, 2> size{m.getSize()};  //size[0] is number of rows.
    std::vector<std::vector<double>> targetData;
    targetData.push_back( Data[size[0]-1] ); //last row of matrix is target.
    Matrix target{targetData};
    Data.pop_back();                      //the rest of the matrix is input
    Matrix input{Data};
    Dataset d(input, target);
    return d;
}

std::vector<Result> testNeuralNets(Dataset& dataset, std::vector<size_t>& hidden_neurons, double lr, 
                                    size_t max_iters, const char* af1, const char* af2)
{
    std::vector<Result> R;
    for (size_t i = 0; i < hidden_neurons.size(); i++)
    {
        NeuralNet nn(dataset, hidden_neurons[i], af1, af2, lr, max_iters);
        Result r = nn.fit();
        R.push_back(r);
    }
    return R;
}

Result findBestNeuralNet(Dataset& dataset, std::vector<size_t>& hidden_neurons, double lr, 
                                    size_t max_iters, const char* af1, const char* af2)
{
    std::vector<Result> R = testNeuralNets(dataset, hidden_neurons, lr, max_iters, af1, af2);
    double test_loss{}, min{};
    size_t index;
    for (size_t i = 0; i < R.size(); i++)
    {
        test_loss = R[i].getTestLoss();
        if( min >= test_loss )
        {
            min = test_loss;
            index = i;
        }
    }
    return R[index];
}

void estimateFunction(const char* filename, size_t hidden_neurons_no)
{
    Dataset d{loadFuncDataset(filename)};
    NeuralNet nn(d, hidden_neurons_no);
    Result r{nn.fit()};
    std::vector<double> test_result{nn.testResult()};
    std::vector<std::vector<double>> test_target{d.getTestTargets().getMatData()};
    size_t test_sample = d.getNoOfTestSamples();

    std::cout<<std::setiosflags(std::ios::left)<<std::setw(8)<<"No";
    std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<"Target";
    std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<"Estimated"<<std::endl;
    std::cout<<"--------------------------------\n";
    for (size_t i = 0; i < test_sample; i++)
    {
        std::cout<<std::setiosflags(std::ios::left)<<std::setw(8)<<std::setprecision(3)<<i+1;
        std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<std::setprecision(3)<<test_target[0][i];
        std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<std::setprecision(3)<<test_result[i];
        std::cout<<std::endl;
    }
}

//mine.
//gets a dataset as input instead of filename.
void estimateFunction(Dataset d, size_t hidden_neurons_no)
{
    NeuralNet nn(d, hidden_neurons_no);
    Result r{nn.fit()};
    std::vector<double> test_result{nn.testResult()};
    std::vector<std::vector<double>> test_target{d.getTestTargets().getMatData()};
    size_t test_sample = d.getNoOfTestSamples();

    std::cout<<std::setiosflags(std::ios::left)<<std::setw(8)<<"No";
    std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<"Target";
    std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<"Estimated"<<std::endl;
    std::cout<<"--------------------------------\n";
    for (size_t i = 0; i < test_sample; i++)
    {
        std::cout<<std::setiosflags(std::ios::left)<<std::setw(8)<<std::setprecision(3)<<i+1;
        std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<std::setprecision(3)<<test_target[0][i];
        std::cout<<std::setiosflags(std::ios::left)<<std::setw(12)<<std::setprecision(3)<<test_result[i];
        std::cout<<std::endl;
    }
    
}




