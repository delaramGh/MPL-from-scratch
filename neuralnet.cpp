#include "NeuralNet.h"

NeuralNet::NeuralNet(Dataset dataset, size_t hidden_layer_neurons, const char* f1, const char* f2, 
                                                    double lr, size_t max_iters, double min_loss)
{
    this->dataset = dataset;
    this->hidden_layer_neurons = hidden_layer_neurons;
    af1 = f1;
    af2 = f2;
    this->lr = lr;
    this->max_iters = max_iters;
    this->min_loss = min_loss;
	//setting initial value for w1, w2, b1, b2.
    Matrix temp_w1(hidden_layer_neurons, dataset.getInputDim(), 1);
    w1 = temp_w1;
    Matrix temp_b1(hidden_layer_neurons, 1, 1);
    b1 = temp_b1;
    Matrix temp_w2(dataset.getTargetDim(), hidden_layer_neurons, 1);
    w2 = temp_w2;
    Matrix temp_b2(dataset.getTargetDim(), 1, 1);
    b2 = temp_b2;   
}

Matrix NeuralNet::forwardPropagate(Matrix& input)
{
    //assuming input is a single column matrix.
    n1 = ( w1 * input ) + b1;
    if( !strcmp(af1, "Sigmoid") )  //strcmp: false if equal!
        a1 = sigmoid(n1);
    if( !strcmp(af1, "Linear") )
        a1 = linear(n1);

    n2 = ( w2 * a1 ) + b2;
    if( !strcmp(af2, "Sigmoid") )
        a2 = sigmoid(n2);
    if( !strcmp(af2, "Linear") )
        a2 = linear(n2);
    return a2;
}

void NeuralNet::backPropagate(Matrix& input, Matrix& target)
{
    size_t f2_dim = dataset.getTargetDim();
    Matrix f2_dot;
    if( !strcmp(af2, "Sigmoid") )
        f2_dot = sigmoidDot(f2_dim, n2);
    if( !strcmp(af2, "Linear") )
        f2_dot = linearDot(f2_dim, n2);
    a2 = forwardPropagate(input);
    Matrix e = target - a2;
    s2 = f2_dot * e * -2;
    Matrix f1_dot;
    if( !strcmp(af1, "Sigmoid") )
        f1_dot = sigmoidDot(hidden_layer_neurons, n1);
    if( !strcmp(af1, "Linear") )
        f1_dot = linearDot(hidden_layer_neurons, n1);
    s1 = f1_dot * w2.T() * s2;
    
    //update
    w1 = w1 - ( ( s1 * input.T() ) * lr );
    w2 = w2 - ( ( s2 * a1.T() ) * lr );
    b1 = b1 - ( s1 * lr );
    b2 = b2 - ( s2 * lr );
}

double NeuralNet::trainLoss()
{
    size_t trainSample = dataset.getNoOfTrainSamples();
    Matrix mLoss(dataset.getTargetDim(), dataset.getTargetDim(), 0); //setting initial value.
    Matrix trainInput = dataset.getTrainInputs();
    Matrix trainTarget = dataset.getTrainTargets();
    for (size_t i = 0; i < trainSample; i++)
    {
        Matrix a = trainInput.col(i);
        Matrix A = forwardPropagate(a);
        Matrix t = trainTarget.col(i);
        Matrix e = t - A;
        mLoss = mLoss + ( e.T() * e );
    }
    std::vector<std::vector<double>> Loss{mLoss.getMatData()};
    double loss = Loss[0][0] / static_cast<double>(trainSample);
    return loss;
}

double NeuralNet::testLoss()
{
    size_t testSample = dataset.getNoOfTestSamples();
    Matrix mLoss(dataset.getTargetDim(), dataset.getTargetDim(), 0);
    Matrix testInput = dataset.getTestInputs();
    Matrix testTarget = dataset.getTestTargets();
    for (size_t i = 0; i < testSample; i++)
    {
        Matrix a = testInput.col(i);
        Matrix A = forwardPropagate(a);
        Matrix t = testTarget.col(i);
        Matrix e = t - A; 
        mLoss = mLoss + ( e.T() * e );
    }
    std::vector<std::vector<double>> Loss{mLoss.getMatData()};
    double loss = Loss[0][0] / static_cast<double>(testSample);
    return loss; 
}

Result NeuralNet::fit()
{
    //min_loss = 0.001;
    srand(time(NULL));
    int r;
    Matrix trainInput = dataset.getTrainInputs();
    Matrix trainTarget = dataset.getTrainTargets();
    size_t max_iters2 = max_iters;
    size_t no_of_train_sample = dataset.getNoOfTrainSamples();
    double train_loss2{1};
    while( train_loss2 > min_loss )
    {
        r = rand() % no_of_train_sample;
        Matrix a0 = trainInput.col(r);
        Matrix a = forwardPropagate(a0);
        Matrix t = trainTarget.col(r);
        backPropagate(a0, t);  //w and b are up to date.
        train_loss2 = trainLoss();
        max_iters--;
        if( !max_iters )
            break;
    }
    //std::cout<<"nn fit: min loss is: "<<min_loss<<"\n";
    size_t iterations = max_iters2 - max_iters;
    double train_loss = trainLoss();
    double test_loss = testLoss();
    Result result(train_loss, test_loss, hidden_layer_neurons, lr, iterations, af1, af2);
    return result;
}

std::vector<double> NeuralNet::testResult()
{
    std::vector<double> result;
    size_t no_of_test_sample = dataset.getNoOfTestSamples();
    Matrix testSample = dataset.getTestInputs();
    for (size_t i = 0; i < no_of_test_sample; i++)
    {
        Matrix p{testSample.col(i)};
        Matrix a{forwardPropagate(p)};
        std::vector<std::vector<double>> tempResult{a.getMatData()};
        result.push_back(tempResult[0][0]);
    }
    return result;
}

void NeuralNet::show()
{
    std::cout<<"Neural Network: \n";
    std::cout<<std::setw(5)<<""<<"No of shidden neurons: "<<hidden_layer_neurons<<std::endl;
    std::cout<<std::setw(5)<<""<<"Input dimension: "<<dataset.getInputDim()<<std::endl;
    std::cout<<std::setw(5)<<""<<"Output dimension: "<<dataset.getTargetDim()<<std::endl;
    std::cout<<std::setw(5)<<""<<"Layer 1 activation function: "<<af1<<std::endl;
    std::cout<<std::setw(5)<<""<<"Layer 2 activation function: "<<af2<<std::endl;
}

const char* NeuralNet::show2()
{
    std::cout<<"Neural Network: \n";
    std::cout<<std::setw(5)<<""<<"No of shidden neurons: "<<hidden_layer_neurons<<std::endl;
    std::cout<<std::setw(5)<<""<<"Input dimension: "<<dataset.getInputDim()<<std::endl;
    std::cout<<std::setw(5)<<""<<"Output dimension: "<<dataset.getTargetDim()<<std::endl;
    std::cout<<std::setw(5)<<""<<"Layer 1 activation function: "<<af1<<std::endl;
    std::cout<<std::setw(5)<<""<<"Layer 2 activation function: ";    
    return af2;
}

std::ostream &operator<<(std::ostream &os, NeuralNet &nn)
{
    os << nn.show2();
    return os;
}

double Sigmoid(double x)
{
    return 1 / (1+exp(-1*x));
}

Matrix sigmoid(Matrix m)
{
    std::vector<std::vector<double>> n, a;
    n = m.getMatData();
    for (size_t i = 0; i < n.size(); i++)
    {
        std::vector<double> temp;
        temp.push_back( Sigmoid(n[i][0]) );  //sigmoid
        a.push_back(temp);
    }
    Matrix A{a};
    return A;
}

Matrix linear(Matrix m)
{
    return m;
}

Matrix sigmoidDot(size_t number_of_neurons, Matrix n)
{
    Matrix f(number_of_neurons, number_of_neurons, false);
    std::vector<std::vector<double>> fData{f.getMatData()};
    std::vector<std::vector<double>> nData{n.getMatData()};

    for (size_t i = 0; i < number_of_neurons; i++)
        fData[i][i] = Sigmoid(n[i][0]) * ( 1 - Sigmoid(n[i][0]) );
    Matrix f2{fData};
    return f2;    
}

Matrix linearDot(size_t number_of_neurons, Matrix n)
{
    Matrix f(number_of_neurons, number_of_neurons, false);
    std::vector<std::vector<double>> fData{f.getMatData()};

    for (size_t i = 0; i < number_of_neurons; i++)
        fData[i][i] = 1;
    Matrix f2{fData};
    return f2;
}

void NeuralNet::setW1(Matrix& w)
{
    w1 = w;
}

void NeuralNet::setW2(Matrix& w)
{
    w2 = w;
}

void NeuralNet::setB1(Matrix& b)
{
    b1 = b;
}

void NeuralNet::setB2(Matrix& b)
{
    b2 = b;
}

Matrix NeuralNet::getW1()
{
    return w1;
}

Matrix NeuralNet::getW2()
{
    return w2;
}

Matrix NeuralNet::getB1()
{
    return b1;
}

Matrix NeuralNet::getB2()
{
    return b2;
}



