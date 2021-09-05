#include "Result.h"

Result::Result(double train_loss, double test_loss, size_t no_of_hidden_neurons, 
                double lr, size_t iters, const char* af1, const char* af2)
{
    this->train_loss = train_loss;
    this->test_loss = test_loss;
    this->no_of_hidden_neurons = no_of_hidden_neurons;
    this->lr = lr;
    this->iters = iters;
    this->af1 = af1;
    this->af2 = af2;
}

Result::Result(double test_loss)
{
    this->test_loss = test_loss;
    train_loss = -1;
    no_of_hidden_neurons = 0;
}

bool Result::operator < (Result r)
{
    if( test_loss < r.test_loss )
        return true;
    return false;
}

bool Result::operator > (Result r)
{
    if( test_loss > r.test_loss )
        return true;
    return false;
}

bool Result::operator >= (Result r)
{
    if( test_loss >= r.test_loss )
        return true;
    return false;
}

bool Result::operator <= (Result r)
{
    if( test_loss <= r.test_loss )
        return true;
    return false;
}

double Result::getTestLoss()
{
    return test_loss;
}

void Result::show()
{
    std::cout<<"Result: \n";
    std::cout<<std::setw(5)<<""<<"Train loss:  "<<train_loss<<std::endl;
    std::cout<<std::setw(5)<<""<<"Test loss:  "<<test_loss<<std::endl;
    std::cout<<std::setw(5)<<""<<"No of Iterations took to fit:  "<<iters<<std::endl;
    std::cout<<std::setw(5)<<""<<"No of hidden neurons:  "<<no_of_hidden_neurons<<std::endl;
    std::cout<<std::setw(5)<<""<<"Layer 1 activation function:  "<<af1<<std::endl;
    std::cout<<std::setw(5)<<""<<"Layer 2 activation function:  "<<af2<<std::endl;
}

const char* Result::show2()
{
    std::cout<<"Result: \n";
    std::cout<<std::setw(5)<<""<<"Train loss: "<<train_loss<<std::endl;
    std::cout<<std::setw(5)<<""<<"Test loss: "<<test_loss<<std::endl;
    std::cout<<std::setw(5)<<""<<"No of hidden neurons: "<<no_of_hidden_neurons<<std::endl;
    std::cout<<std::setw(5)<<""<<"Layer 1 activation function: "<<af1<<std::endl;
    std::cout<<std::setw(5)<<""<<"Layer 2 activation function: ";
    return af2;
}

std::ostream &operator<<(std::ostream &os, Result &r)
{
    os << r.show2();
    return os;
}



