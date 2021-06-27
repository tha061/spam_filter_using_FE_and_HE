#include <chrono>
#include <iostream>
#include<vector>
#include <cmath>
// using namespace std;
using namespace std::chrono;

#include "examples.h"
using namespace std;
using namespace seal;

#define COLUMN_1  40 
#define COLUMN_2  20 
#define COLUMN_3 10
#define COLUMN_4 2

/**
**Encoding:
1. for the vector x, pack 2000 elements repeated 4 times to create a vector that contains 8000 elements.
2. for the matrix w1, pack the first four columns into a vector that contains 8000 elements.
3. repeat step 2 for the rest of the columns in w1.
As result, we will have 1 ctxt for x, 10 ctxts for w1.
**Computation:
1. use the ctxt of x to perform element-wise multiplication with one of the ctxt from w1. e.g., r[0]=x*w1[0]. This will produce a ctxt that contains the products in each of the 2000 blocks.
2. create 4 bit mask, each has 1 set in its corresponding 2000 blocks. e.g. bm1=1111...00000, bm2=00000...1111...000, etc
3. multiply c0=r[0]*bm1, c1=r[0]*bm2, ...; this will give you 4 ctxts.
4. run totalSum on c0, c1, c2, c3
5. create bit mask bm1'=100..., bm2'=0100..., bm3'=00100.., etc
6. c0*=bm1'; c1*=bm2'; c2*=bm3'; c3*=bm4'.
7. c0+=c1+c2+c3; discard c1,c2,c3.
8. repeat steps 1-7 for the rest of the columns. The only changes are the indexes. e.g., in step 7, we will reuse c0;
 * */

void spam_filter_plaintext()
{
    
    cout<<"\n=========================================================\n";
    cout<<"|| Reading plaintext input vector, weights               ||\n";
    cout<<"|| Encode, encrypt input vector and model weights         ||";
    cout<<"\n=========================================================\n";

    size_t input_vector_size = 4000;
    vector<vector<double>> input_email_vector = read_csv_file("/home/tham/SpamFilter_SEAL/model_weights/vec_4000/trec07_email_X_test_vector_length_4000_entire_Apr-06-2021.csv");
    // cout<<"test ok"<<endl;
            
    for (int i =0; i < input_email_vector.size(); i++)
    {

        for (int j=0; j<input_vector_size; j++)
        {
            input_email_vector[i][j] = int(input_email_vector[i][j]);
            // cout<<"test ok"<<endl;
            
        }

    }
    cout<<"input_email_vector.size() = "<<input_email_vector.size()<<endl;

    vector<vector<double>> y_spam_ham = read_csv_file("/home/tham/SpamFilter_SEAL/model_weights/vec_4000/trec07_email_y_test_vector_length_4000_entire_Apr-06-2021.csv");

    for (int i=0; i < y_spam_ham.size(); i++)
    {
        for (int j=0; j<1; j++)
        {
            y_spam_ham[i][j] = int64_t(y_spam_ham[i][j]);
        }
    }

    vector<int64_t> y_ground_truth(input_email_vector.size(), 0ULL);
    for (int i =0; i < input_email_vector.size(); i++)
    {
        y_ground_truth[i] = y_spam_ham[i][0];
        // cout<<"ground truth = "<<y_ground_truth[i]<<"\t";
    }
    cout << endl;

    vector<vector<double>> W1 = read_csv_file("/home/tham/SpamFilter_SEAL/model_weights/vec_4000/vec_4000_weight_1.csv");
       
    for (int i =0; i < W1.size(); i++)
    {

        for (int j=0; j<input_vector_size; j++)
        {
            W1[i][j] = int(W1[i][j]);
            
        }

    }
    cout<<"W1.size() = "<<W1.size()<<endl;

    vector<vector<double>> W2 = read_csv_file("/home/tham/SpamFilter_SEAL/model_weights/vec_4000/vec_4000_weight_2.csv");
       
    for (int i =0; i < W2.size(); i++)
    {

        for (int j=0; j<COLUMN_1; j++)
        {
            W2[i][j] = int(W2[i][j]);
            
        }

    }

    // for (int i =0; i < COLUMN_2; i++)
    // {
    //     for (int j=0; j<COLUMN_1; j++)
    //     {
    //         cout<<W2[i][j]<<"\t";
    //     }
    //     cout<<endl;
    //     cout<<endl;
    // }
    
    cout<<"W2.size()= "<<W2.size()<<endl;

    vector<vector<double>> W3 = read_csv_file("/home/tham/SpamFilter_SEAL/model_weights/vec_4000/vec_4000_weight_3.csv");
       
    for (int i =0; i < W3.size(); i++)
    {

        for (int j=0; j<COLUMN_2; j++)
        {
            W3[i][j] = int(W3[i][j]);
            
        }

    }
    cout<<"W3.size()= "<<W3.size()<<endl;
    vector<vector<double>> W4 = read_csv_file("/home/tham/SpamFilter_SEAL/model_weights/vec_4000/vec_4000_weight_4.csv");
       
    for (int i =0; i < W4.size(); i++)
    {

        for (int j=0; j<COLUMN_3; j++)
        {
            W4[i][j] = int(W4[i][j]);
            
        }

    }

    cout<<"W4.size()= "<<W4.size()<<endl;
    cout<<"Input vector size = "<<input_vector_size<<endl;
    int64_t slot_count = 8192;
    int64_t row_size = slot_count/2;
    
    vector<int64_t> vector_X(slot_count, 0ULL); 

    for (int i=0; i < input_vector_size; i++)
    {
        vector_X[i] = input_email_vector[0][i];
        // vector_X[i+input_vector_size] = input_email_vector[0][i];
    }

    // Get starting timepoint
    auto start = high_resolution_clock::now();

    // cout << "Input plaintext vector_X:" << endl;
    // print_matrix(vector_X, row_size);

    

    // Get ending timepoint
    auto stop = high_resolution_clock::now();
  
    // Get duration. Substart timepoints to 
    // get durarion. To cast it to proper unit
    // use duration cast method
    auto duration_encryption_encode_email = duration_cast<milliseconds>(stop - start);

    
    // Get starting timepoint
    auto start2 = high_resolution_clock::now();

    // encoding matrix W1
    // cout << "Encode and encrypt weight_1" << endl;
    // vector<int64_t> matrix_all_zeros(slot_count, 0ULL); 
    vector<vector<int64_t>> weight_1;
    vector<int64_t> vector_0(slot_count, 0ULL); 
    

    for (int i=0; i<COLUMN_1; i++)
    {
        for (int j = 0; j < slot_count; j ++)
        {
            vector_0[j] = 0ULL;
        }
        
        for (int j =0; j< input_vector_size; j ++)
        {
            vector_0[j] = W1[i][j];
            // vector_0[j+input_vector_size] = W1[2*i+1][j];
        }
        weight_1.push_back(vector_0);

    }
    
   
    vector<vector<int64_t>> weight_2;
    vector<int64_t> vector_1(slot_count, 0ULL); 
    

    for (int i=0; i < COLUMN_2; i++)
    {
        for (int j=0; j < COLUMN_1; j++)
        {
            vector_1[j] = W2[i][j];
        }
        weight_2.push_back(vector_1);
    }
    
   
    vector<vector<int64_t>> weight_3;
    vector<int64_t> vector_3(slot_count, 0ULL); 
    

    for (int i=0; i < COLUMN_3; i++)
    {
        for (int j=0; j < COLUMN_2; j++)
        {
            vector_3[j] = W3[i][j];
        }
        weight_3.push_back(vector_3);
    }
    
    
    vector<vector<int64_t>> weight_4;
    vector<int64_t> vector_4(slot_count, 0ULL); 
   

     for (int i=0; i < COLUMN_4; i++)
    {
        for (int j=0; j < COLUMN_3; j++)
        {
            vector_4[j] = W4[i][j];
        }
        weight_4.push_back(vector_4);
    }
    
    
   
    // Get ending timepoint
    auto stop2 = high_resolution_clock::now();
  
    // Get duration. Substart timepoints to 
    // get durarion. To cast it to proper unit
    // use duration cast method
    auto duration_encryption_encode_model_weight = duration_cast<milliseconds>(stop2 - start2);

    
    // cout<<"computation in plaintexts"<<endl;
    vector<int64_t> result(input_email_vector.size(), 0ULL);
    for (int index=0; index < input_email_vector.size(); index++)
    {
        
        cout<<"\nindex = "<<index<<endl;
        vector<int64_t> X(input_vector_size);
        for (int i=0; i<input_vector_size; i++)
        {
            X[i] = input_email_vector[index][i];
        }
        // cout<<"OK"<<endl;
        
        vector<int64_t> X_mul_W1(COLUMN_1, 0);

        for (int i=0; i<COLUMN_1; i++)
        {
            for (int j =0; j<input_vector_size; j++)
            {
                X_mul_W1[i] = X_mul_W1[i] + X[j]*W1[i][j];
            }
            
        }
        // cout<<"X_mul_W1_square:"<<endl;
        vector<int64_t> X_mul_W1_square(COLUMN_1);
        for (int i=0; i < COLUMN_1; i++)
        {
            X_mul_W1_square[i] = pow(X_mul_W1[i],2);
        }
        // cout<<endl;

        // for (int i=0; i < COLUMN_1; i++)
        // {
        //     cout<<X_mul_W1_square[i]<<"\t";
        // }
        // cout<<endl;

        vector<int64_t> X_mul_W1_squared_mul_W2(COLUMN_2, 0);
        for (int i=0; i<COLUMN_2; i++)
        {
            for (int j=0; j<COLUMN_1; j++)
            {
                X_mul_W1_squared_mul_W2[i] = X_mul_W1_squared_mul_W2[i] + X_mul_W1_square[j]*W2[i][j];
            //    cout<<"X_mul_W1_squared_mul_W2[i] = "<<i<<":"<<X_mul_W1_squared_mul_W2[i]<<endl;
            }
            // cout<<endl; 
        }
        // cout<<"X_mul_W1_squared_mul_W2:"<<endl;
        // for (int i=0; i < COLUMN_2; i++)
        // {
        //     cout<<X_mul_W1_squared_mul_W2[i]<<"\t";
        // }
        // cout<<endl;

        vector<int64_t> X_mul_W1_squared_mul_W2_mul_W3(COLUMN_3, 0);
        for (int i=0; i<COLUMN_3; i++)
        {
            for (int j=0; j<COLUMN_2; j++)
            {
                X_mul_W1_squared_mul_W2_mul_W3[i] = X_mul_W1_squared_mul_W2_mul_W3[i] + X_mul_W1_squared_mul_W2[j]*W3[i][j];
            } 
        }
        // cout<<"X_mul_W1_squared_mul_W2_mul_W3:"<<endl;
        // for (int i=0; i < COLUMN_3; i++)
        // {
        //     cout<<X_mul_W1_squared_mul_W2_mul_W3[i]<<"\t";
        // }
        // cout<<endl;

        vector<int64_t> X_mul_W1_squared_mul_W2_mul_W3_mul_W4(COLUMN_4, 0);
        for (int i=0; i<COLUMN_4; i++)
        {
            for (int j=0; j<COLUMN_3; j++)
            {
                X_mul_W1_squared_mul_W2_mul_W3_mul_W4[i] = X_mul_W1_squared_mul_W2_mul_W3_mul_W4[i] + X_mul_W1_squared_mul_W2_mul_W3[j]*W4[i][j];
            } 
        }
        // cout<<"X_mul_W1_squared_mul_W2_mul_W3_mul_W4:"<<endl;
        // for (int i=0; i < COLUMN_4; i++)
        // {
        //     cout<<X_mul_W1_squared_mul_W2_mul_W3_mul_W4[i]<<"\t";
        // }
        // cout<<endl;

        double sum_exp_t = exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[0]) + exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[1]);
        // cout<<"sum_exp = "<<sum_exp_t<<endl;
        double score_0_t = double(exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[0])/sum_exp_t);
        double score_1_t = double(exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[1])/sum_exp_t);

        // cout<<"softmax activation: score_0 = "<<score_0_t<<"\t"<<"score_1 ="<<score_1_t<<endl;

        
        double score_0_sigmoid_t;
        score_0_sigmoid_t = double(1/(1+exp(-X_mul_W1_squared_mul_W2_mul_W3_mul_W4[0])));
        double score_1_sigmoid_t;
        score_1_sigmoid_t = double(1/(1+exp(-X_mul_W1_squared_mul_W2_mul_W3_mul_W4[1])));

        cout<<"sigmoid activation: score_0 = "<<score_0_sigmoid_t<<"\t"<<"score_1 = "<<score_1_sigmoid_t<<endl;
        int64_t test = int64_t(score_0_sigmoid_t);
        if (test == 1)
        {
            result[index] = 0;
            
        }
        else 
        {
            result[index] = 1;
        }
    }
    int sum = 0;
    for (int i =0; i < input_email_vector.size(); i++)
    {
        // cout <<result[i]<<"\t"<<y_ground_truth[i]<<endl;
        if (result[i] == y_ground_truth[i]) sum++;
    }
    cout<<endl;
    cout<<"sum = "<<sum<<endl;
    double accuracy;
    accuracy = double(sum/double(input_email_vector.size()));
    cout<<"accuracy = "<<accuracy<<endl;
    
} 

