#include <chrono>
#include <iostream>
#include<vector>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
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

void trec07_dataset_test()
{
    print_example_banner("Test spam filter on trec07 dataset");

    // TRACK_LIST time_track_list;
    // high_resolution_clock::time_point t1, t2;

    EncryptionParameters parms(scheme_type::bfv);

    size_t poly_modulus_degree = 32768;//16384;//32768; //16384;  //8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 60));

    SEALContext context(parms);
    print_parameters(context);
    cout << endl;

    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);


    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    BatchEncoder batch_encoder(context);
    size_t slot_count = batch_encoder.slot_count();
    size_t row_size = slot_count/2;
    std::vector<int64_t> result;
    bool print_cmd = 0;
    bool test_plain = 1;
    int test_index = 8;
    int iterations = 5;

    ofstream myfile;
    myfile.open ("native/spam_filter/SF_results/runtime_ceas08_vec_3000_5_cloudlab.csv");
    myfile << "Encrypt weights (ms), Encrypt input vector (ms), Prediction (ms)\n";

    // size_t slot_count = 4;
    cout << "Plaintext Batch size: "<<slot_count<<endl;
    cout << "Plaintext matrix row size: " << row_size << endl;

    cout<<"\n=========================================================\n";
    cout<<"|| Reading plaintext input vector, weights               ||\n";
    cout<<"|| Encode, encrypt input vector and model weights         ||";
    cout<<"\n=========================================================\n";

    // size_t input_vector_size = 3000;
    string input_file_name = "native/spam_filter/all_datasets/ceas08-1_dataset/vec_3000/ceas08_X_test_vector_length_3000_entire_17-04-2021.csv";
    string w1_file_name = "native/spam_filter/all_datasets/ceas08-1_dataset/vec_3000/vec_3000_weight_1.csv";
    string w2_file_name = "native/spam_filter/all_datasets/ceas08-1_dataset/vec_3000/vec_3000_weight_2.csv";
    string w3_file_name = "native/spam_filter/all_datasets/ceas08-1_dataset/vec_3000/vec_3000_weight_3.csv";
    string w4_file_name = "native/spam_filter/all_datasets/ceas08-1_dataset/vec_3000/vec_3000_weight_4.csv";
    vector<vector<double>> input_email_vector = read_csv_file(input_file_name);
    
    cout<<"input_email_vector.size() = "<<input_email_vector.size()<<endl;
    cout<<"input_email_vector.size() = "<<input_email_vector[0].size()<<endl;
    size_t input_vector_size = input_email_vector[0].size();
    for (int i =0; i < input_email_vector.size(); i++)
    {

        for (int j=0; j<input_vector_size; j++)
        {
            input_email_vector[i][j] = int64_t(input_email_vector[i][j]);
            
        }

    }

    // cout<<"input_email_vector.size() = "<<input_email_vector.size()<<endl;

    vector<vector<double>> W1 = read_csv_file(w1_file_name);
       
    for (int i =0; i < W1.size(); i++)
    {

        for (int j=0; j<input_vector_size; j++)
        {
            W1[i][j] = int64_t(W1[i][j]);
            
        }

    }
    cout<<"W1.size() = "<<W1.size()<<endl;

    vector<vector<double>> W2 = read_csv_file(w2_file_name);
       
    for (int i =0; i < W2.size(); i++)
    {

        for (int j=0; j<COLUMN_1; j++)
        {
            W2[i][j] = int64_t(W2[i][j]);
            
        }

    }

    cout<<"W2.size()= "<<W2.size()<<endl;

    vector<vector<double>> W3 = read_csv_file(w3_file_name);
       
    for (int i =0; i < W3.size(); i++)
    {

        for (int j=0; j<COLUMN_2; j++)
        {
            W3[i][j] = int64_t(W3[i][j]);
            
        }

    }
    cout<<"W3.size()= "<<W3.size()<<endl;
    vector<vector<double>> W4 = read_csv_file(w4_file_name);
       
    for (int i =0; i < W4.size(); i++)
    {

        for (int j=0; j<COLUMN_3; j++)
        {
            W4[i][j] = int64_t(W4[i][j]);
            
        }

    }

    cout<<"W4.size()= "<<W4.size()<<endl;
    cout<<"Input vector size = "<<input_vector_size<<endl;

    
    
    for (int iter=0; iter < iterations; iter++)
    {
        TRACK_LIST time_track_list;
        high_resolution_clock::time_point t1, t2;
        // Get starting timepoint
        auto start2 = high_resolution_clock::now();
        t1 = high_resolution_clock::now();

        // encoding matrix W1
        // cout << "Encode and encrypt weight_1" << endl;
        // vector<int64_t> matrix_all_zeros(slot_count, 0ULL); 
        vector<vector<int64_t>> weight_1;
        vector<int64_t> vector_0(slot_count, 0ULL); 
        
        for (int i=0; i<COLUMN_1/2; i++)
        {
            for (int j = 0; j < slot_count; j ++)
            {
                vector_0[j] = 0ULL;
            }
            
            for (int j =0; j< input_vector_size; j ++)
            {
                vector_0[j] = W1[2*i][j];
                vector_0[j+input_vector_size] = W1[2*i+1][j];
            }
            weight_1.push_back(vector_0);

        }
        
        // cout << "Input plaintext weight_1[0]" << endl;
        // print_matrix(weight_1[0], row_size);
        // cout << "Input plaintext weight_1[19]" << endl;
        // print_matrix(weight_1[19], row_size);
        
    
        
        vector<Plaintext> plain_matrixes(COLUMN_1);
        vector<Ciphertext> encrypted_weight_1(COLUMN_1);
        for (int i=0; i<COLUMN_1/2; i++)
        {
            // cout << "Encrypting weight_1[" << i << "]:" << endl;
            batch_encoder.encode(weight_1[i], plain_matrixes[i]);
            encryptor.encrypt(plain_matrixes[i], encrypted_weight_1[i]);
        }
        


        // cout << "Encode and encrypt weight_2" << endl;
        // vector<int64_t> matrix_all_zeros(slot_count, 0ULL); 
        vector<vector<int64_t>> weight_2;
        vector<int64_t> vector_1(slot_count, 0ULL); 
        // for (int i=0; i < COLUMN_2; i++)
        // {
        //     for (int j=0; j < COLUMN_1; j++)
        //     {
        //         vector_1[j] = 1ULL; //suposed to be different for each i
        //     }
        //     weight_2.push_back(vector_1);
        // }

        for (int i=0; i < COLUMN_2; i++)
        {
            for (int j=0; j < COLUMN_1; j++)
            {
                vector_1[j] = W2[i][j];
            }
            weight_2.push_back(vector_1);
        }
        
        // cout << "Input plaintext weight_2[0]" << endl;
        // print_matrix(weight_2[0], row_size);
        // cout << "Input plaintext weight_2[19]" << endl;
        // print_matrix(weight_2[19], row_size);
        
        
        vector<Plaintext> plain_matrixes2(COLUMN_2);
        vector<Ciphertext> encrypted_weight_2(COLUMN_2);
        for (int i=0; i<COLUMN_2; i++)
        {
            // cout << "Encrypting weight_1[" << i << "]:" << endl;
            batch_encoder.encode(weight_2[i], plain_matrixes2[i]);
            encryptor.encrypt(plain_matrixes2[i], encrypted_weight_2[i]);
        }


        // cout << "Encode weight_3 in clear" << endl;
        // vector<int64_t> matrix_all_zeros(slot_count, 0ULL); 
        vector<vector<int64_t>> weight_3;
        vector<int64_t> vector_3(slot_count, 0ULL); 
        // for (int i=0; i < COLUMN_3; i++)
        // {
        //     for (int j=0; j < COLUMN_2; j++)
        //     {
        //         vector_3[j] = 1ULL; //suposed to be different for each i
        //     }
        //     weight_3.push_back(vector_3);
        // }

        for (int i=0; i < COLUMN_3; i++)
        {
            for (int j=0; j < COLUMN_2; j++)
            {
                vector_3[j] = W3[i][j];
            }
            weight_3.push_back(vector_3);
        }
        
        // cout << "Input plaintext weight_3[0]" << endl;
        // print_matrix(weight_3[0], row_size);
        // cout << "Input plaintext weight_3[9]" << endl;
        // print_matrix(weight_3[9], row_size);

        vector<Plaintext> plain_matrixes3(COLUMN_3);
        // vector<Ciphertext> encrypted_weight_3(COLUMN_3);
        for (int i=0; i<COLUMN_3; i++)
        {
            // cout << "Encrypting weight_1[" << i << "]:" << endl;
            batch_encoder.encode(weight_3[i], plain_matrixes3[i]);
            // encryptor.encrypt(plain_matrixes3[i], encrypted_weight_3[i]);
        }

        // cout << "Encode weight_4 in clear" << endl;
        // vector<int64_t> matrix_all_zeros(slot_count, 0ULL); 
        vector<vector<int64_t>> weight_4;
        vector<int64_t> vector_4(slot_count, 0ULL); 
        // for (int i=0; i < COLUMN_4; i++)
        // {
        //     for (int j=0; j < COLUMN_3; j++)
        //     {
        //         vector_4[j] = 1ULL; //suposed to be different for each i
        //     }
        //     weight_4.push_back(vector_4);
        // }

        for (int i=0; i < COLUMN_4; i++)
        {
            for (int j=0; j < COLUMN_3; j++)
            {
                vector_4[j] = W4[i][j];
            }
            weight_4.push_back(vector_4);
        }
        
        
        // cout << "Input plaintext weight_4[0]" << endl;
        // print_matrix(weight_4[0], row_size);
        // cout << "Input plaintext weight_4[1]" << endl;
        // print_matrix(weight_4[1], row_size);

        vector<Plaintext> plain_matrixes4(COLUMN_4);
        // vector<Ciphertext> encrypted_weight_3(COLUMN_3);
        for (int i=0; i<COLUMN_4; i++)
        {
            // cout << "Encrypting weight_1[" << i << "]:" << endl;
            batch_encoder.encode(weight_4[i], plain_matrixes4[i]);
            // encryptor.encrypt(plain_matrixes3[i], encrypted_weight_3[i]);
        }

        // Get ending timepoint
        t2 = high_resolution_clock::now();
        trackTaskPerformance(time_track_list, "encrypting weights (ms)", t1, t2);
        auto stop2 = high_resolution_clock::now();
    
        // Get duration. Substart timepoints to 
        // get durarion. To cast it to proper unit
        // use duration cast method
        auto duration_encryption_encode_model_weight = duration_cast<milliseconds>(stop2 - start2);

        

    // for (int iter=0; iter < iterations; iter++)
    // {
        cout<<"iter = "<<iter<<endl;

        

        // Get starting timepoint
        auto start = high_resolution_clock::now();
        t1 = high_resolution_clock::now();

        vector<int64_t> vector_X(slot_count, 0); 

        for (int i=0; i < input_vector_size; i++)
        {
            vector_X[i] = input_email_vector[iter][i];
            vector_X[i+input_vector_size] = input_email_vector[iter][i];
        }

        // cout << "Input plaintext vector_X:" << endl;
        // print_matrix(vector_X, row_size);

        Plaintext plain_matrix;
        // cout << endl;
        // print_line(__LINE__);
        // cout << "Encode and encrypt vector_X" << endl;
        batch_encoder.encode(vector_X, plain_matrix);
        Ciphertext encrypted_vector_X;
        encryptor.encrypt(plain_matrix, encrypted_vector_X);

        // Get ending timepoint
        t2 = high_resolution_clock::now();
        trackTaskPerformance(time_track_list, "encrypting input vector (ms)", t1, t2);
        auto stop = high_resolution_clock::now();
    
        // Get duration. Substart timepoints to 
        // get durarion. To cast it to proper unit
        // use duration cast method
        auto duration_encryption_encode_email = duration_cast<milliseconds>(stop - start);

        // cout<<"\n==========================================================\n";
        // cout<<"||                       X*W1                            ||";
        // cout<<"\n==========================================================\n";

        // cout<< "\nStep 1 Multiply element-wise of encrypted_vector_X with encrypted_weight_1\n";
        // cout << "Multiply element-wise of encrypted_vector_X with encrypted_matrix_W0"<<endl;
        
        // Get starting timepoint

        auto start3 = high_resolution_clock::now(); 
        
        t1 = high_resolution_clock::now();   
        vector<Ciphertext> encrypted_result_X_mul_weight1(COLUMN_1);
        int i;
#pragma omp parallel for shared() private(i)
        for (int i=0; i<COLUMN_1/2; i++)
        {
            // cout<< "X multiply weight1["<<i<<"]"<<endl;
            evaluator.multiply(encrypted_vector_X, encrypted_weight_1[i], encrypted_result_X_mul_weight1[i]);
            evaluator.relinearize_inplace(encrypted_result_X_mul_weight1[i], relin_keys);
        }

        

        // cout<< "\nStep 2: Create bit-1 masking: bm1, bm2\n";
        // cout << "Create bit-1 masking: bm1, bm2"<<endl;
        vector<int64_t> bitmask_1(slot_count, 0ULL); // a vector of size 4 with 0s
        for (int i=0; i < input_vector_size; i++)
        {
            bitmask_1[i] = 1ULL;
        }

        Plaintext plain_bitmask_1;
        batch_encoder.encode(bitmask_1, plain_bitmask_1);
        
        

        vector<int64_t> bitmask_2(slot_count, 0ULL); // a vector of size 4 with 0s
        for (int i=0; i < input_vector_size; i++)
        {
            bitmask_2[i+input_vector_size] = 1ULL;
        }

        Plaintext plain_bitmask_2;
        batch_encoder.encode(bitmask_2, plain_bitmask_2);
        
        
        // cout<< "\nStep 3: multiply with bit_mask\n";

        vector<Ciphertext> masked_encrypted_X_mul_weight1_firstBlock(COLUMN_1/2);
        vector<Ciphertext> masked_encrypted_X_mul_weight1_secondBlock(COLUMN_1/2);
    #pragma omp parallel for shared() private(i)
        for (int i=0; i<COLUMN_1/2; i++)
        {
    #pragma omp sections nowait
           {
    #pragma omp section
            evaluator.multiply_plain(encrypted_result_X_mul_weight1[i], plain_bitmask_1, masked_encrypted_X_mul_weight1_firstBlock[i]);
    #pragma omp section
            evaluator.multiply_plain(encrypted_result_X_mul_weight1[i], plain_bitmask_2, masked_encrypted_X_mul_weight1_secondBlock[i]);
            // evaluator.relinearize_inplace(masked_encrypted_X_mul_weight1_firstBlock[i], relin_keys);
            // evaluator.relinearize_inplace(masked_encrypted_X_mul_weight1_secondBlock[i], relin_keys);
            }
        }
        // cout<< "\nStep 4: total sum\n";
    #pragma omp parallel for shared() private(i)
        for (int i=0; i<COLUMN_1/2; i++)
        {
    #pragma omp sections nowait
           {
    #pragma omp section
            evaluator.total_sum(masked_encrypted_X_mul_weight1_firstBlock[i],galois_keys, slot_count);
    #pragma omp section
            evaluator.total_sum(masked_encrypted_X_mul_weight1_secondBlock[i],galois_keys, slot_count);
            }
        }
        
        // cout<< "\nStep 5 Create bitmask vector\n";
        // cout << "Create bitmask again to keep only the one element for the total sum"<<endl;
        
        vector<vector<int64_t>> bit_mask_t;
        vector<int64_t> vector_bit_mask(slot_count, 0ULL);
        for (int i=0; i<COLUMN_1; i++)
        {
            for (int j=0; j<slot_count;j++)
            {
                vector_bit_mask[j] = 0ULL;
            }
            vector_bit_mask[i] = 1ULL;
            bit_mask_t.push_back(vector_bit_mask);
        }
        // cout << "test OK"<<endl;
        
    
        vector<Plaintext> plain_matrixes_bit_mask(COLUMN_1);
        // vector<Ciphertext> encrypted_bit_mask_vector(COLUMN_1);
        for (int i=0; i<COLUMN_1; i++)
        {
            batch_encoder.encode(bit_mask_t[i], plain_matrixes_bit_mask[i]);
        }

        // vector<int64_t> bit_mask_zeros(slot_count, 0ULL);
        // Plaintext plaintext;
        // Ciphertext encrypted_bit_mask_zeros;
        // batch_encoder.encode(bit_mask_zeros, plaintext);
        // encryptor.encrypt(plaintext, encrypted_bit_mask_zeros);


        

        // cout<< "\nStep 6 multiply with bitmask\n";
        // cout << "Multiply bitmask_3_t with masked_encrypted_result_X_W0_1"<<endl;
        // int j = 0;
    #pragma omp parallel for shared() private(i)
        for (int i=0; i<COLUMN_1/2; i++)
        {
    #pragma omp sections nowait
            {
    #pragma omp section
            evaluator.multiply_plain_inplace(masked_encrypted_X_mul_weight1_firstBlock[i], plain_matrixes_bit_mask[2*i]);
            // evaluator.relinearize_inplace(masked_encrypted_X_mul_weight1_firstBlock[j], relin_keys);
    #pragma omp section
            evaluator.multiply_plain_inplace(masked_encrypted_X_mul_weight1_secondBlock[i], plain_matrixes_bit_mask[2*i+1]);
            // evaluator.relinearize_inplace(masked_encrypted_X_mul_weight1_secondBlock[j], relin_keys);
        
            }
        }
        // cout<< "Step 7: Add these together"<<endl;

        evaluator.add_inplace(masked_encrypted_X_mul_weight1_firstBlock[0],masked_encrypted_X_mul_weight1_secondBlock[0]);
    #pragma omp parallel for shared() private(i)
        for (int i=1; i<COLUMN_1/2; i++)
        {
    #pragma omp sections nowait
            {
    #pragma omp section
            evaluator.add_inplace(masked_encrypted_X_mul_weight1_firstBlock[0], masked_encrypted_X_mul_weight1_firstBlock[i]);
    #pragma omp section
            evaluator.add_inplace(masked_encrypted_X_mul_weight1_firstBlock[0], masked_encrypted_X_mul_weight1_secondBlock[i]);
            }
        }
        // for (int i=1; i<COLUMN_1/2; i++)
        // {
        //     evaluator.add_inplace(masked_encrypted_X_mul_weight1_firstBlock[0], masked_encrypted_X_mul_weight1_secondBlock[i]);
        // }

        if (print_cmd == 1)
        {
            Plaintext result_final;
            decryptor.decrypt(masked_encrypted_X_mul_weight1_firstBlock[0], result_final);
            
            batch_encoder.decode(result_final, vector_X);
            
            print_matrix(vector_X, row_size);

            for(int i=0;i<COLUMN_1+1;i++)
            {
                cout<<vector_X[i]<<"\t";
            }
            cout<<endl;
        }

        // cout<<"\n==========================================================\n";
        // cout<<"||                (X*W1) squared                          ||";
        // cout<<"\n==========================================================\n";

        evaluator.square_inplace(masked_encrypted_X_mul_weight1_firstBlock[0]);
        evaluator.relinearize_inplace(masked_encrypted_X_mul_weight1_firstBlock[0], relin_keys);

        Ciphertext masked_encrypted_X_mul_weight1_squared = masked_encrypted_X_mul_weight1_firstBlock[0];
        
        if (print_cmd == 1)
        {

            Plaintext result_final_square;
            decryptor.decrypt(masked_encrypted_X_mul_weight1_squared, result_final_square);
            
            batch_encoder.decode(result_final_square, vector_X);
            
            print_matrix(vector_X, row_size);
        }


        // cout<<"\n==========================================================\n";
        // cout<<"||                ((X*W1)squared)*W2                     ||";
        // cout<<"\n==========================================================\n";

            
        vector<Ciphertext> encrypted_result_X_mul_weight1_sqr_mul_weight2(COLUMN_2);
    #pragma omp parallel for shared() private(i)
        for (int i=0; i < COLUMN_2; i++)
        {
            // cout<< "X multiply weight1["<<i<<"]"<<endl;
            evaluator.multiply(masked_encrypted_X_mul_weight1_squared, encrypted_weight_2[i], encrypted_result_X_mul_weight1_sqr_mul_weight2[i]);
            evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[i], relin_keys);
            // cout << " noise = " << decryptor.invariant_noise_budget(encrypted_result_X_mul_weight1_sqr_mul_weight2[i]) << " bits"<<endl;
        }

        

        // Plaintext mult_th;
        // decryptor.decrypt(encrypted_result_X_mul_weight1_sqr_mul_weight2[1], mult_th);
        // batch_encoder.decode(mult_th, vector_X);
        // print_matrix(vector_X, row_size);


        // cout<< "\nTotal sum\n";
    #pragma omp parallel for shared() private(i)
        for (int i=0; i<COLUMN_2; i++)
        {
            evaluator.total_sum(encrypted_result_X_mul_weight1_sqr_mul_weight2[i],galois_keys, slot_count);
            // evaluator.total_sum(masked_encrypted_X_mul_weight1_sqr_mul_weight2_secondBlock[i],galois_keys, slot_count);
        }

        Plaintext total_sum;
        decryptor.decrypt(encrypted_result_X_mul_weight1_sqr_mul_weight2[1], total_sum);
        
        batch_encoder.decode(total_sum, vector_X);
        
        // print_matrix(vector_X, row_size);
        
        // cout<< "\nCreate bitmask vector\n";
        // cout << "Create bitmask again to keep only the one element for the total sum"<<endl;
        
        vector<vector<int64_t>> bit_mask_t_w2;
        vector<int64_t> vector_bit_mask_w2(slot_count, 0ULL);
        for (int i=0; i<COLUMN_2; i++)
        {
            for (int j=0; j<slot_count;j++)
            {
                vector_bit_mask_w2[j] = 0ULL;
            }
            vector_bit_mask_w2[i] = 1ULL;
            bit_mask_t_w2.push_back(vector_bit_mask_w2);
        }
        // cout << "test OK"<<endl;

        // print_matrix(bit_mask_t_w2[3], row_size);
        
    
        vector<Plaintext> plain_matrixes_bit_mask_w2(COLUMN_2);
        // vector<Ciphertext> encrypted_bit_mask_vector_w2(COLUMN_2);
        for (int i=0; i<COLUMN_2; i++)
        {
            // cout << "Encrypting bit_mask_vector[" << i << "]:" << endl;
            batch_encoder.encode(bit_mask_t_w2[i], plain_matrixes_bit_mask_w2[i]);
            // encryptor.encrypt(plain_matrixes_bit_mask_w2[i], encrypted_bit_mask_vector_w2[i]);
        }

        
        // cout<< "\nMultiply with bitmask\n";
        // cout << "Multiply bitmask_3_t with masked_encrypted_result_X_W0_1"<<endl;
        // int j_w2 = 0;
    #pragma omp parallel for shared() private(i)
        for (int i=0; i<COLUMN_2; i++)
        {
        
            evaluator.multiply_plain_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[i], plain_matrixes_bit_mask_w2[i]);
            evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[i], relin_keys);
        
        }
        
        
        // cout<< "Add ciphertexts together"<<endl;

        // evaluator.add_inplace(masked_encrypted_X_mul_weight1_sqr_mul_weight2_firstBlock[0],masked_encrypted_X_mul_weight1_sqr_mul_weight2_secondBlock[0]);
        for (int i=1; i<COLUMN_2; i++)
        {
            evaluator.add_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[0], encrypted_result_X_mul_weight1_sqr_mul_weight2[i]);
        }

        if (print_cmd == 1)
        {

            Plaintext result_final_w2;
            decryptor.decrypt(encrypted_result_X_mul_weight1_sqr_mul_weight2[0], result_final_w2);
            
            batch_encoder.decode(result_final_w2, vector_X);
            
            print_matrix(vector_X, row_size);
        }

        // cout<<"\n==========================================================\n";
        // cout<<"||                ((X*W1)squared)*W2*W3                 ||";
        // cout<<"\n==========================================================\n";

        

        Ciphertext result_encrypted_X_mul_weight1_squared_mul_weight2 = encrypted_result_X_mul_weight1_sqr_mul_weight2[0];    
        vector<Ciphertext> encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3(COLUMN_3);
    #pragma omp parallel for shared() private(i)
        for (int i=0; i < COLUMN_3; i++)
        {
            // cout<< "X multiply weight1["<<i<<"]"<<endl;
            evaluator.multiply_plain(result_encrypted_X_mul_weight1_squared_mul_weight2, plain_matrixes3[i], encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i]);
            evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i], relin_keys);
            // cout << " noise = " << decryptor.invariant_noise_budget(encrypted_result_X_mul_weight1_sqr_mul_weight2[i]) << " bits"<<endl;
        }

        // cout<< "\nTotal sum\n";
    #pragma omp parallel for shared() private(i)
        for (int i=0; i<COLUMN_3; i++)
        {
            evaluator.total_sum(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i],galois_keys, slot_count);
            // evaluator.total_sum(masked_encrypted_X_mul_weight1_sqr_mul_weight2_secondBlock[i],galois_keys, slot_count);
        }

        // cout<< "\nCreate bitmask vector\n";
        // cout << "Create bitmask again to keep only the one element for the total sum"<<endl;
        
        vector<vector<int64_t>> bit_mask_t_w3;
        vector<int64_t> vector_bit_mask_w3(slot_count, 0ULL);
        for (int i=0; i<COLUMN_3; i++)
        {
            for (int j=0; j<slot_count;j++)
            {
                vector_bit_mask_w3[j] = 0ULL;
            }
            vector_bit_mask_w3[i] = 1ULL;
            bit_mask_t_w3.push_back(vector_bit_mask_w3);
        }
        // cout << "test OK"<<endl;

        // print_matrix(bit_mask_t_w3[3], row_size);

        vector<Plaintext> plain_matrixes_bit_mask_w3(COLUMN_3);
        // vector<Ciphertext> encrypted_bit_mask_vector_w2(COLUMN_2);
        for (int i=0; i<COLUMN_3; i++)
        {
            // cout << "Encrypting bit_mask_vector[" << i << "]:" << endl;
            batch_encoder.encode(bit_mask_t_w3[i], plain_matrixes_bit_mask_w3[i]);
            // encryptor.encrypt(plain_matrixes_bit_mask_w2[i], encrypted_bit_mask_vector_w2[i]);
        }

        
        // cout<< "\nMultiply with bitmask\n";
        // cout << "Multiply bitmask_3_t with masked_encrypted_result_X_W0_1"<<endl;
        // int j_w2 = 0;
    #pragma omp parallel for shared() private(i)
        for (int i=0; i<COLUMN_3; i++)
        {
        
            evaluator.multiply_plain_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i], plain_matrixes_bit_mask_w3[i]);
            evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i], relin_keys);
        
        }

        // cout<< "Add ciphertexts together"<<endl;

        // evaluator.add_inplace(masked_encrypted_X_mul_weight1_sqr_mul_weight2_firstBlock[0],masked_encrypted_X_mul_weight1_sqr_mul_weight2_secondBlock[0]);
    #pragma omp parallel for shared() private(i)
        for (int i=1; i<COLUMN_3; i++)
        {
            evaluator.add_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[0], encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i]);
        }

        if (print_cmd == 1)
        {
            Plaintext result_final_w2_w3;
            std::vector<int64_t> result;
            decryptor.decrypt(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[0], result_final_w2_w3);
            
            batch_encoder.decode(result_final_w2_w3, result);
            
            print_matrix(result, row_size);

            for (int i=0; i<COLUMN_3+1; i++)
            {
                cout<<result[i]<<"\t";
            }
            cout<<endl;
        }
        



        // cout<<"\n==========================================================\n";
        // cout<<"||                ((X*W1)squared)*W2*W3*W4                ||";
        // cout<<"\n==========================================================\n";


        Ciphertext result_encrypted_X_mul_weight1_squared_mul_weight2_mul_weight3 = encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[0];    
        vector<Ciphertext> encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4(COLUMN_4);
    #pragma omp parallel for shared() private(i)
        for (int i=0; i < COLUMN_4; i++)
        {
            // cout<< "X multiply weight1["<<i<<"]"<<endl;
            evaluator.multiply_plain(result_encrypted_X_mul_weight1_squared_mul_weight2_mul_weight3, plain_matrixes4[i], encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i]);
            evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i], relin_keys);
            // cout << " noise = " << decryptor.invariant_noise_budget(encrypted_result_X_mul_weight1_sqr_mul_weight2[i]) << " bits"<<endl;
        }

        // cout<< "\nTotal sum\n";
    #pragma omp parallel for shared() private(i)
        for (int i=0; i<COLUMN_4; i++)
        {
            evaluator.total_sum(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i],galois_keys, slot_count);
            // evaluator.total_sum(masked_encrypted_X_mul_weight1_sqr_mul_weight2_secondBlock[i],galois_keys, slot_count);
        }

        // cout<< "\nCreate bitmask vector\n";
        // cout << "Create bitmask again to keep only the one element for the total sum"<<endl;
        
        vector<vector<int64_t>> bit_mask_t_w4;
        vector<int64_t> vector_bit_mask_w4(slot_count, 0ULL);
        for (int i=0; i<COLUMN_4; i++)
        {
            for (int j=0; j<slot_count;j++)
            {
                vector_bit_mask_w4[j] = 0ULL;
            }
            vector_bit_mask_w4[i] = 1ULL;
            bit_mask_t_w4.push_back(vector_bit_mask_w4);
        }
        // cout << "test OK"<<endl;

        // print_matrix(bit_mask_t_w4[0], row_size);

        vector<Plaintext> plain_matrixes_bit_mask_w4(COLUMN_4);
        // vector<Ciphertext> encrypted_bit_mask_vector_w2(COLUMN_2);
        for (int i=0; i<COLUMN_4; i++)
        {
            // cout << "Encrypting bit_mask_vector[" << i << "]:" << endl;
            batch_encoder.encode(bit_mask_t_w4[i], plain_matrixes_bit_mask_w4[i]);
            // encryptor.encrypt(plain_matrixes_bit_mask_w2[i], encrypted_bit_mask_vector_w2[i]);
        }

        
        // cout<< "\nMultiply with bitmask\n";
        // cout << "Multiply bitmask_3_t with masked_encrypted_result_X_W0_1"<<endl;
        // int j_w2 = 0;
    #pragma omp parallel for shared() private(i)
        for (int i=0; i<COLUMN_4; i++)
        {
        
            evaluator.multiply_plain_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i], plain_matrixes_bit_mask_w4[i]);
            evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i], relin_keys);
        
        }

        // cout<< "Add ciphertexts together"<<endl;

        // evaluator.add_inplace(masked_encrypted_X_mul_weight1_sqr_mul_weight2_firstBlock[0],masked_encrypted_X_mul_weight1_sqr_mul_weight2_secondBlock[0]);
    #pragma omp parallel for shared() private(i)
        for (int i=1; i<COLUMN_4; i++)
        {
            evaluator.add_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[0], encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i]);
        }

        Plaintext result_final_w2_w3_w4_final;
            
        decryptor.decrypt(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[0], result_final_w2_w3_w4_final);
        
        batch_encoder.decode(result_final_w2_w3_w4_final, result);

        double score_0_sigmoid = double(1/(1+exp(-result[0])));
        double score_1_sigmoid = double(1/(1+exp(-result[1])));
        

        // Get ending timepoint
        t2 = high_resolution_clock::now();
        trackTaskPerformance(time_track_list, "prediction (ms)", t1, t2);
        auto stop3 = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(stop3 - start3);
    

        
        cout << "Time taken to encode and encrypt input email vector: "
            << duration_encryption_encode_email.count() << " miliseconds" << endl;

        cout << "Time taken to encode and encrypt model weight: "
            << duration_encryption_encode_model_weight.count() << " miliseconds" << endl;
            
        cout << "Time taken to predict/classifying: "
            << duration.count() << " seconds" << endl;

        cout<<"final decrypted computation:"<<endl;
        for (int i=0; i<COLUMN_4+1; i++)
        {
            cout<<result[i]<<"\t";
        }
        cout<<endl;

        // double score_0_sigmoid = double(1/(1+exp(-result[0])));
        // double score_1_sigmoid = double(1/(1+exp(-result[1])));

        cout<<"with sigmoid activation: score_0 = "<<score_0_sigmoid<<"\t"<<"score_1 = "<<score_1_sigmoid<<endl;

        for (auto itr = time_track_list.begin(); itr != time_track_list.end(); itr++)
        {
            string time_diff = itr->second;
            myfile << time_diff << ", " ;
        }
    
        myfile << "\n";


    }

    


    if (test_plain == 1)
    {
        cout<<"\nComputation in plaintexts"<<endl;

        for (int iter =0; iter < iterations; iter ++)
        {
            vector<int64_t> X(input_vector_size);
            for (int i=0; i<input_vector_size; i++)
            {
                X[i] = input_email_vector[iter][i];
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
            

            // cout<<"\nX_mul_W1_square:"<<endl;
            vector<int64_t> X_mul_W1_square(COLUMN_1);
            for (int i=0; i < COLUMN_1; i++)
            {
                X_mul_W1_square[i] = pow(X_mul_W1[i],2);
            }
            

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
           
            vector<int64_t> X_mul_W1_squared_mul_W2_mul_W3(COLUMN_3, 0);
            for (int i=0; i<COLUMN_3; i++)
            {
                for (int j=0; j<COLUMN_2; j++)
                {
                    X_mul_W1_squared_mul_W2_mul_W3[i] = X_mul_W1_squared_mul_W2_mul_W3[i] + X_mul_W1_squared_mul_W2[j]*W3[i][j];
                } 
            }
            

            vector<int64_t> X_mul_W1_squared_mul_W2_mul_W3_mul_W4(COLUMN_4, 0);
            for (int i=0; i<COLUMN_4; i++)
            {
                for (int j=0; j<COLUMN_3; j++)
                {
                    X_mul_W1_squared_mul_W2_mul_W3_mul_W4[i] = X_mul_W1_squared_mul_W2_mul_W3_mul_W4[i] + X_mul_W1_squared_mul_W2_mul_W3[j]*W4[i][j];
                } 
            }
            cout<<"\nX_mul_W1_squared_mul_W2_mul_W3_mul_W4:"<<endl;
            for (int i=0; i < COLUMN_4; i++)
            {
                cout<<X_mul_W1_squared_mul_W2_mul_W3_mul_W4[i]<<"\t";
            }
            cout<<endl;

            // double sum_exp_t = exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[0]) + exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[1]);
            // cout<<"sum_exp = "<<sum_exp_t<<endl;
            // double score_0_t = double(exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[0])/sum_exp_t);
            // double score_1_t = double(exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[1])/sum_exp_t);

            // cout<<"softmax activation: score_0 = "<<score_0_t<<"\t"<<"score_1 ="<<score_1_t<<endl;

            
            double score_0_sigmoid_t = double(1/(1+exp(-X_mul_W1_squared_mul_W2_mul_W3_mul_W4[0])));
            double score_1_sigmoid_t = double(1/(1+exp(-X_mul_W1_squared_mul_W2_mul_W3_mul_W4[1])));

            cout<<"sigmoid activation: score_0 = "<<score_0_sigmoid_t<<"\t"<<"score_1 = "<<score_1_sigmoid_t<<endl;
        }
        
    }
    
    myfile.close();



   

} 

