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

void spam_filter_CKKS()
{
    print_example_banner("Example: Matrix multiplication in CKKS");

    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 16384; //16384;//32768; //16384;  //8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));
    // parms.set_plain_modulus(1024);

    double scale = pow(2.0, 40);

    SEALContext context(parms);
    print_parameters(context);
    cout << endl;

    

    KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);
    
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    std::vector<double> result;
    bool print_cmd = 1;
    bool test_plain = 1;
    int test_index = 1;

    // size_t slot_count = 4;
    cout << "Plaintext Batch size: "<<slot_count<<endl;
    // cout << "Plaintext matrix row size: " << row_size << endl;

    cout<<"\n=========================================================\n";
    cout<<"|| Reading plaintext input vector, weights               ||\n";
    cout<<"|| Encode, encrypt input vector and model weights         ||";
    cout<<"\n=========================================================\n";

    size_t input_vector_size = 2000;
    vector<vector<double>> input_email_vector = read_csv_file("/home/tham/SpamFilter_SEAL/trec07_email_X_test_vector_length_2000_entire_Apr-04-2021.csv");
       
    // for (int i =0; i < input_email_vector.size(); i++)
    // {

    //     for (int j=0; j<input_vector_size; j++)
    //     {
    //         input_email_vector[i][j] = double(input_email_vector[i][j]);
            
    //     }

    // }

    vector<vector<double>> W1 = read_csv_file("/home/tham/SpamFilter_SEAL/model_weights/weight_1.csv");
       
    // for (int i =0; i < W1.size(); i++)
    // {

    //     for (int j=0; j<input_vector_size; j++)
    //     {
    //         W1[i][j] = double(W1[i][j]);
            
    //     }

    // }
    cout<<"W1.size() = "<<W1.size()<<endl;

    vector<vector<double>> W2 = read_csv_file("/home/tham/SpamFilter_SEAL/model_weights/weight_2.csv");
       
    // for (int i =0; i < W2.size(); i++)
    // {

    //     for (int j=0; j<COLUMN_1; j++)
    //     {
    //         W2[i][j] = double(W2[i][j]);
            
    //     }

    // }

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

    vector<vector<double>> W3 = read_csv_file("/home/tham/SpamFilter_SEAL/model_weights/weight_3.csv");
       
    // for (int i =0; i < W3.size(); i++)
    // {

    //     for (int j=0; j<COLUMN_2; j++)
    //     {
    //         W3[i][j] = double(W3[i][j]);
            
    //     }

    // }
    cout<<"W3.size()= "<<W3.size()<<endl;
    vector<vector<double>> W4 = read_csv_file("/home/tham/SpamFilter_SEAL/model_weights/weight_4.csv");
       
    // for (int i =0; i < W4.size(); i++)
    // {

    //     for (int j=0; j<COLUMN_3; j++)
    //     {
    //         W4[i][j] = double(W4[i][j]);
            
    //     }

    // }

    cout<<"W4.size()= "<<W4.size()<<endl;
    cout<<"Input vector size = "<<input_vector_size<<endl;
    
    vector<double> vector_X(slot_count, 0); 

    // for (int i=0; i < slot_count; i++)
    // {
    //     vector_X[i] = 0;
    // }

    for (int i=0; i < input_vector_size; i++)
    {
        vector_X[i] = input_email_vector[test_index][i];
        // vector_X[i+input_vector_size] = input_email_vector[0][i];
    }

    print_vector(vector_X, 3, 7);

    // Get starting timepoint
    auto start = high_resolution_clock::now();

    cout << "Input plaintext vector_X:" << endl;
    print_vector(vector_X, 3,7);

    Plaintext plain_matrix;
    cout << endl;
    print_line(__LINE__);
    cout << "Encode and encrypt vector_X" << endl;
    encoder.encode(vector_X, scale, plain_matrix);
    Ciphertext encrypted_vector_X;
    encryptor.encrypt(plain_matrix, encrypted_vector_X);

    // Get ending timepoint
    auto stop = high_resolution_clock::now();
  
    // Get duration. Substart timepoints to 
    // get durarion. To cast it to proper unit
    // use duration cast method
    auto duration_encryption_encode_email = duration_cast<milliseconds>(stop - start);

    
    // Get starting timepoint
    auto start2 = high_resolution_clock::now();

    // encoding matrix W1
    cout << "Encode and encrypt weight_1" << endl;
    // vector<double> matrix_all_zeros(slot_count, 0ULL); 
    vector<vector<double>> weight_1;
    vector<double> vector_0(slot_count, 0ULL); 
    // for (int i=0; i < COLUMN_1; i++)
    // {
    //     for (int j=0; j < input_vector_size; j++)
    //     {
    //         vector_0[j] = 1ULL; //suposed to be different for each i
    //         vector_0[j+input_vector_size] = 1ULL; //suposed to be different for each i
    //     }
    //     weight_1.push_back(vector_0);
    // }

    for (int i=0; i<COLUMN_1; i++)
    {
        // for (int j = 0; j < slot_count; j ++)
        // {
        //     vector_0[j] = 0ULL;
        // }
        
        for (int j =0; j < input_vector_size; j ++)
        {
            vector_0[j] = W1[i][j];
            // vector_0[j] = 1ULL;
            // vector_0[j+input_vector_size] = W1[2*i+1][j];
        }
        weight_1.push_back(vector_0);

    }
    
    // cout << "Input plaintext weight_1[0]" << endl;
    // print_matrix(weight_1[0], row_size);
    // cout << "Input plaintext weight_1[19]" << endl;
    // print_matrix(weight_1[19], row_size);
    
   
    
    vector<Plaintext> plain_matrixes(COLUMN_1);
    vector<Ciphertext> encrypted_weight_1(COLUMN_1);
    for (int i=0; i<COLUMN_1; i++)
    {
        // cout << "Encrypting weight_1[" << i << "]:" << endl;
        encoder.encode(weight_1[i], scale, plain_matrixes[i]);
        encryptor.encrypt(plain_matrixes[i], encrypted_weight_1[i]);
    }
    


    cout << "Encode and encrypt weight_2" << endl;
    // vector<double> matrix_all_zeros(slot_count, 0ULL); 
    vector<vector<double>> weight_2;
    vector<double> vector_1(slot_count, 0ULL); 
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
        encoder.encode(weight_2[i], scale, plain_matrixes2[i]);
        encryptor.encrypt(plain_matrixes2[i], encrypted_weight_2[i]);
    }


    cout << "Encode weight_3 in clear" << endl;
    // vector<double> matrix_all_zeros(slot_count, 0ULL); 
    vector<vector<double>> weight_3;
    vector<double> vector_3(slot_count, 0ULL); 
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
        encoder.encode(weight_3[i], scale, plain_matrixes3[i]);
        // encryptor.encrypt(plain_matrixes3[i], encrypted_weight_3[i]);
    }

    cout << "Encode weight_4 in clear" << endl;
    // vector<double> matrix_all_zeros(slot_count, 0ULL); 
    vector<vector<double>> weight_4;
    vector<double> vector_4(slot_count, 0ULL); 
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
        encoder.encode(weight_4[i], scale, plain_matrixes4[i]);
        // encryptor.encrypt(plain_matrixes3[i], encrypted_weight_3[i]);
    }

    // Get ending timepoint
    auto stop2 = high_resolution_clock::now();
  
    // Get duration. Substart timepoints to 
    // get durarion. To cast it to proper unit
    // use duration cast method
    auto duration_encryption_encode_model_weight = duration_cast<milliseconds>(stop2 - start2);
   
    cout<<"\n==========================================================\n";
    cout<<"||                       X*W1                            ||";
    cout<<"\n==========================================================\n";

    // cout<< "\nStep 1 Multiply element-wise of encrypted_vector_X with encrypted_weight_1\n";
    // cout << "Multiply element-wise of encrypted_vector_X with encrypted_matrix_W0"<<endl;
    
    // Get starting timepoint
    auto start3 = high_resolution_clock::now();    
    vector<Ciphertext> encrypted_result_X_mul_weight1(COLUMN_1);
    for (int i=0; i<COLUMN_1; i++)
    {
        // cout<< "X multiply weight1["<<i<<"]"<<endl;
        evaluator.multiply(encrypted_vector_X, encrypted_weight_1[i], encrypted_result_X_mul_weight1[i]);
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1[i], relin_keys);
       
        evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1[i]);
         
    }

    // cout<<"test OK"<<endl;

    for (int i=0; i<COLUMN_1; i++)
    {
        evaluator.total_sum_ckks(encrypted_result_X_mul_weight1[i], galois_keys, slot_count);
        // evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1[i]);
        // evaluator.total_sum_ckks(masked_encrypted_X_mul_weight1_secondBlock[i],galois_keys, slot_count);
    }

    // cout<<"test OK"<<endl;

    // cout<< "\nStep 5 Create bitmask vector\n";
    // cout << "Create bitmask again to keep only the one element for the total sum"<<endl;
    
    vector<vector<double>> bit_mask_t;
    vector<double> vector_bit_mask(slot_count, 0);
    for (int i=0; i<COLUMN_1; i++)
    {
        for (int j=0; j<slot_count;j++)
        {
            vector_bit_mask[j] = 0;
        }
        vector_bit_mask[i] = 1;
        bit_mask_t.push_back(vector_bit_mask);
    }
    // cout << "test OK"<<endl;
    
    // cout<<"test OK"<<endl;

    vector<Plaintext> plain_matrixes_bit_mask(COLUMN_1);
    // vector<Ciphertext> encrypted_bit_mask_vector(COLUMN_1);
    for (int i=0; i<COLUMN_1; i++)
    {
        encoder.encode(bit_mask_t[i], scale, plain_matrixes_bit_mask[i]);
    }

    // vector<double> bit_mask_zeros(slot_count, 0ULL);
    // Plaintext plaintext;
    // Ciphertext encrypted_bit_mask_zeros;
    // batch_encoder.encode(bit_mask_zeros, plaintext);
    // encryptor.encrypt(plaintext, encrypted_bit_mask_zeros);


    // cout << "Parameters used by all three terms are different or not." << endl;
    // cout << "    + Modulus chain index for x3_encrypted: "
    //      << context.get_context_data(encrypted_result_X_mul_weight1[0].parms_id())->chain_index() << endl;
    // cout << "    + Modulus chain index for x1_encrypted: "
    //      << context.get_context_data(plain_matrixes_bit_mask[0].parms_id())->chain_index() << endl;

    // cout << "Normalize scales to 2^40." << endl;
    // encrypted_result_X_mul_weight1[0].scale() = pow(2.0, 40);
    // plain_matrixes_bit_mask[0].scale() = pow(2.0, 40);

    // cout << "Normalize encryption parameters to the lowest level." << endl;
    // parms_id_type last_parms_id = encrypted_result_X_mul_weight1[0].parms_id();
    // evaluator.mod_switch_to_inplace(encrypted_result_X_mul_weight1[0], last_parms_id);
    // evaluator.mod_switch_to_inplace(plain_matrixes_bit_mask[0], last_parms_id);



    // cout<< "\nStep 6 multiply with bitmask\n";
    // cout << "Multiply bitmask_3_t with masked_encrypted_result_X_W0_1"<<endl;
    // int j = 0;
    // for (int i=0; i < COLUMN_1; i++)
    // {
    //     cout<<context.get_context_data(encrypted_result_X_mul_weight1[i].parms_id())->chain_index();

    //     cout<<"\t"<<context.get_context_data(plain_matrixes_bit_mask[i].parms_id())->chain_index()<<endl;
    // }

    parms_id_type last_parms_id = encrypted_result_X_mul_weight1[0].parms_id();
    for (int i=0; i<COLUMN_1; i++)
    {
        encrypted_result_X_mul_weight1[i].scale() = pow(2.0, 40);
        plain_matrixes_bit_mask[i].scale() = pow(2.0, 40);
      
        evaluator.mod_switch_to_inplace(encrypted_result_X_mul_weight1[i], last_parms_id);
        evaluator.mod_switch_to_inplace(plain_matrixes_bit_mask[i], last_parms_id);
    }
    // cout<<"test OK"<<endl;

    //  for (int i=0; i < COLUMN_1; i++)
    // {
    //     cout<<context.get_context_data(encrypted_result_X_mul_weight1[i].parms_id())->chain_index();

    //     cout<<"\t"<<context.get_context_data(plain_matrixes_bit_mask[i].parms_id())->chain_index()<<endl;
    // }

    for (int i=0; i<COLUMN_1; i++)
    {
        evaluator.multiply_plain_inplace(encrypted_result_X_mul_weight1[i], plain_matrixes_bit_mask[i]);
        // cout<<"test OK 1"<<endl;
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1[i], relin_keys);
        // cout<<"test OK 2"<<endl;
        // evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1[i]);
        // cout<<"test OK 3"<<endl;

        // evaluator.relinearize_inplace(masked_encrypted_X_mul_weight1_firstBlock[j], relin_keys);
        // evaluator.multiply_plain_inplace(masked_encrypted_X_mul_weight1_secondBlock[i], plain_matrixes_bit_mask[2*i+1]);
        // evaluator.relinearize_inplace(masked_encrypted_X_mul_weight1_secondBlock[j], relin_keys);
       
    }

    // cout<<"test OK"<<endl;

    // cout<< "Step 7: Add these together"<<endl;

    // evaluator.add_inplace(masked_encrypted_X_mul_weight1_firstBlock[0],masked_encrypted_X_mul_weight1_secondBlock[0]);
    last_parms_id = encrypted_result_X_mul_weight1[0].parms_id();
    for (int i=0; i < COLUMN_1; i++)
    {
        // cout<<context.get_context_data(encrypted_result_X_mul_weight1[i].parms_id())->chain_index() << endl;
        encrypted_result_X_mul_weight1[i].scale() = pow(2.0, 40);
        evaluator.mod_switch_to_inplace(encrypted_result_X_mul_weight1[i], last_parms_id);
        // evaluator.mod_switch_to_inplace(plain_matrixes_bit_mask[i], last_parms_id);
    }
    // cout<<"test OK"<<endl;

    // cout<<context.get_context_data(encrypted_result_X_mul_weight1[0].parms_id())->chain_index()<<endl;

    for (int i=1; i<COLUMN_1; i++)
    {
        // cout<<context.get_context_data(encrypted_result_X_mul_weight1[0].parms_id())->chain_index();
        // cout<<context.get_context_data(encrypted_result_X_mul_weight1[i].parms_id())->chain_index()<<endl;

        evaluator.add_inplace(encrypted_result_X_mul_weight1[0], encrypted_result_X_mul_weight1[i]);
        // cout<<"test ok1"<<endl;
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1[0], relin_keys);
        // cout<<"test ok2"<<endl;
        // evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1[0]);
        // cout<<"test ok3"<<endl;

        // evaluator.add_inplace(masked_encrypted_X_mul_weight1_firstBlock[0], masked_encrypted_X_mul_weight1_secondBlock[i]);
    }
    // cout<<"test OK"<<endl;
    // for (int i=1; i<COLUMN_1/2; i++)
    // {
    //     evaluator.add_inplace(masked_encrypted_X_mul_weight1_firstBlock[0], masked_encrypted_X_mul_weight1_secondBlock[i]);
    // }

    if (print_cmd == 1)
    {
        Plaintext result_final;
        decryptor.decrypt(encrypted_result_X_mul_weight1[0], result_final);
        
        encoder.decode(result_final, vector_X);
        
        // print_matrix(vector_X, row_size);

        for(int i=0;i<COLUMN_1;i++)
        {
            cout<<vector_X[i]<<"\t";
        }
        cout<<endl;
    }

    cout<<"\n==========================================================\n";
    cout<<"||                (X*W1) squared                          ||";
    cout<<"\n==========================================================\n";
    Ciphertext encrypted_result_X_mul_weight1_squared; 
    evaluator.square(encrypted_result_X_mul_weight1[0], encrypted_result_X_mul_weight1_squared);
    evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_squared, relin_keys);
    // evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1_squared);

    // Ciphertext encrypted_result_X_mul_weight1_squared = encrypted_result_X_mul_weight1[0];
    
    if (print_cmd == 1)
    {

        Plaintext result_final_square;
        decryptor.decrypt(encrypted_result_X_mul_weight1_squared, result_final_square);
        
        encoder.decode(result_final_square, vector_X);
        
        // print_matrix(vector_X, row_size);
        for (int i=0; i<COLUMN_1; i++)
        {
            cout<<vector_X[i]<<"\t";
        
        }
        cout<<endl;
    }

    // cout<<context.get_context_data(encrypted_result_X_mul_weight1_squared.parms_id())->chain_index()<<endl;
    cout<<"\n==========================================================\n";
    cout<<"||                ((X*W1)squared)*W2                     ||";
    cout<<"\n==========================================================\n";

    last_parms_id = encrypted_result_X_mul_weight1_squared.parms_id();
    for (int i=0; i<COLUMN_2; i++)
    {
        // cout<<context.get_context_data(encrypted_weight_2[i].parms_id())->chain_index()<<endl;
        encrypted_weight_2[i].scale() = pow(2.0, 40);
        evaluator.mod_switch_to_inplace(encrypted_weight_2[i], last_parms_id);
    }
    cout<<"test OK"<<endl;  

    vector<Ciphertext> encrypted_result_X_mul_weight1_sqr_mul_weight2(COLUMN_2);
    for (int i=0; i < COLUMN_2; i++)
    {
        // cout<< "X multiply weight1["<<i<<"]"<<endl;
        evaluator.multiply(encrypted_result_X_mul_weight1_squared, encrypted_weight_2[i], encrypted_result_X_mul_weight1_sqr_mul_weight2[i]);
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[i], relin_keys);
        // evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[i]);
        // cout << " noise = " << decryptor.invariant_noise_budget(encrypted_result_X_mul_weight1_sqr_mul_weight2[i]) << " bits"<<endl;
    }

    cout<<"test OK"<<endl;  
    for (int i=0; i<COLUMN_2; i++)
    {
        evaluator.total_sum_ckks(encrypted_result_X_mul_weight1_sqr_mul_weight2[i],galois_keys, slot_count);
        
    }

    Plaintext total_sum_ckks;
    decryptor.decrypt(encrypted_result_X_mul_weight1_sqr_mul_weight2[1], total_sum_ckks);
    
    encoder.decode(total_sum_ckks, vector_X);
    
    print_vector(vector_X, 3,7);
    
    // cout<< "\nCreate bitmask vector\n";
    // cout << "Create bitmask again to keep only the one element for the total sum"<<endl;
    
    vector<vector<double>> bit_mask_t_w2;
    vector<double> vector_bit_mask_w2(slot_count, 0ULL);
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
        encoder.encode(bit_mask_t_w2[i], scale, plain_matrixes_bit_mask_w2[i]);
        // encryptor.encrypt(plain_matrixes_bit_mask_w2[i], encrypted_bit_mask_vector_w2[i]);
    }

    cout<<"test OK"<<endl;
    // cout<< "\nMultiply with bitmask\n";
    // cout << "Multiply bitmask_3_t with masked_encrypted_result_X_W0_1"<<endl;
    // int j_w2 = 0;
    for (int i=0; i<COLUMN_2; i++)
    {
       
        evaluator.multiply_plain_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[i], plain_matrixes_bit_mask_w2[i]);
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[i], relin_keys);
        evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[i]);
       
    }
    
    
    // cout<< "Add ciphertexts together"<<endl;

    // evaluator.add_inplace(masked_encrypted_X_mul_weight1_sqr_mul_weight2_firstBlock[0],masked_encrypted_X_mul_weight1_sqr_mul_weight2_secondBlock[0]);
    for (int i=1; i<COLUMN_2; i++)
    {
        evaluator.add_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[0], encrypted_result_X_mul_weight1_sqr_mul_weight2[i]);
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[0], relin_keys);
        evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2[0]);
    }

    if (print_cmd == 1)
    {

        Plaintext result_final_w2;
        decryptor.decrypt(encrypted_result_X_mul_weight1_sqr_mul_weight2[0], result_final_w2);
        
        encoder.decode(result_final_w2, vector_X);
        
        // print_matrix(vector_X, row_size);
        for (int i=0; i<COLUMN_2; i++)
        {
            cout<<vector_X[i]<<"\t";
        }
        cout<<endl;
    }

    cout<<"\n==========================================================\n";
    cout<<"||                ((X*W1)squared)*W2*W3                 ||";
    cout<<"\n==========================================================\n";

   
    Ciphertext result_encrypted_X_mul_weight1_squared_mul_weight2 = encrypted_result_X_mul_weight1_sqr_mul_weight2[0];    
    vector<Ciphertext> encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3(COLUMN_3);
    for (int i=0; i < COLUMN_3; i++)
    {
        // cout<< "X multiply weight1["<<i<<"]"<<endl;
        evaluator.multiply_plain(result_encrypted_X_mul_weight1_squared_mul_weight2, plain_matrixes3[i], encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i]);
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i], relin_keys);
        evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i]);
        // cout << " noise = " << decryptor.invariant_noise_budget(encrypted_result_X_mul_weight1_sqr_mul_weight2[i]) << " bits"<<endl;
    }

    // cout<< "\nTotal sum\n";

    for (int i=0; i<COLUMN_3; i++)
    {
        evaluator.total_sum_ckks(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i],galois_keys, slot_count);
        // evaluator.total_sum_ckks(masked_encrypted_X_mul_weight1_sqr_mul_weight2_secondBlock[i],galois_keys, slot_count);
    }

    // cout<< "\nCreate bitmask vector\n";
    // cout << "Create bitmask again to keep only the one element for the total sum"<<endl;
    
    vector<vector<double>> bit_mask_t_w3;
    vector<double> vector_bit_mask_w3(slot_count, 0ULL);
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
        encoder.encode(bit_mask_t_w3[i], scale, plain_matrixes_bit_mask_w3[i]);
        // encryptor.encrypt(plain_matrixes_bit_mask_w2[i], encrypted_bit_mask_vector_w2[i]);
    }

    
    // cout<< "\nMultiply with bitmask\n";
    // cout << "Multiply bitmask_3_t with masked_encrypted_result_X_W0_1"<<endl;
    // int j_w2 = 0;
    for (int i=0; i<COLUMN_3; i++)
    {
       
        evaluator.multiply_plain_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i], plain_matrixes_bit_mask_w3[i]);
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i], relin_keys);
        evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i]);
       
    }

    // cout<< "Add ciphertexts together"<<endl;

    // evaluator.add_inplace(masked_encrypted_X_mul_weight1_sqr_mul_weight2_firstBlock[0],masked_encrypted_X_mul_weight1_sqr_mul_weight2_secondBlock[0]);
    for (int i=1; i<COLUMN_3; i++)
    {
        evaluator.add_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[0], encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[i]);
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[0], relin_keys);
        evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[0]);
    }

    if (print_cmd == 1)
    {
        Plaintext result_final_w2_w3;
        std::vector<double> result;
        decryptor.decrypt(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[0], result_final_w2_w3);
        
        encoder.decode(result_final_w2_w3, result);
        
        // print_matrix(result, row_size);

        for (int i=0; i<COLUMN_3+1; i++)
        {
            cout<<result[i]<<"\t";
        }
        cout<<endl;
    }
    



    cout<<"\n==========================================================\n";
    cout<<"||                ((X*W1)squared)*W2*W3*W4                ||";
    cout<<"\n==========================================================\n";

    
    Ciphertext result_encrypted_X_mul_weight1_squared_mul_weight2_mul_weight3 = encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3[0];    
    vector<Ciphertext> encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4(COLUMN_4);
    for (int i=0; i < COLUMN_4; i++)
    {
        // cout<< "X multiply weight1["<<i<<"]"<<endl;
        evaluator.multiply_plain(result_encrypted_X_mul_weight1_squared_mul_weight2_mul_weight3, plain_matrixes4[i], encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i]);
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i], relin_keys);
        evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i]);
        // cout << " noise = " << decryptor.invariant_noise_budget(encrypted_result_X_mul_weight1_sqr_mul_weight2[i]) << " bits"<<endl;
    }

    // cout<< "\nTotal sum\n";

    for (int i=0; i<COLUMN_4; i++)
    {
        evaluator.total_sum_ckks(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i],galois_keys, slot_count);
        // evaluator.total_sum_ckks(masked_encrypted_X_mul_weight1_sqr_mul_weight2_secondBlock[i],galois_keys, slot_count);
    }

    // cout<< "\nCreate bitmask vector\n";
    // cout << "Create bitmask again to keep only the one element for the total sum"<<endl;
    
    vector<vector<double>> bit_mask_t_w4;
    vector<double> vector_bit_mask_w4(slot_count, 0ULL);
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
        encoder.encode(bit_mask_t_w4[i], scale, plain_matrixes_bit_mask_w4[i]);
        // encryptor.encrypt(plain_matrixes_bit_mask_w2[i], encrypted_bit_mask_vector_w2[i]);
    }

    
    // cout<< "\nMultiply with bitmask\n";
    // cout << "Multiply bitmask_3_t with masked_encrypted_result_X_W0_1"<<endl;
    // int j_w2 = 0;
    for (int i=0; i<COLUMN_4; i++)
    {
       
        evaluator.multiply_plain_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i], plain_matrixes_bit_mask_w4[i]);
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i], relin_keys);
        evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i]);
       
    }

    // cout<< "Add ciphertexts together"<<endl;

    // evaluator.add_inplace(masked_encrypted_X_mul_weight1_sqr_mul_weight2_firstBlock[0],masked_encrypted_X_mul_weight1_sqr_mul_weight2_secondBlock[0]);
    for (int i=1; i<COLUMN_4; i++)
    {
        evaluator.add_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[0], encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[i]);
        evaluator.relinearize_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[0], relin_keys);
        evaluator.rescale_to_next_inplace(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[0]);
    }

    // Get ending timepoint
    auto stop3 = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop3 - start3);

    if (print_cmd == 1)
    {
        Plaintext result_final_w2_w3_w4_final;
        
        decryptor.decrypt(encrypted_result_X_mul_weight1_sqr_mul_weight2_mul_weight3_mul_weight4[0], result_final_w2_w3_w4_final);
        
        encoder.decode(result_final_w2_w3_w4_final, result);
        
        // print_matrix(result, row_size);

        for (int i=0; i<COLUMN_4+1; i++)
        {
            cout<<result[i]<<"\t";
        }
        cout<<endl;
    }
    

       
    cout << "Time taken to encode and encrypt input email vector: "
         << duration_encryption_encode_email.count() << " miliseconds" << endl;

    cout << "Time taken to encode and encrypt model weight: "
         << duration_encryption_encode_model_weight.count() << " miliseconds" << endl;
         
    cout << "Time taken to predict/classifying: "
         << duration.count() << " seconds" << endl;



    double sum_exp = exp(result[0]) + exp(result[1]);
    cout<<"sum_exp = "<<sum_exp<<endl;
    double score_0 = double(exp(result[0])/sum_exp);
    double score_1 = double(exp(result[1])/sum_exp);

    cout<<"softmax activation: score_0 = "<<score_0<<"\t"<<"score_1 ="<<score_1<<endl;

    
    double score_0_sigmoid = double(1/(1+exp(-result[0])));
    double score_1_sigmoid = double(1/(1+exp(-result[1])));

    cout<<"sigmoid activation: score_0 = "<<score_0_sigmoid<<"\t"<<"score_1 = "<<score_1_sigmoid<<endl;


    if (test_plain == 1)
    {
        cout<<"\nComputation in plaintexts"<<endl;
        vector<double> X(input_vector_size);
        for (int i=0; i<input_vector_size; i++)
        {
            X[i] = input_email_vector[test_index][i];
        }
        // cout<<"OK"<<endl;
        
        vector<double> X_mul_W1(COLUMN_1, 0);

        for (int i=0; i<COLUMN_1; i++)
        {
            for (int j =0; j<input_vector_size; j++)
            {
                X_mul_W1[i] = X_mul_W1[i] + X[j]*W1[i][j];
            }
            
        }
        cout<<"\nX_mul_weight1: "<<endl;
        for (int i=0; i < COLUMN_1; i++)
        {
            cout<<X_mul_W1[i]<<"\t";
        }
        cout<<endl;

        cout<<"\nX_mul_W1_square:"<<endl;
        vector<double> X_mul_W1_square(COLUMN_1);
        for (int i=0; i < COLUMN_1; i++)
        {
            X_mul_W1_square[i] = pow(X_mul_W1[i],2);
        }
        // cout<<endl;

        for (int i=0; i < COLUMN_1; i++)
        {
            cout<<X_mul_W1_square[i]<<"\t";
        }
        cout<<endl;

        vector<double> X_mul_W1_squared_mul_W2(COLUMN_2, 0);
        for (int i=0; i<COLUMN_2; i++)
        {
            for (int j=0; j<COLUMN_1; j++)
            {
                X_mul_W1_squared_mul_W2[i] = X_mul_W1_squared_mul_W2[i] + X_mul_W1_square[j]*W2[i][j];
            //    cout<<"X_mul_W1_squared_mul_W2[i] = "<<i<<":"<<X_mul_W1_squared_mul_W2[i]<<endl;
            }
            // cout<<endl; 
        }
        cout<<"\nX_mul_W1_squared_mul_W2:"<<endl;
        for (int i=0; i < COLUMN_2; i++)
        {
            cout<<X_mul_W1_squared_mul_W2[i]<<"\t";
        }
        cout<<endl;

        vector<double> X_mul_W1_squared_mul_W2_mul_W3(COLUMN_3, 0);
        for (int i=0; i<COLUMN_3; i++)
        {
            for (int j=0; j<COLUMN_2; j++)
            {
                X_mul_W1_squared_mul_W2_mul_W3[i] = X_mul_W1_squared_mul_W2_mul_W3[i] + X_mul_W1_squared_mul_W2[j]*W3[i][j];
            } 
        }
        cout<<"\nX_mul_W1_squared_mul_W2_mul_W3:"<<endl;
        for (int i=0; i < COLUMN_3; i++)
        {
            cout<<X_mul_W1_squared_mul_W2_mul_W3[i]<<"\t";
        }
        cout<<endl;

        vector<double> X_mul_W1_squared_mul_W2_mul_W3_mul_W4(COLUMN_4, 0);
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

        double sum_exp_t = exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[0]) + exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[1]);
        cout<<"sum_exp = "<<sum_exp_t<<endl;
        double score_0_t = double(exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[0])/sum_exp_t);
        double score_1_t = double(exp(X_mul_W1_squared_mul_W2_mul_W3_mul_W4[1])/sum_exp_t);

        cout<<"softmax activation: score_0 = "<<score_0_t<<"\t"<<"score_1 ="<<score_1_t<<endl;

        
        double score_0_sigmoid_t = double(1/(1+exp(-X_mul_W1_squared_mul_W2_mul_W3_mul_W4[0])));
        double score_1_sigmoid_t = double(1/(1+exp(-X_mul_W1_squared_mul_W2_mul_W3_mul_W4[1])));

        cout<<"sigmoid activation: score_0 = "<<score_0_sigmoid_t<<"\t"<<"score_1 = "<<score_1_sigmoid_t<<endl;
    }

} 

