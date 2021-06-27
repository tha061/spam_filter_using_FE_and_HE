// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "examples.h"

using namespace std;
using namespace seal;

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
void mat_mul()
{
    print_example_banner("Example: Matrix multiplication in BFV");

    EncryptionParameters parms(scheme_type::bfv);

    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));

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
    // size_t slot_count = 4;
    cout << "Plaintext Batch size: "<<slot_count<<endl;
    cout << "Plaintext matrix row size: " << row_size << endl;

    vector<uint64_t> vector_X(slot_count, 0ULL); // a vector of size 4 with 0s
    for (int i=0; i < 2000; i++)
    {
        vector_X[i] = 1ULL;
        vector_X[i+2000] = 1ULL;
        vector_X[i+4096] = 1ULL;
        vector_X[i+6096] = 1ULL;
    }
    


    cout << "Input plaintext vector_X:" << endl;
    print_matrix(vector_X, row_size);

    Plaintext plain_matrix;
    cout << endl;
    print_line(__LINE__);
    cout << "Encode and encrypt vector_X" << endl;
    batch_encoder.encode(vector_X, plain_matrix);
    Ciphertext encrypted_vector_X;
    encryptor.encrypt(plain_matrix, encrypted_vector_X);
    cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_vector_X) << " bits"
         << endl;
    cout << endl;

    // encoding matrix W1
    vector<uint64_t> matrix_W0(slot_count, 0ULL); 
    for (int i=0; i < 2000; i++)
    {
        matrix_W0[i] = 1ULL;
        matrix_W0[i+2000] = 2ULL;
        matrix_W0[i+4096] = 3ULL;
        matrix_W0[i+6096] = 4ULL;
    }
    
    cout << "Input plaintext matrix_W0 that contains 4 columns of the matrix W1:" << endl;
    print_matrix(matrix_W0, row_size);
    // cout<<matrix_W0[4095]<<endl;
    // cout<<matrix_W0[4096]<<endl;
    // cout<<matrix_W0[6095]<<endl;
    // cout<<matrix_W0[6096]<<endl;
    // cout<<matrix_W0[8096]<<endl;


    Plaintext plain_matrix_W0;
    cout << endl;
    // print_line(__LINE__);
    // cout << "Encode and encrypt vector_X" << endl;
    batch_encoder.encode(matrix_W0, plain_matrix_W0);
    Ciphertext encrypted_matrix_W0;
    encryptor.encrypt(plain_matrix_W0, encrypted_matrix_W0);
    cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_matrix_W0) << " bits"
         << endl;
    cout << endl;

    vector<uint64_t> matrix_W1(slot_count, 0ULL); 
    for (int i=0; i < 2000; i++)
    {
        matrix_W1[i] = 1ULL;
        matrix_W1[i+2000] = 1ULL;
        matrix_W1[i+4000] = 1ULL;
        matrix_W1[i+6000] = 1ULL;
    }
    
    // cout << "Input plaintext matrix_W1 that contains 4 columns of the matrix W1:" << endl;
    // print_matrix(matrix_W1, row_size);

    Plaintext plain_matrix_W1;
    cout << endl;
    // print_line(__LINE__);
    // cout << "Encode and encrypt vector_X" << endl;
    batch_encoder.encode(matrix_W1, plain_matrix_W1);
    Ciphertext encrypted_matrix_W1;
    encryptor.encrypt(plain_matrix_W1, encrypted_matrix_W1);
    // cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_matrix_W1) << " bits"
    //      << endl;
    // cout << endl;


    cout<< "\n ==========Step 1========\n";
    cout << "Multiply element-wise of encrypted_vector_X with encrypted_matrix_W0"<<endl;
    Ciphertext encrypted_result_X_W0, encrypted_result_X_W1;
    evaluator.multiply(encrypted_vector_X, encrypted_matrix_W0, encrypted_result_X_W0);
    cout << "relinearize to a ciphertext of size 2"<<endl;
    evaluator.relinearize_inplace(encrypted_result_X_W0, relin_keys);
    cout << "Multiply element-wise of encrypted_vector_X with encrypted_matrix_W0"<<endl;
    evaluator.multiply(encrypted_vector_X, encrypted_matrix_W1, encrypted_result_X_W1);
    cout << "relinearize to a ciphertext of size 2"<<endl;
    evaluator.relinearize_inplace(encrypted_result_X_W1, relin_keys);

    cout<< "\n ==========Step 2========\n";
    cout << "Create bit-1 masking: bm1, bm2 bm3, bm4"<<endl;
    vector<uint64_t> bitmask_1(slot_count, 0ULL); // a vector of size 4 with 0s
    for (int i=0; i < 2000; i++)
    {
        bitmask_1[i] = 1ULL;
    }

    Plaintext plain_bitmask_1;
    cout << endl;
    // print_line(__LINE__);
    // cout << "Encode and encrypt vector_X" << endl;
    batch_encoder.encode(bitmask_1, plain_bitmask_1);
    Ciphertext encrypted_bitmask_1;
    encryptor.encrypt(plain_bitmask_1, encrypted_bitmask_1);
    // cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_bitmask_1) << " bits"
    //      << endl;
    // cout << endl;

    vector<uint64_t> bitmask_2(slot_count, 0ULL); // a vector of size 4 with 0s
    for (int i=0; i < 2000; i++)
    {
        bitmask_2[i+2000] = 1ULL;
    }

    Plaintext plain_bitmask_2;
    cout << endl;
    // print_line(__LINE__);
    // cout << "Encode and encrypt vector_X" << endl;
    batch_encoder.encode(bitmask_2, plain_bitmask_2);
    Ciphertext encrypted_bitmask_2;
    encryptor.encrypt(plain_bitmask_2, encrypted_bitmask_2);
    // cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_bitmask_2) << " bits"
        //  << endl;
    // cout << endl;

    vector<uint64_t> bitmask_3(slot_count, 0ULL); // a vector of size 4 with 0s
    for (int i=0; i < 2000; i++)
    {
        bitmask_3[i+4096] = 1ULL;
        
    }

    Plaintext plain_bitmask_3;
    cout << endl;
    // print_line(__LINE__);
    // cout << "Encode and encrypt vector_X" << endl;
    batch_encoder.encode(bitmask_3, plain_bitmask_3);
    Ciphertext encrypted_bitmask_3;
    encryptor.encrypt(plain_bitmask_3, encrypted_bitmask_3);
    // cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_bitmask_2) << " bits"
        //  << endl;
    // cout << endl;

    vector<uint64_t> bitmask_4(slot_count, 0ULL); // a vector of size 4 with 0s
    for (int i=0; i < 2000; i++)
    {
        bitmask_4[i+6096] = 1ULL;
        
    }

    Plaintext plain_bitmask_4;
    cout << endl;
    // print_line(__LINE__);
    // cout << "Encode and encrypt vector_X" << endl;
    batch_encoder.encode(bitmask_4, plain_bitmask_4);
    Ciphertext encrypted_bitmask_4;
    encryptor.encrypt(plain_bitmask_4, encrypted_bitmask_4);
    // cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_bitmask_2) << " bits"
        //  << endl;
    // cout << endl;

    // cout << "Testing: print encrypted_result_X_W0 decrypted"<<endl;
    // Plaintext tmp;
    // decryptor.decrypt(encrypted_result_X_W0, tmp);
    
    // batch_encoder.decode(tmp, vector_X);
    
    // print_matrix(vector_X, row_size);


    cout<< "\n ==========Step 3========\n";
    cout << "multiple encrypted_result_X_W0 with encrypted bitmask1,2,3,4"<<endl;
    Ciphertext masked_encrypted_result_X_W0_1;
    evaluator.multiply(encrypted_result_X_W0, encrypted_bitmask_1, masked_encrypted_result_X_W0_1);
    cout << "relinearize to a ciphertext of size 2"<<endl;
    evaluator.relinearize_inplace(masked_encrypted_result_X_W0_1, relin_keys);

    cout << "Testing: print masked_encrypted_result_X_W0_1 decrypted"<<endl;
    Plaintext tmp1;
    decryptor.decrypt(masked_encrypted_result_X_W0_1, tmp1);
    
    batch_encoder.decode(tmp1, vector_X);
    print_matrix(vector_X, row_size);
    
    
    Ciphertext masked_encrypted_result_X_W0_2;
    evaluator.multiply(encrypted_result_X_W0, encrypted_bitmask_2, masked_encrypted_result_X_W0_2);
    cout << "relinearize to a ciphertext of size 2"<<endl;
    evaluator.relinearize_inplace(masked_encrypted_result_X_W0_2, relin_keys);

    cout << "Testing: print masked_encrypted_result_X_W0_2 decrypted"<<endl;
    Plaintext tmp2;
    decryptor.decrypt(masked_encrypted_result_X_W0_2, tmp1);
    
    batch_encoder.decode(tmp2, vector_X);
    
    print_matrix(vector_X, row_size);
    
    Ciphertext masked_encrypted_result_X_W0_3;
    evaluator.multiply(encrypted_result_X_W0, encrypted_bitmask_3, masked_encrypted_result_X_W0_3);
    cout << "relinearize to a ciphertext of size 2"<<endl;
    evaluator.relinearize_inplace(masked_encrypted_result_X_W0_3, relin_keys);

    // cout << "Testing: print masked_encrypted_result_X_W0_3 decrypted"<<endl;
    Plaintext tmp3;
    decryptor.decrypt(masked_encrypted_result_X_W0_3, tmp3);
    
    batch_encoder.decode(tmp3, vector_X);
    
    print_matrix(vector_X, row_size);

    Ciphertext masked_encrypted_result_X_W0_4;
    evaluator.multiply(encrypted_result_X_W0, encrypted_bitmask_4, masked_encrypted_result_X_W0_4);
    cout << "relinearize to a ciphertext of size 2"<<endl;
    evaluator.relinearize_inplace(masked_encrypted_result_X_W0_4, relin_keys);

    cout << "Testing: print masked_encrypted_result_X_W0_4 decrypted"<<endl;
    Plaintext tmp4;
    decryptor.decrypt(masked_encrypted_result_X_W0_4, tmp4);
    
    batch_encoder.decode(tmp4, vector_X);
    
    print_matrix(vector_X, row_size);
    // for (int i=0; i<slot_count; i++)
    // {
    //     cout<<vector_X[i]<<"\t";
    // }
    cout<<endl;


    cout<< "\n ==========Step 4========\n";
    cout<< "Total sum on masked_encrypted_result_X_W0_1"<<endl;
    evaluator.total_sum(masked_encrypted_result_X_W0_1, galois_keys, slot_count);

    
            
    // cout <<"To extract the total sum of the masked_encrypted_result_X_W0_1"<<endl;
    Plaintext plain_result_total_sum_result_X_W0_1;
    decryptor.decrypt(masked_encrypted_result_X_W0_1, plain_result_total_sum_result_X_W0_1);
    
    batch_encoder.decode(plain_result_total_sum_result_X_W0_1, vector_X);
    
    print_matrix(vector_X, row_size);


    cout << "Total sum of encrypted_result_X_W0_2"<<endl;


    evaluator.total_sum(masked_encrypted_result_X_W0_2, galois_keys, slot_count);

    // cout <<"To extract the total sum of the masked_encrypted_result_X_W0_2"<<endl;
    Plaintext plain_result_total_sum_result_X_W0_2;
    decryptor.decrypt(masked_encrypted_result_X_W0_2, plain_result_total_sum_result_X_W0_2);
    
    batch_encoder.decode(plain_result_total_sum_result_X_W0_2, vector_X);
    
    print_matrix(vector_X, row_size);

    cout << "Total sum of encrypted_result_X_W0_3"<<endl;
    evaluator.total_sum(masked_encrypted_result_X_W0_3, galois_keys, slot_count);

    // cout <<"To extract the total sum of the masked_encrypted_result_X_W0_3"<<endl;
    Plaintext plain_result_total_sum_result_X_W0_3;
    decryptor.decrypt(masked_encrypted_result_X_W0_3, plain_result_total_sum_result_X_W0_3);
    
    batch_encoder.decode(plain_result_total_sum_result_X_W0_3, vector_X);
    
    print_matrix(vector_X, row_size);

    cout << "Total sum of encrypted_result_X_W0_4"<<endl;
    evaluator.total_sum(masked_encrypted_result_X_W0_4, galois_keys, slot_count);

    // cout <<"To extract the total sum of the masked_encrypted_result_X_W0_4"<<endl;
    Plaintext plain_result_total_sum_result_X_W0_4;
    decryptor.decrypt(masked_encrypted_result_X_W0_4, plain_result_total_sum_result_X_W0_4);
    
    batch_encoder.decode(plain_result_total_sum_result_X_W0_4, vector_X);
    
    print_matrix(vector_X, row_size);

    cout<< "\n ==========Step 5========\n";
    cout << "Create bitmask again to keep only the one element for the total sum"<<endl;
    vector<uint64_t> bitmask_1_t(slot_count, 0ULL); 
    bitmask_1_t[0] = 1ULL;
    
    vector<uint64_t> bitmask_2_t(slot_count, 0ULL); 
    bitmask_2_t[1] = 1ULL;
    
    vector<uint64_t> bitmask_3_t(slot_count, 0ULL); 
    bitmask_3_t[4096+2] = 1ULL;

    vector<uint64_t> bitmask_4_t(slot_count, 0ULL); 
    bitmask_4_t[4096+3] = 1ULL;

    Plaintext plain_bitmask_1_t;
    batch_encoder.encode(bitmask_1_t, plain_bitmask_1_t);
    Ciphertext encrypted_bitmask_1_t;
    encryptor.encrypt(plain_bitmask_1_t, encrypted_bitmask_1_t);

    Plaintext plain_bitmask_2_t;
    batch_encoder.encode(bitmask_2_t, plain_bitmask_2_t);
    Ciphertext encrypted_bitmask_2_t;
    encryptor.encrypt(plain_bitmask_2_t, encrypted_bitmask_2_t);


    Plaintext plain_bitmask_3_t;
    batch_encoder.encode(bitmask_3_t, plain_bitmask_3_t);
    Ciphertext encrypted_bitmask_3_t;
    encryptor.encrypt(plain_bitmask_3_t, encrypted_bitmask_3_t);

    Plaintext plain_bitmask_4_t;
    batch_encoder.encode(bitmask_4_t, plain_bitmask_4_t);
    Ciphertext encrypted_bitmask_4_t;
    encryptor.encrypt(plain_bitmask_4_t, encrypted_bitmask_4_t);

    cout<< "\n ==========Step 6========\n";
    cout << "Multiply bitmask_1_t with masked_encrypted_result_X_W0_1"<<endl;
    evaluator.multiply_inplace(masked_encrypted_result_X_W0_1, encrypted_bitmask_1_t);
    evaluator.relinearize_inplace(masked_encrypted_result_X_W0_1, relin_keys);

    cout << "Multiply bitmask_1_t with masked_encrypted_result_X_W0_2"<<endl;
    evaluator.multiply_inplace(masked_encrypted_result_X_W0_2, encrypted_bitmask_2_t);
    evaluator.relinearize_inplace(masked_encrypted_result_X_W0_2, relin_keys);
    
    cout << "Multiply bitmask_1_t with masked_encrypted_result_X_W0_3"<<endl;
    evaluator.multiply_inplace(masked_encrypted_result_X_W0_3, encrypted_bitmask_3_t);
    evaluator.relinearize_inplace(masked_encrypted_result_X_W0_3, relin_keys);

    cout << "Multiply bitmask_1_t with masked_encrypted_result_X_W0_4"<<endl;
    evaluator.multiply_inplace(masked_encrypted_result_X_W0_4, encrypted_bitmask_4_t);
    evaluator.relinearize_inplace(masked_encrypted_result_X_W0_4, relin_keys);


    cout <<"To extract the total sum of the masked_encrypted_result_X_W0_1"<<endl;
    Plaintext plain_result_total_sum_result_X_W0_1_t;
    decryptor.decrypt(masked_encrypted_result_X_W0_1, plain_result_total_sum_result_X_W0_1_t);
    
    batch_encoder.decode(plain_result_total_sum_result_X_W0_1_t, vector_X);
    
    print_matrix(vector_X, row_size);
    // cout<<vector_X[0]<<endl;
    // cout<<vector_X[6096]<<endl;

    cout <<"To extract the total sum of the masked_encrypted_result_X_W0_2"<<endl;
    Plaintext plain_result_total_sum_result_X_W0_2_t;
    decryptor.decrypt(masked_encrypted_result_X_W0_2, plain_result_total_sum_result_X_W0_2_t);
    
    batch_encoder.decode(plain_result_total_sum_result_X_W0_2_t, vector_X);
    
    print_matrix(vector_X, row_size);
    // cout<<vector_X[2000]<<endl;
    // cout<<vector_X[6096]<<endl;

    cout <<"To extract the total sum of the masked_encrypted_result_X_W0_3"<<endl;
    Plaintext plain_result_total_sum_result_X_W0_3_t;
    decryptor.decrypt(masked_encrypted_result_X_W0_3, plain_result_total_sum_result_X_W0_3_t);
    
    batch_encoder.decode(plain_result_total_sum_result_X_W0_3_t, vector_X);
    
    print_matrix(vector_X, row_size);
    // cout<<vector_X[4096]<<endl;
    // cout<<vector_X[6097]<<endl;

    cout <<"To extract the total sum of the masked_encrypted_result_X_W0_4"<<endl;
    Plaintext plain_result_total_sum_result_X_W0_4_t;
    decryptor.decrypt(masked_encrypted_result_X_W0_4, plain_result_total_sum_result_X_W0_4_t);
    
    batch_encoder.decode(plain_result_total_sum_result_X_W0_4_t, vector_X);
    
    print_matrix(vector_X, row_size);
    // cout<<vector_X[6096]<<endl;
    // cout<<vector_X[0]<<endl;


    cout<< "=====Step 7: Add these together======"<<endl;
    // vector<Ciphertext> ct_result(4);
    // ct_result[0] = masked_encrypted_result_X_W0_1;
    // ct_result[1] = masked_encrypted_result_X_W0_2;
    // ct_result[2] = masked_encrypted_result_X_W0_3;
    // ct_result[3] = masked_encrypted_result_X_W0_4;

    // Ciphertext ct_final;
    // evaluator.add_many(ct_result, ct_final);

    // Plaintext result;
    // decryptor.decrypt(ct_final, result);
    
    // batch_encoder.decode(result, vector_X);
    
    // print_matrix(vector_X, row_size);

    evaluator.add_inplace(masked_encrypted_result_X_W0_1, masked_encrypted_result_X_W0_2);
    evaluator.add_inplace(masked_encrypted_result_X_W0_1, masked_encrypted_result_X_W0_3);
    evaluator.add_inplace(masked_encrypted_result_X_W0_1, masked_encrypted_result_X_W0_4);

    Plaintext result;
    decryptor.decrypt(masked_encrypted_result_X_W0_1, result);
    
    batch_encoder.decode(result, vector_X);
    
    print_matrix(vector_X, row_size);



} 
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
// function to convert decimal to binary 
void decToBinary(int n) 
{ 
    // array to store binary number 
    int binaryNum[32]; 
  
    // counter for binary array 
    int i = 0; 
    while (n > 0) { 
  
        // storing remainder in binary array 
        binaryNum[i] = n % 2; 
        n = n / 2; 
        i++; 
    } 
  
    // printing binary array in reverse order 
    for (int j = i - 1; j >= 0; j--) 
        cout << binaryNum[j]; 
} 

void total_sum_bfv_example()
{
    print_example_banner("Example: Dot Product in BFV");

    EncryptionParameters parms(scheme_type::bfv);

    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::BFVDefault(poly_modulus_degree));
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));

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
    size_t row_size = slot_count / 2;
    // size_t slot_count = 4;
    cout << "Plaintext Batch size: "<<slot_count<<endl;
    cout << "Plaintext matrix row size: " << row_size << endl;

    vector<uint64_t> vector_X(slot_count, 0ULL); // a vector of size 4 with 0s
    vector_X[0] = 1ULL;
    vector_X[1] = 1ULL;
    vector_X[2] = 1ULL;
    vector_X[3] = 1ULL;
    vector_X[4] = 1ULL;
    // vector_X[row_size] = 100ULL;
    // vector_X[row_size + 1] = 100ULL;
    // vector_X[row_size + 2] = 100ULL;
    // vector_X[row_size + 3] = 100ULL;


    cout << "Input plaintext matrix1:" << endl;
    print_matrix(vector_X, row_size);

    
    vector<uint64_t> vector_X2(slot_count, 0ULL); // a vector of size 4 with 0s
    vector_X2[0] = 0ULL;
    vector_X2[1] = 1ULL;
    vector_X2[2] = 2ULL;
    vector_X2[3] = 3ULL;
    vector_X2[4] = 4ULL;
    // vector_X2[row_size] = 4ULL;
    // vector_X2[row_size + 1] = 5ULL;
    // vector_X2[row_size + 2] = 6ULL;
    // vector_X2[row_size + 3] = 7ULL;

    
    cout << "Input plaintext matrix2:" << endl;
    
    print_matrix(vector_X2, row_size);

    

    /*
    First we use BatchEncoder to encode the matrix into a plaintext. We encrypt
    the plaintext as usual.
    */
    Plaintext plain_matrix;
    cout << endl;
    print_line(__LINE__);
    cout << "Encode and encrypt matrix1" << endl;
    batch_encoder.encode(vector_X, plain_matrix);
    Ciphertext encrypted_matrix;
    encryptor.encrypt(plain_matrix, encrypted_matrix);
    cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
         << endl;
    cout << endl;

    Plaintext plain_matrix2;
    cout << endl;
    print_line(__LINE__);
    cout << "Encode and encrypt matrix2" << endl;
    batch_encoder.encode(vector_X2, plain_matrix2);
    Ciphertext encrypted_matrix2;
    encryptor.encrypt(plain_matrix2, encrypted_matrix2);
    cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_matrix2) << " bits"
         << endl;
    cout << endl;


    cout <<"\nTest the total sum of (encrypted_matrix1 . encrypted_matrix2)"<<endl;

    Ciphertext encrypted_total_sum_test;

    evaluator.total_sum_bfv(encrypted_matrix, encrypted_matrix2, encrypted_total_sum_test, galois_keys, relin_keys, slot_count);

    Plaintext plain_result_total_sum_test;
    decryptor.decrypt(encrypted_total_sum_test, plain_result_total_sum_test);
    
    batch_encoder.decode(plain_result_total_sum_test, vector_X);
    
    print_matrix(vector_X, row_size);

    cout << endl;



    vector<uint64_t> mask_1_matrix(slot_count, 0ULL); // a vector having bit 1 at the first element to extract the total sum as a number
    mask_1_matrix[0] = 1ULL;

    cout << "Encode and encrypt mask 1 matrix" << endl;
    Plaintext plain_matrix_mask_1;
    batch_encoder.encode(mask_1_matrix, plain_matrix_mask_1);
    Ciphertext encrypted_mask_1_matrix;
    encryptor.encrypt(plain_matrix_mask_1, encrypted_mask_1_matrix);
    cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_mask_1_matrix) << " bits"
         << endl;
    cout << endl;
    
    cout << "Multiply the mask vector with total sum vector, result is stored in encrypted_total_sum_test" <<endl;
    
    evaluator.multiply_inplace(encrypted_total_sum_test, encrypted_mask_1_matrix);

    // vector<uint64_t> mask_2_matrix(slot_count, 0ULL); // a vector having bit 1 at the first element to extract the total sum as a number
    // mask_2_matrix[0] = 1ULL;
    // vector<Ciphertext> ct_tmp;

    // for (int i = 0; i < slot_count; i++)
    // {
    //     encryptor.encrypt(mask_2_matrix[i], ct_tmp[i]);
    // }

    Plaintext plain_result_dot_product;
    decryptor.decrypt(encrypted_total_sum_test, plain_result_dot_product);
    
    batch_encoder.decode(plain_result_dot_product, vector_X);
    
    print_matrix(vector_X, row_size);
    // cout<< "plain_result_dot_product ="<< plain_result_dot_product.to_string() << endl;

    /**
    cout <<"\nTest the square of encrypted_matrix_result:"<<endl;

    Ciphertext encrypted_total_sum_squared;
    evaluator.square(encrypted_total_sum_test, encrypted_total_sum_squared);

    Plaintext plain_result_total_sum_squared;
    decryptor.decrypt(encrypted_total_sum_squared, plain_result_total_sum_squared);
    
    batch_encoder.decode(plain_result_total_sum_squared, vector_X);
    
    print_matrix(vector_X, row_size);

    cout << endl;

    */




    
} 






 

