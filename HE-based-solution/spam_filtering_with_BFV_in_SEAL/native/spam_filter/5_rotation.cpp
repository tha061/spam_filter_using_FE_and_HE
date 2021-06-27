// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "examples.h"

using namespace std;
using namespace seal;

/*
Both the BFV scheme (with BatchEncoder) as well as the CKKS scheme support native
vectorized computations on encrypted numbers. In addition to computing slot-wise,
it is possible to rotate the encrypted vectors cyclically.
*/
void example_rotation_bfv()
{
    print_example_banner("Example: Rotation / Rotation in BFV");

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
    // RelinKeys relin_keys;
    // keygen.create_relin_keys(relin_keys);
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    BatchEncoder batch_encoder(context);
    // size_t slot_count = batch_encoder.slot_count();
    // size_t row_size = 1; //slot_count / 2;
    // cout << "Plaintext matrix row size: " << row_size << endl;
    // cout << "slot_count: "<<slot_count<<endl;

    vector<uint64_t> pod_matrix(4, 1);
    pod_matrix[0] = 0;
    pod_matrix[1] = 1;
    pod_matrix[2] = 2;
    pod_matrix[3] = 3;

    
    vector<uint64_t> pod_matrix2(4, 1);
    pod_matrix2[0] = 100;
    pod_matrix2[1] = 100;
    pod_matrix2[2] = 100;
    pod_matrix2[3] = 100;
    // pod_matrix[0] = 0ULL;
    // pod_matrix[1] = 1ULL;
    // pod_matrix[2] = 2ULL;
    // pod_matrix[3] = 3ULL;
    // pod_matrix[row_size] = 4ULL;
    // pod_matrix[row_size + 1] = 5ULL;
    // pod_matrix[row_size + 2] = 6ULL;
    // pod_matrix[row_size + 3] = 7ULL;

    cout << "Input plaintext matrix:" << endl;
    for (int i=0; i<4; i++)
    {
        cout<<pod_matrix[i]<<"\t";
    }

    cout<<endl;
     cout << "Input plaintext matrix2:" << endl;
    for (int i=0; i<4; i++)
    {
        cout<<pod_matrix2[i]<<"\t";
    }

    // print_matrix(pod_matrix, row_size);

    /*
    First we use BatchEncoder to encode the matrix into a plaintext. We encrypt
    the plaintext as usual.
    */
    Plaintext plain_matrix;
    print_line(__LINE__);
    cout << "Encode and encrypt." << endl;
    batch_encoder.encode(pod_matrix, plain_matrix);
    Ciphertext encrypted_matrix;
    encryptor.encrypt(plain_matrix, encrypted_matrix);
    cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
         << endl;
    cout << endl;

    Plaintext plain_matrix2;
    print_line(__LINE__);
    cout << "Encode and encrypt." << endl;
    batch_encoder.encode(pod_matrix2, plain_matrix2);
    Ciphertext encrypted_matrix2;
    encryptor.encrypt(plain_matrix2, encrypted_matrix2);
    cout << "    + Noise budget in fresh encryption: " << decryptor.invariant_noise_budget(encrypted_matrix2) << " bits"
         << endl;
    cout << endl;


    cout << "Compute encrypted_result (matrix * matrix2)." << endl;
    Ciphertext encrypted_result;
    evaluator.multiply(encrypted_matrix, encrypted_matrix2, encrypted_result);
    cout << "    + size of encrypted_result: " << encrypted_result.size() << endl;
    cout << "    + noise budget in encrypted_result: " << decryptor.invariant_noise_budget(encrypted_result) << " bits"
         << endl;

    cout << "Decrypt encrypted_result." << endl;
    Plaintext plain_result;
    decryptor.decrypt(encrypted_result, plain_result);
    batch_encoder.decode(plain_result, pod_matrix);
    for (int i =0; i<4; i++)
    {
        cout << pod_matrix[i] <<"\t";
    }

    cout << endl;

    cout << "Now calculate the dot product in encrypted_result" << endl;

    print_line(__LINE__);
    cout << "Generate relinearization keys." << endl;
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    evaluator.relinearize_inplace(encrypted_result, relin_keys);
    cout << "    + size of encrypted value (after relinearization): " << encrypted_result.size() << endl;

    /*
    Rotations require yet another type of special key called `Galois keys'. These
    are easily obtained from the KeyGenerator.
    */
    GaloisKeys galois_keys;
    keygen.create_galois_keys(galois_keys);

    /*
    Now rotate both matrix rows 1 step to the right, decrypt, decode, and print.
    */
    print_line(__LINE__);
    cout << "Rotate rows 1 steps right." << endl;
    evaluator.rotate_rows_inplace(encrypted_result, 1, galois_keys);
    // evaluator.rotate_rows_inplace(encrypted_result, 1, galois_keys);
    Plaintext plain_result_rotate;
    cout << "    + Noise budget after rotation: " << decryptor.invariant_noise_budget(encrypted_result) << " bits"
         << endl;
    cout << "    + Decrypt and decode ...... Correct." << endl;
    decryptor.decrypt(encrypted_result, plain_result_rotate);
    batch_encoder.decode(plain_result_rotate, pod_matrix);

    for (int i =0; i<4; i++)
    {
        cout << pod_matrix[i] <<"\t";
    }

    cout << endl;
    
    // /*
    // We can also rotate the columns, i.e., swap the rows.
    // */
    // print_line(__LINE__);
    // cout << "Rotate columns." << endl;
    // evaluator.rotate_columns_inplace(encrypted_matrix, galois_keys);
    // cout << "    + Noise budget after rotation: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
    //      << endl;
    // cout << "    + Decrypt and decode ...... Correct." << endl;
    // decryptor.decrypt(encrypted_matrix, plain_result);
    // batch_encoder.decode(plain_result, pod_matrix);
    // print_matrix(pod_matrix, row_size);

    // /*
    // Finally, we rotate the rows 4 steps to the right, decrypt, decode, and print.
    // */
    // print_line(__LINE__);
    // cout << "Rotate rows 4 steps right." << endl;
    // evaluator.rotate_rows_inplace(encrypted_matrix, -4, galois_keys);
    // cout << "    + Noise budget after rotation: " << decryptor.invariant_noise_budget(encrypted_matrix) << " bits"
    //      << endl;
    // cout << "    + Decrypt and decode ...... Correct." << endl;
    // decryptor.decrypt(encrypted_matrix, plain_result);
    // batch_encoder.decode(plain_result, pod_matrix);
    // print_matrix(pod_matrix, row_size);

    // /*
    // Note that rotations do not consume any noise budget. However, this is only
    // the case when the special prime is at least as large as the other primes. The
    // same holds for relinearization. Microsoft SEAL does not require that the
    // special prime is of any particular size, so ensuring this is the case is left
    // for the user to do.
    // */
}

void example_rotation_ckks()
{
    print_example_banner("Example: Rotation / Rotation in CKKS");

    /*
    Rotations in the CKKS scheme work very similarly to rotations in BFV.
    */
    EncryptionParameters parms(scheme_type::ckks);

    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 40, 40, 40, 40, 40 }));

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

    CKKSEncoder ckks_encoder(context);

    size_t slot_count = ckks_encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;
    vector<double> input;
    input.reserve(slot_count);
    double curr_point = 0;
    double step_size = 1.0 / (static_cast<double>(slot_count) - 1);
    for (size_t i = 0; i < slot_count; i++, curr_point += step_size)
    {
        input.push_back(curr_point);
    }
    cout << "Input vector:" << endl;
    print_vector(input, 3, 7);

    auto scale = pow(2.0, 50);

    print_line(__LINE__);
    cout << "Encode and encrypt." << endl;
    Plaintext plain;
    ckks_encoder.encode(input, scale, plain);
    Ciphertext encrypted;
    encryptor.encrypt(plain, encrypted);

    Ciphertext rotated;
    print_line(__LINE__);
    cout << "Rotate 2 steps left." << endl;
    evaluator.rotate_vector(encrypted, 2, galois_keys, rotated);
    cout << "    + Decrypt and decode ...... Correct." << endl;
    decryptor.decrypt(rotated, plain);
    vector<double> result;
    ckks_encoder.decode(plain, result);
    print_vector(result, 3, 7);

    /*
    With the CKKS scheme it is also possible to evaluate a complex conjugation on
    a vector of encrypted complex numbers, using Evaluator::complex_conjugate.
    This is in fact a kind of rotation, and requires also Galois keys.
    */
}

void example_rotation()
{
    print_example_banner("Example: Rotation");

    /*
    Run all rotation examples.
    */
    example_rotation_bfv();
    example_rotation_ckks();
}
