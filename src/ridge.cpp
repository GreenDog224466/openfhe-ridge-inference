#include "openfhe.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono> // Added for Latency Timer

using namespace lbcrypto;

// --- Configuration ---
const uint32_t MULT_DEPTH = 2;
const uint32_t SCALE_MOD_SIZE = 50;
const uint32_t BATCH_SIZE = 16; 
const std::string DATA_DIR = "../cpp_inputs/";

// Model Dimensions (Replacing Magic Numbers)
const size_t NUM_FEATURES = 10;
const size_t NUM_SAMPLES = 4096;

// --- Helper: Read Binary Data ---
std::vector<double> readBinary(const std::string& filename, size_t expected_elements) {
    std::string filepath = DATA_DIR + filename;
    std::ifstream infile(filepath, std::ios::binary | std::ios::ate);
    
    if (!infile) {
        std::cerr << "FATAL: Could not open " << filepath << std::endl;
        exit(1);
    }
    
    std::streamsize size_bytes = infile.tellg();
    
    // FIX: Cast size_bytes to size_t to safely compare and eliminate the compiler warning
    if (static_cast<size_t>(size_bytes) != expected_elements * sizeof(double)) {
        std::cerr << "FATAL: Size mismatch " << filename << ". Expected " 
                  << expected_elements * sizeof(double) << ", got " << size_bytes << std::endl;
        exit(1);
    }
    
    infile.seekg(0, std::ios::beg);
    std::vector<double> buffer(expected_elements);
    infile.read(reinterpret_cast<char*>(buffer.data()), size_bytes);
    return buffer;
}

int main() {
    std::cout << "--- Ridge Regression Encrypted Inference (Profiled) ---" << std::endl;

    // 1. Setup
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(MULT_DEPTH);
    parameters.SetScalingModSize(SCALE_MOD_SIZE);
    parameters.SetBatchSize(BATCH_SIZE);
    parameters.SetSecurityLevel(HEStd_128_classic);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    std::cout << "CKKS Scheme Initialized." << std::endl;

    // 2. Keys
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    // 3. Load Data
    auto w_lin = readBinary("weights_linear.bin", NUM_FEATURES);
    auto w_quad = readBinary("weights_quad.bin", NUM_FEATURES);
    auto bias_vec = readBinary("bias.bin", 1); // Load Bias
    double bias = bias_vec[0];
    
    // Load full test set but slice one sample row for this test
    auto all_x = readBinary("x_test.bin", NUM_SAMPLES * NUM_FEATURES); 
    std::vector<double> x_sample(all_x.begin(), all_x.begin() + NUM_FEATURES);

    std::cout << "Data Loaded." << std::endl;

    // --- TIMER START ---
    auto start_time = std::chrono::high_resolution_clock::now();

    // 4. Encrypt
    Plaintext pt_x = cc->MakeCKKSPackedPlaintext(x_sample);
    auto ct_x = cc->Encrypt(keys.publicKey, pt_x);

    // 5. Inference
    // Term 2: x^2 (Level 0 -> 1 -> 2)
    auto ct_x_sq = cc->EvalMult(ct_x, ct_x);
    ct_x_sq = cc->Rescale(ct_x_sq);
    
    Plaintext pt_w_quad = cc->MakeCKKSPackedPlaintext(w_quad);
    auto ct_term_quad = cc->EvalMult(ct_x_sq, pt_w_quad);
    ct_term_quad = cc->Rescale(ct_term_quad);

    // Term 1: x (Level 0 -> 1)
    Plaintext pt_w_lin = cc->MakeCKKSPackedPlaintext(w_lin);
    auto ct_term_lin = cc->EvalMult(ct_x, pt_w_lin);
    ct_term_lin = cc->Rescale(ct_term_lin);

    // Alignment (L1 -> L2)
    if (ct_term_lin->GetLevel() < ct_term_quad->GetLevel()) {
        cc->LevelReduceInPlace(ct_term_lin, nullptr, ct_term_quad->GetLevel() - ct_term_lin->GetLevel());
    }

    // Sum + Bias
    auto ct_result = cc->EvalAdd(ct_term_lin, ct_term_quad);
    ct_result = cc->EvalAdd(ct_result, bias); // Add Bias!

    // 6. Decrypt
    Plaintext pt_result;
    cc->Decrypt(keys.secretKey, ct_result, &pt_result);
    pt_result->SetLength(NUM_FEATURES);

    // --- TIMER END ---
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = end_time - start_time;

    // Output
    auto final_result = pt_result->GetRealPackedValue();
    std::cout << "\n--- Performance Metrics ---" << std::endl;
    std::cout << "Latency: " << ms_double.count() << " ms" << std::endl;

    std::cout << "\n--- Full Verification ---" << std::endl;
    for (size_t i = 0; i < NUM_FEATURES; i++) {
        double expected = (w_lin[i] * x_sample[i]) + (w_quad[i] * x_sample[i] * x_sample[i]) + bias;
        std::cout << "Feat " << i << ": HE=" << final_result[i] << " | Ref=" << expected << std::endl;
    }

    return 0;
}