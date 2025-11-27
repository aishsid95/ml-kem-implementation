# ML-KEM-512 Educational Implementation

An educational implementation of ML-KEM-512 (Module-Lattice-based Key Encapsulation Mechanism), demonstrating post-quantum cryptography concepts with clarity and transparency.

## 🔐 Overview

ML-KEM (Module Learning with Errors Key Encapsulation Mechanism) is a post-quantum cryptographic algorithm standardized by NIST as FIPS 203. This implementation focuses on **educational transparency** rather than production optimization, making it ideal for:

- **Learning** lattice-based cryptography concepts
- **Understanding** the Module-LWE hard problem
- **Demonstrating** quantum-resistant key exchange
- **Exploring** NIST PQC standardization

### Why ML-KEM?

- **Quantum Resistance**: Based on the Module-LWE problem, believed to be resistant to quantum attacks
- **NIST Standardized**: Selected as the primary post-quantum KEM in FIPS 203
- **Practical Applications**: Used in TLS 1.3, VPNs, IoT devices, and enterprise communications

## ✨ Features

### Core Implementation
- ✅ **Complete ML-KEM-512 Protocol**
  - Key Generation (`KeyGen`)
  - Encapsulation (`Encaps`)
  - Decapsulation (`Decaps`)

- ✅ **Educational Transparency**
  - Schoolbook polynomial multiplication (no NTT optimization)
  - Step-by-step algorithm visualization
  - Detailed logging of cryptographic operations
  
- ✅ **Interactive Demo**
  - Alice-Bob key exchange simulation
  - Custom message encapsulation
  - Color-coded terminal output

### Technical Features
- Polynomial ring arithmetic in R_q = Z_q[X]/(X^n + 1)
- Centered binomial distribution (CBD) for noise sampling
- Compression/decompression for bandwidth efficiency
- SHA3-256 based pseudorandom functions
- Fujisaki-Okamoto transform for IND-CCA2 security

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone Repository
```bash
git clone https://github.com/aishsid95/ml-kem-implementation.git
cd ml-kem-implementation
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

No external cryptographic libraries required - uses Python standard library only!

## 🎯 Quick Start

### Basic Usage

```python
from mlkem import MLKEM, MLKEMParameters

# Initialize ML-KEM-512
mlkem = MLKEM(MLKEMParameters.ML_KEM_512)

# Alice generates key pair
public_key, secret_key = mlkem.key_generation()

# Bob encapsulates a shared secret
ciphertext, shared_secret_bob = mlkem.encapsulate(public_key)

# Alice decapsulates to recover the shared secret
shared_secret_alice = mlkem.decapsulate(ciphertext, secret_key)

# Verify both parties have the same secret
assert shared_secret_alice == shared_secret_bob
print("✓ Key exchange successful!")
```

### Interactive Demo

```bash
python ml-kem-user-input.py
```

Choose between:
1. **Standard mode**: Random message generation (authentic ML-KEM)
2. **Educational mode**: Custom message input for learning

Follow the step-by-step Alice → Bob → Alice protocol visualization.

## 📁 Project Structure

```
ml-kem-implementation/
├── mlkem.py                    # Core ML-KEM implementation
│   ├── MLKEMParameters         # Parameter sets (512/768/1024)
│   ├── Polynomial              # Ring arithmetic operations
│   ├── PolynomialVector        # Module lattice operations
│   └── MLKEM                   # Main protocol class
│
├── ml-kem-user-input.py       # Interactive demonstration
│   ├── Alice's perspective     # Key generation
│   ├── Bob's perspective       # Encapsulation
│   └── Alice's perspective     # Decapsulation
│
├── test_mlkem_protocol.py     # Comprehensive test suite
│   ├── Protocol correctness    # End-to-end tests
│   ├── Edge cases              # Boundary conditions
│   └── Parameter validation    # All parameter sets
│
├── README.md                   # This file
├── requirements.txt            # Python dependencies
└── LICENSE                     # MIT License
```

## 💡 Usage Examples

### Parameter Sets

ML-KEM supports three security levels:

```python
# ML-KEM-512: Security Level 1 (128-bit)
mlkem_512 = MLKEM(MLKEMParameters.ML_KEM_512)

# ML-KEM-768: Security Level 3 (192-bit)
mlkem_768 = MLKEM(MLKEMParameters.ML_KEM_768)

# ML-KEM-1024: Security Level 5 (256-bit)
mlkem_1024 = MLKEM(MLKEMParameters.ML_KEM_1024)
```

### Custom Message Encapsulation (Educational)

```python
# Educational feature: specify message for encapsulation
user_message = "secret_key_material"
ciphertext, K = mlkem.encapsulate(public_key, user_message)
```

**Note**: Production ML-KEM always generates random messages internally.

### Accessing Key Sizes

```python
public_key, secret_key = mlkem.key_generation()

print(f"Public key:  {len(public_key)} bytes")    # 800 B for ML-KEM-512
print(f"Secret key:  {len(secret_key)} bytes")    # 1632 B for ML-KEM-512

ciphertext, K = mlkem.encapsulate(public_key)
print(f"Ciphertext:  {len(ciphertext)} bytes")    # 768 B for ML-KEM-512
print(f"Shared key:  {len(K)} bytes")             # 32 B
```

## 🔬 Technical Details

### Cryptographic Parameters (ML-KEM-512)

| Parameter | Value | Description |
|-----------|-------|-------------|
| **n** | 256 | Polynomial degree |
| **q** | 3329 | Modulus (chosen for NTT efficiency) |
| **k** | 2 | Module rank (number of polynomials) |
| **η₁** | 3 | Noise parameter for key generation |
| **η₂** | 2 | Noise parameter for encryption |
| **d_u** | 10 | Compression bits for u |
| **d_v** | 4 | Compression bits for v |

### Security Foundation

ML-KEM's security relies on the **Module Learning with Errors (Module-LWE)** problem:

Given matrix **A** and vector **b = As + e**, recovering the secret vector **s** is computationally hard when **e** is small noise.

### Algorithms

#### 1. Key Generation
```
Input: None
Output: (pk, sk)

1. Generate random seed d ∈ {0,1}^256
2. Expand seed to matrix A ∈ R_q^(k×k)
3. Sample secret s ∈ R_q^k from CBD_η₁
4. Sample error e ∈ R_q^k from CBD_η₁
5. Compute t = As + e
6. Return pk = (ρ, t), sk = s
```

#### 2. Encapsulation
```
Input: pk = (ρ, t)
Output: (c, K)

1. Generate random m ∈ {0,1}^256
2. Sample r, e₁, e₂ from CBD using H(m)
3. Compute u = A^T r + e₁
4. Compute v = t^T r + e₂ + Encode(m)
5. Compress u, v and create c = (u, v)
6. Derive K = H(m || H(c))
7. Return (c, K)
```

#### 3. Decapsulation
```
Input: c = (u, v), sk = s
Output: K

1. Decompress u, v
2. Compute m' = v - s^T u
3. Decode message m from m'
4. Derive K = H(m || H(c))
5. Return K
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_mlkem_protocol.py
```

### Test Coverage
- ✅ Complete protocol execution (KeyGen → Encaps → Decaps)
- ✅ Key exchange correctness verification
- ✅ All three parameter sets (ML-KEM-512/768/1024)
- ✅ Polynomial arithmetic operations
- ✅ Compression/decompression correctness
- ✅ Edge cases and boundary conditions

Expected output:
```
Testing ML-KEM-512...
✓ Key generation successful
✓ Encapsulation successful
✓ Decapsulation successful
✓ Shared secrets match!

All tests passed! ✓
```

## 🎓 Academic Context

This implementation was developed as part of a coursework at Kingston University. 

### Learning Objectives
- Understanding lattice-based cryptography fundamentals
- Exploring post-quantum cryptographic protocols
- Implementing NIST-standardized algorithms
- Analyzing security-performance trade-offs

### Educational Design Choices

**Why Schoolbook Multiplication?**
- Makes polynomial operations visible and understandable
- Easier to debug and verify correctness
- Suitable for educational demonstrations
- Production implementations use NTT for O(n log n) complexity

**Why Verbose Logging?**
- Shows intermediate algorithm steps
- Helps trace cryptographic operations
- Demonstrates how Module-LWE security works
- Enables step-by-step verification

## 📚 References

1. **NIST FIPS 203** - Module-Lattice-Based Key-Encapsulation Mechanism Standard (2024)
   - [https://csrc.nist.gov/pubs/fips/203/final](https://csrc.nist.gov/pubs/fips/203/final)

2. **CRYSTALS-Kyber Specification** - Avanzi et al. (2021)
   - Original algorithm specification (ML-KEM is derived from Kyber)

3. **Regev (2005)** - "On lattices, learning with errors, random linear codes, and cryptography"
   - Foundation of LWE-based cryptography

4. **Shor (1997)** - "Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer"
   - Motivation for post-quantum cryptography


## 🤝 Contributing

This is an educational project. Contributions that improve clarity, fix bugs, or enhance documentation are welcome!


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This is an **educational implementation** created for learning purposes. It prioritizes clarity and understanding over performance and security. 

**DO NOT use this code in production systems or security-critical applications.**

For production cryptography, use professionally audited libraries like liboqs or official NIST reference implementations.

---


---

**Made with 🔐 for the post-quantum future**
