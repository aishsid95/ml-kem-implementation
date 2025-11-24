"""
ML-KEM (Kyber) Educational Implementation
A simplified educational tool for demonstrating post-quantum key encapsulation

This implementation focuses on clarity and educational value rather than 
production security or optimization.

Author: Aisha Siddiqa
Course: CI7100 Cryptography and Applications
Institution: Kingston University
"""

import hashlib
import secrets
# import time
from typing import Tuple, List
# import json


class MLKEMParameters:
    """Parameter sets for ML-KEM (based on NIST standardization)"""
    
    # ML-KEM-512 parameters (simplified for education)
    ML_KEM_512 = {
        'name': 'ML-KEM-512',
        'n': 256,           # polynomial degree
        'q': 3329,          # modulus
        'k': 2,             # module rank (number of polynomials in vector)
        'eta1': 3,          # noise distribution parameter for key generation
        'eta2': 2,          # noise distribution parameter for encryption
        'du': 10,           # compression parameter for u
        'dv': 4,            # compression parameter for v
        'security_level': 1  # NIST security level
    }
    
    # ML-KEM-768 parameters
    ML_KEM_768 = {
        'name': 'ML-KEM-768',
        'n': 256,
        'q': 3329,
        'k': 3,
        'eta1': 2,
        'eta2': 2,
        'du': 10,
        'dv': 4,
        'security_level': 3
    }
    
    # ML-KEM-1024 parameters
    ML_KEM_1024 = {
        'name': 'ML-KEM-1024',
        'n': 256,
        'q': 3329,
        'k': 4,
        'eta1': 2,
        'eta2': 2,
        'du': 11,
        'dv': 5,
        'security_level': 5
    }


class Polynomial:
    """Represents a polynomial in R_q = Z_q[X]/(X^n + 1)"""
    
    def __init__(self, coefficients: List[int], q: int, n: int):
        """
        Initialize polynomial
        
        Args:
            coefficients: List of coefficients (will be padded/truncated to n)
            q: Modulus
            n: Degree (polynomial is degree n-1)
        """
        self.q = q
        self.n = n
        # Ensure we have exactly n coefficients, padded with zeros if needed
        self.coeffs = (coefficients + [0] * n)[:n]
        # Reduce modulo q
        self.coeffs = [c % q for c in self.coeffs]
    
    def __add__(self, other):
        """Add two polynomials"""
        new_coeffs = [(a + b) % self.q for a, b in zip(self.coeffs, other.coeffs)]
        return Polynomial(new_coeffs, self.q, self.n)
    
    def __sub__(self, other):
        """Subtract two polynomials"""
        new_coeffs = [(a - b) % self.q for a, b in zip(self.coeffs, other.coeffs)]
        return Polynomial(new_coeffs, self.q, self.n)
    
    def __mul__(self, other):
        """
        Multiply two polynomials in R_q = Z_q[X]/(X^n + 1)
        Uses schoolbook multiplication with reduction
        """
        # Initialize result
        result = [0] * (2 * self.n - 1)
        
        # Schoolbook multiplication
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] = (result[i + j] + a * b) % self.q
        
        # Reduce by X^n + 1
        # If we have X^n terms or higher, replace X^n with -1
        reduced = [0] * self.n
        for i, coeff in enumerate(result):
            if i < self.n:
                reduced[i] = (reduced[i] + coeff) % self.q
            else:
                # X^n = -1, so X^(n+k) = -X^k
                reduced[i - self.n] = (reduced[i - self.n] - coeff) % self.q
        
        return Polynomial(reduced, self.q, self.n)
    
    def compress(self, d: int) -> 'Polynomial':
        """
        Compress polynomial coefficients
        Maps Z_q to Z_{2^d} using rounding
        
        Args:
            d: Number of bits to compress to
        """
        compressed = []
        for c in self.coeffs:
            # Round(2^d / q * c) mod 2^d
            compressed_val = round((2**d * c) / self.q) % (2**d)
            compressed.append(compressed_val)
        return Polynomial(compressed, 2**d, self.n)
    
    def decompress(self, d: int, q: int) -> 'Polynomial':
        """
        Decompress polynomial coefficients
        Maps Z_{2^d} back to Z_q
        
        Args:
            d: Number of bits compressed to
            q: Original modulus
        """
        decompressed = []
        for c in self.coeffs:
            # Round(q / 2^d * c)
            decompressed_val = round((q * c) / (2**d))
            decompressed.append(decompressed_val)
        return Polynomial(decompressed, q, self.n)
    
    def to_bytes(self) -> bytes:
        """Convert polynomial to bytes (simple encoding)"""
        # Simple encoding: 2 bytes per coefficient (assumes q < 65536)
        result = bytearray()
        for c in self.coeffs:
            result.extend(c.to_bytes(2, 'little'))
        return bytes(result)
    
    @classmethod
    def from_bytes(cls, data: bytes, q: int, n: int) -> 'Polynomial':
        """Create polynomial from bytes"""
        coeffs = []
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                c = int.from_bytes(data[i:i+2], 'little')
                coeffs.append(c)
        return cls(coeffs, q, n)
    
    def __str__(self):
        """String representation (show first few coefficients)"""
        if len(self.coeffs) <= 8:
            return f"Poly({self.coeffs})"
        return f"Poly([{', '.join(map(str, self.coeffs[:4]))}, ..., {', '.join(map(str, self.coeffs[-2:]))}])"


class PolynomialVector:
    """Vector of polynomials (used in module lattices)"""
    
    def __init__(self, polynomials: List[Polynomial]):
        self.polys = polynomials
        self.k = len(polynomials)
        self.q = polynomials[0].q if polynomials else 0
        self.n = polynomials[0].n if polynomials else 0
    
    def __add__(self, other):
        """Add two polynomial vectors"""
        new_polys = [p1 + p2 for p1, p2 in zip(self.polys, other.polys)]
        return PolynomialVector(new_polys)
    
    def __sub__(self, other):
        """Subtract two polynomial vectors"""
        new_polys = [p1 - p2 for p1, p2 in zip(self.polys, other.polys)]
        return PolynomialVector(new_polys)
    
    def dot(self, other) -> Polynomial:
        """Dot product of two polynomial vectors"""
        result = Polynomial([0] * self.n, self.q, self.n)
        for p1, p2 in zip(self.polys, other.polys):
            result = result + (p1 * p2)
        return result
    
    def compress(self, d: int) -> 'PolynomialVector':
        """Compress all polynomials in vector"""
        return PolynomialVector([p.compress(d) for p in self.polys])
    
    def decompress(self, d: int, q: int) -> 'PolynomialVector':
        """Decompress all polynomials in vector"""
        return PolynomialVector([p.decompress(d, q) for p in self.polys])
    
    def to_bytes(self) -> bytes:
        """Convert vector to bytes"""
        return b''.join(p.to_bytes() for p in self.polys)


class MLKEM:
    """
    ML-KEM (Module Learning with Errors Key Encapsulation Mechanism)
    Educational implementation based on CRYSTALS-Kyber/ML-KEM
    """
    
    def __init__(self, parameter_set: dict):
        """
        Initialize ML-KEM with specific parameter set
        
        Args:
            parameter_set: Dictionary containing n, q, k, eta1, eta2, du, dv
        """
        self.params = parameter_set
        self.n = parameter_set['n']
        self.q = parameter_set['q']
        self.k = parameter_set['k']
        self.eta1 = parameter_set['eta1']
        self.eta2 = parameter_set['eta2']
        self.du = parameter_set['du']
        self.dv = parameter_set['dv']
    
    def _centered_binomial_distribution(self, eta: int, seed: bytes, counter: int) -> Polynomial:
        """
        Sample from centered binomial distribution
        Used to generate small error polynomials
        
        Args:
            eta: Distribution parameter
            seed: Random seed
            counter: Counter for domain separation
        """
        # Generate pseudo-random bytes using SHA3-256
        hasher = hashlib.sha3_256()
        hasher.update(seed)
        hasher.update(counter.to_bytes(1, 'little'))
        random_bytes = hasher.digest()
        
        coeffs = []
        for i in range(self.n):
            # Sample 2*eta bits from random_bytes
            byte_idx = (i * eta) // 4
            if byte_idx < len(random_bytes):
                random_val = random_bytes[byte_idx]
            else:
                # Need more randomness
                hasher = hashlib.sha3_256()
                hasher.update(seed)
                hasher.update((counter + byte_idx).to_bytes(2, 'little'))
                random_bytes = hasher.digest()
                random_val = random_bytes[0]
            
            # Centered binomial: sum of eta bits minus sum of eta other bits
            a = sum((random_val >> j) & 1 for j in range(eta))
            b = sum((random_val >> (j + eta)) & 1 for j in range(eta))
            coeffs.append(a - b)
        
        return Polynomial(coeffs, self.q, self.n)
    
    def _sample_ntt_polynomial(self, seed: bytes, counter: int) -> Polynomial:
        """
        Sample a uniform polynomial from seed
        (Simplified version - in real ML-KEM this uses rejection sampling)
        """
        hasher = hashlib.sha3_256()
        hasher.update(seed)
        hasher.update(counter.to_bytes(2, 'little'))
        random_bytes = hasher.digest()
        
        # Extend random bytes if needed
        while len(random_bytes) < self.n * 2:
            hasher = hashlib.sha3_256()
            hasher.update(random_bytes)
            random_bytes += hasher.digest()
        
        coeffs = []
        for i in range(self.n):
            # Take 2 bytes and reduce modulo q
            val = int.from_bytes(random_bytes[i*2:(i*2)+2], 'little') % self.q
            coeffs.append(val)
        
        return Polynomial(coeffs, self.q, self.n)
    
    def key_generation(self, seed: bytes = None) -> Tuple[bytes, bytes]:
        """
        Generate ML-KEM key pair
        
        Returns:
            (public_key, secret_key): Tuple of serialized keys
        """
        print(f"\n{'='*60}")
        print(f"ML-KEM Key Generation ({self.params['name']})")
        print(f"{'='*60}")
        
        # Generate random seed if not provided
        if seed is None:
            seed = secrets.token_bytes(32)
        
        print(f"Random seed generated: {seed.hex()[:32]}...")
        
        # Generate matrix A (public randomness)
        print(f"\nGenerating public matrix A ({self.k}x{self.k} polynomials)...")
        A = []
        for i in range(self.k):
            row = []
            for j in range(self.k):
                poly = self._sample_ntt_polynomial(seed, i * self.k + j)
                row.append(poly)
            A.append(row)
        print(f"Matrix A generated with {self.k * self.k} polynomials")
        
        # Generate secret vector s (small coefficients)
        print(f"\nGenerating secret vector s ({self.k} polynomials)...")
        s_polys = []
        for i in range(self.k):
            s_poly = self._centered_binomial_distribution(self.eta1, seed, 100 + i)
            s_polys.append(s_poly)
            print(f"  s[{i}]: {s_poly}")
        s = PolynomialVector(s_polys)
        
        # Generate error vector e (small coefficients)
        print(f"\nGenerating error vector e ({self.k} polynomials)...")
        e_polys = []
        for i in range(self.k):
            e_poly = self._centered_binomial_distribution(self.eta1, seed, 200 + i)
            e_polys.append(e_poly)
            print(f"  e[{i}]: {e_poly}")
        e = PolynomialVector(e_polys)
        
        # Compute public key: t = A*s + e
        print(f"\nComputing public key t = A·s + e...")
        t_polys = []
        for i in range(self.k):
            # Compute A[i] · s (dot product of row i with vector s)
            As_i = Polynomial([0] * self.n, self.q, self.n)
            for j in range(self.k):
                As_i = As_i + (A[i][j] * s.polys[j])
            # Add error
            t_i = As_i + e.polys[i]
            t_polys.append(t_i)
            print(f"  t[{i}] = (A[{i}]·s) + e[{i}]")
        t = PolynomialVector(t_polys)
        
        # Serialize keys
        pk = seed + t.to_bytes()
        sk = s.to_bytes()
        
        print(f"\n{'='*60}")
        print(f"Key Generation Complete!")
        print(f"Public key size: {len(pk)} bytes")
        print(f"Secret key size: {len(sk)} bytes")
        print(f"{'='*60}\n")
        
        return pk, sk
    
    def encapsulate(self, public_key: bytes, user_message=None) -> Tuple[bytes, bytes]:
        """
        Encapsulate a shared secret using public key
        
        Args:
            public_key: Recipient's public key
            user_message: Optional user-provided message for demonstration
            
        Returns:
            (ciphertext, shared_secret): Tuple of ciphertext and shared secret
        """

        if user_message:
            m = hashlib.sha256(user_message.encode()).digest()
        else:
            m = secrets.token_bytes(32)
        
        print(f"\n{'='*60}")
        print(f"ML-KEM Encapsulation ({self.params['name']})")
        print(f"{'='*60}")
        
        # Extract A seed and t from public key
        seed = public_key[:32]
        t_bytes = public_key[32:]
        
        # Reconstruct A
        print(f"Reconstructing public matrix A from seed...")
        A = []
        for i in range(self.k):
            row = []
            for j in range(self.k):
                poly = self._sample_ntt_polynomial(seed, i * self.k + j)
                row.append(poly)
            A.append(row)
        
        # Generate random message
        # m = secrets.token_bytes(32)
        # print(f"Random message m: {m.hex()[:32]}...")
        
        # Generate randomness for encryption
        enc_seed = hashlib.sha3_256(m).digest()
        
        # Sample r, e1, e2 (small error polynomials)
        print(f"\nGenerating encryption randomness...")
        r_polys = [self._centered_binomial_distribution(self.eta1, enc_seed, i) 
                   for i in range(self.k)]
        r = PolynomialVector(r_polys)
        print(f"Generated r vector ({self.k} polynomials)")
        
        e1_polys = [self._centered_binomial_distribution(self.eta2, enc_seed, 100 + i) 
                    for i in range(self.k)]
        e1 = PolynomialVector(e1_polys)
        print(f"Generated e1 vector ({self.k} polynomials)")
        
        e2 = self._centered_binomial_distribution(self.eta2, enc_seed, 200)
        print(f"Generated e2 polynomial")
        
        # Compute u = A^T * r + e1
        print(f"\nComputing u = A^T·r + e1...")
        u_polys = []
        for j in range(self.k):
            # Compute A^T[j] · r (column j of A times r)
            Atr_j = Polynomial([0] * self.n, self.q, self.n)
            for i in range(self.k):
                Atr_j = Atr_j + (A[i][j] * r.polys[i])
            u_j = Atr_j + e1.polys[j]
            u_polys.append(u_j)
        u = PolynomialVector(u_polys)
        
        # Encode message as polynomial
        m_poly_coeffs = []
        for byte in m:
            for bit in range(8):
                # Encode bit as 0 or q//2
                m_poly_coeffs.append(((byte >> bit) & 1) * (self.q // 2))
                if len(m_poly_coeffs) >= self.n:
                    break
            if len(m_poly_coeffs) >= self.n:
                break
        m_poly = Polynomial(m_poly_coeffs, self.q, self.n)
        
        # Reconstruct t from public key bytes
        t_polys = []
        poly_size = self.n * 2  # 2 bytes per coefficient
        for i in range(self.k):
            start = i * poly_size
            end = start + poly_size
            if end <= len(t_bytes):
                t_poly = Polynomial.from_bytes(t_bytes[start:end], self.q, self.n)
                t_polys.append(t_poly)
        t = PolynomialVector(t_polys)
        
        # Compute v = t^T * r + e2 + m_poly
        print(f"Computing v = t^T·r + e2 + Encode(m)...")
        v = t.dot(r) + e2 + m_poly
        
        # Compress and serialize ciphertext
        print(f"\nCompressing ciphertext...")
        u_compressed = u.compress(self.du)
        v_compressed = v.compress(self.dv)
        
        ciphertext = u_compressed.to_bytes() + v_compressed.to_bytes()
        
        # Derive shared secret from message
        # shared_secret = hashlib.sha3_256(m).digest()
        K = hashlib.sha3_256(m + hashlib.sha3_256(ciphertext).digest()).digest()
        
        print(f"\n{'='*60}")
        print(f"Encapsulation Complete!")
        print(f"Ciphertext size: {len(ciphertext)} bytes")
        print(f"Shared secret: {K.hex()[:32]}...")
        print(f"{'='*60}\n")
        
        return ciphertext, K 
        # return ciphertext, shared_secret
    
    def decapsulate(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """
        Decapsulate shared secret using secret key
        
        Args:
            ciphertext: Received ciphertext
            secret_key: Recipient's secret key
            
        Returns:
            shared_secret: Decapsulated shared secret
        """
        print(f"\n{'='*60}")
        print(f"ML-KEM Decapsulation ({self.params['name']})")
        print(f"{'='*60}")
        
        # Parse ciphertext
        u_size = self.k * self.n * 2  # Approximate
        u_bytes = ciphertext[:u_size]
        v_bytes = ciphertext[u_size:]
        
        print(f"Parsing ciphertext (u: {len(u_bytes)} bytes, v: {len(v_bytes)} bytes)...")
        
        # Decompress u and v
        u_polys = []
        poly_size = self.n * 2
        for i in range(self.k):
            start = i * poly_size
            end = start + poly_size
            if end <= len(u_bytes):
                u_poly_compressed = Polynomial.from_bytes(u_bytes[start:end], 2**self.du, self.n)
                u_poly = u_poly_compressed.decompress(self.du, self.q)
                u_polys.append(u_poly)
        u = PolynomialVector(u_polys)
        
        v_compressed = Polynomial.from_bytes(v_bytes, 2**self.dv, self.n)
        v = v_compressed.decompress(self.dv, self.q)
        
        # Reconstruct secret key
        s_polys = []
        for i in range(self.k):
            start = i * poly_size
            end = start + poly_size
            if end <= len(secret_key):
                s_poly = Polynomial.from_bytes(secret_key[start:end], self.q, self.n)
                s_polys.append(s_poly)
        s = PolynomialVector(s_polys)
        
        # Compute m_poly = v - s^T * u
        print(f"Computing m = v - s^T·u...")
        m_poly = v - s.dot(u)
        
        # Decode message from polynomial
        m_bytes = bytearray()
        for i in range(0, min(256, len(m_poly.coeffs)), 8):
            byte_val = 0
            for bit in range(8):
                if i + bit < len(m_poly.coeffs):
                    # Decode: closer to q/2 means 1, closer to 0 means 0
                    coeff = m_poly.coeffs[i + bit] % self.q
                    bit_val = 1 if abs(coeff - self.q // 2) < self.q // 4 else 0
                    byte_val |= (bit_val << bit)
            m_bytes.append(byte_val)
        
        m = bytes(m_bytes[:32])
        
        # Derive shared secret
        # shared_secret = hashlib.sha3_256(m).digest()
        K = hashlib.sha3_256(m + hashlib.sha3_256(ciphertext).digest()).digest()
        
        print(f"\n{'='*60}")
        print(f"Decapsulation Complete!")
        print(f"Recovered message: {m.hex()[:32]}...")
        print(f"Shared secret: {K.hex()[:32]}...")
        print(f"{'='*60}\n")
        
        return K
