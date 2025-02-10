#pragma once
#define USE_SIMD_M

#include <iostream>
#include <vector>
#include <immintrin.h>
#include "vec4.h"
using namespace std;

// Matrix class for 4x4 transformation matrices
class alignas(16) matrix {
    union {
        float m[4][4]; // 2D array representation of the matrix
        float a[16];   // 1D array representation of the matrix for linear access
        __m128 row[4];  // store row of matrix by 4 SIMD vectors
    };

public:
    // Default constructor initializes the matrix as an identity matrix
    matrix() {
        identity();
    }

    // Access matrix elements by row and column
    float& operator()(unsigned int row, unsigned int col) { return m[row][col]; }

    // Display the matrix elements in a readable format
    void display() {
        for (unsigned int i = 0; i < 4; i++) {
            for (unsigned int j = 0; j < 4; j++)
                std::cout << m[i][j] << '\t';
            std::cout << std::endl;
        }
    }

    // Multiply the matrix by a 4D vector
    // Input Variables:
    // - v: vec4 object to multiply with the matrix
    // Returns the resulting transformed vec4
    vec4 operator * (const vec4& v) const {
    vec4 result;
    #ifdef USE_SIMD_M
        // load vector4
        __m128 vec = _mm_loadu_ps(v.getV()); 

        // load 4 rows of matrix in register
        __m128 row0 = _mm_loadu_ps(&a[0]);
        __m128 row1 = _mm_loadu_ps(&a[4]);
        __m128 row2 = _mm_loadu_ps(&a[8]);
        __m128 row3 = _mm_loadu_ps(&a[12]);
        // do dot product of each row with vector
        __m128 res0 = _mm_dp_ps(row0, vec, 0xF1);
        __m128 res1 = _mm_dp_ps(row1, vec, 0xF1);
        __m128 res2 = _mm_dp_ps(row2, vec, 0xF1);
        __m128 res3 = _mm_dp_ps(row3, vec, 0xF1);
        // store result to Vec4
        _mm_store_ps(result.v, _mm_set_ps(_mm_cvtss_f32(res3), _mm_cvtss_f32(res2), _mm_cvtss_f32(res1),_mm_cvtss_f32(res0)));
    #else
        result[0] = a[0] * v[0] + a[1] * v[1] + a[2] * v[2] + a[3] * v[3];
        result[1] = a[4] * v[0] + a[5] * v[1] + a[6] * v[2] + a[7] * v[3];
        result[2] = a[8] * v[0] + a[9] * v[1] + a[10] * v[2] + a[11] * v[3];
        result[3] = a[12] * v[0] + a[13] * v[1] + a[14] * v[2] + a[15] * v[3];
        
    #endif
        return result;
    }



    // Multiply the matrix by another matrix
    // Input Variables:
    // - mx: Another matrix to multiply with
    // Returns the resulting matrix
    matrix operator * (const matrix& mx) const {
        matrix ret;
    #ifdef USE_SIMD_M   
        alignas(16) float row_values[4];

        for (int i = 0; i < 4; i++) {
            _mm_store_ps(row_values, row[i]); // store i_th row of matrix
            // dot product
            ret.row[i] = _mm_add_ps(
                _mm_add_ps(
                    _mm_mul_ps(_mm_set1_ps(row_values[0]), mx.row[0]), // [i][0]*mx[0][j]
                    _mm_mul_ps(_mm_set1_ps(row_values[1]), mx.row[1]) // [i][1]*mx[1][j]
                ),
                _mm_add_ps(
                    _mm_mul_ps(_mm_set1_ps(row_values[2]), mx.row[2]), // [i][2]*mx[2][j]
                    _mm_mul_ps(_mm_set1_ps(row_values[3]), mx.row[3]) // [i][3]*mx[3][j]
                )
            );
        }
        
    #else
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                ret.a[row * 4 + col] =
                    a[row * 4 + 0] * mx.a[0 * 4 + col] +
                    a[row * 4 + 1] * mx.a[1 * 4 + col] +
                    a[row * 4 + 2] * mx.a[2 * 4 + col] +
                    a[row * 4 + 3] * mx.a[3 * 4 + col];
            }
        }

    #endif
        return ret;
    }


    // Create a perspective projection matrix
    // Input Variables:
    // - fov: Field of view in radians
    // - aspect: Aspect ratio of the viewport
    // - n: Near clipping plane
    // - f: Far clipping plane
    // Returns the perspective matrix
    static matrix makePerspective(float fov, float aspect, float n, float f) {
        matrix m;
        m.zero();
        float tanHalfFov = std::tan(fov / 2.0f);

        m.a[0] = 1.0f / (aspect * tanHalfFov);
        m.a[5] = 1.0f / tanHalfFov;
        m.a[10] = -f / (f - n);
        m.a[11] = -(f * n) / (f - n);
        m.a[14] = -1.0f;
        return m;
    }

    // Create a translation matrix
    // Input Variables:
    // - tx, ty, tz: Translation amounts along the X, Y, and Z axes
    // Returns the translation matrix
    static matrix makeTranslation(float tx, float ty, float tz) {
        matrix m;
        m.identity();
        m.a[3] = tx;
        m.a[7] = ty;
        m.a[11] = tz;
        return m;
    }

    // Create a rotation matrix around the Z-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    static matrix makeRotateZ(float aRad) {
        matrix m;
        m.identity();
        m.a[0] = std::cos(aRad);
        m.a[1] = -std::sin(aRad);
        m.a[4] = std::sin(aRad);
        m.a[5] = std::cos(aRad);
        return m;
    }

    // Create a rotation matrix around the X-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    static matrix makeRotateX(float aRad) {
        matrix m;
        m.identity();
        m.a[5] = std::cos(aRad);
        m.a[6] = -std::sin(aRad);
        m.a[9] = std::sin(aRad);
        m.a[10] = std::cos(aRad);
        return m;
    }

    // Create a rotation matrix around the Y-axis
    // Input Variables:
    // - aRad: Rotation angle in radians
    // Returns the rotation matrix
    static matrix makeRotateY(float aRad) {
        matrix m;
        m.identity();
        m.a[0] = std::cos(aRad);
        m.a[2] = std::sin(aRad);
        m.a[8] = -std::sin(aRad);
        m.a[10] = std::cos(aRad);
        return m;
    }

    // Create a composite rotation matrix from X, Y, and Z rotations
    // Input Variables:
    // - x, y, z: Rotation angles in radians around each axis
    // Returns the composite rotation matrix
    static matrix makeRotateXYZ(float x, float y, float z) {
        return matrix::makeRotateX(x) * matrix::makeRotateY(y) * matrix::makeRotateZ(z);
    }

    // Create a scaling matrix
    // Input Variables:
    // - s: Scaling factor
    // Returns the scaling matrix
    static matrix makeScale(float s) {
        matrix m;
        s = max(s, 0.01f); // Ensure scaling factor is not too small
        m.identity();
        m.a[0] = s;
        m.a[5] = s;
        m.a[10] = s;
        return m;
    }

    // Create an identity matrix
    // Returns an identity matrix
    static matrix makeIdentity() {
        matrix m;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m.m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        return m;
    }

private:
    // Set all elements of the matrix to 0
    void zero() {
        for (unsigned int i = 0; i < 16; i++)
            a[i] = 0.f;
    }

    // Set the matrix as an identity matrix
    void identity() {
    #ifdef USE_SIMD_M  
        __m128 zero = _mm_setzero_ps();   

        _mm_storeu_ps(&a[0], zero); 
        _mm_storeu_ps(&a[4], zero); 
        _mm_storeu_ps(&a[8], zero); 
        _mm_storeu_ps(&a[12], zero); 

        a[0] = 1.0f;
        a[5] = 1.0f;
        a[10] = 1.0f;
        a[15] = 1.0f;
    #else
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                 m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
    #endif
    }

};


