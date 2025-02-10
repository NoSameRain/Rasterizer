#pragma once
//#define USE_SIMD
//#define USE_THREAD_TILE
#include "mesh.h"
#include "colour.h"
#include "renderer.h"
#include "light.h"
#include <iostream>
#include <immintrin.h> 

// Simple support class for a 2D vector
class vec2D {
public:
    float x, y;

    // Default constructor initializes both components to 0
    vec2D() { x = y = 0.f; };

    // Constructor initializes components with given values
    vec2D(float _x, float _y) : x(_x), y(_y) {}

    // Constructor initializes components from a vec4
    vec2D(vec4 v) {
        x = v[0];
        y = v[1];
    }

    // Display the vector components
    void display() { std::cout << x << '\t' << y << std::endl; }

    // Overloaded subtraction operator for vector subtraction
    vec2D operator- (vec2D& v) {
        vec2D q;
        q.x = x - v.x;
        q.y = y - v.y;
        return q;
    }
};

// Class representing a triangle for rendering purposes
class triangle {
    Vertex v[3];       // Vertices of the triangle
    float area;        // Area of the triangle
    colour col[3];     // Colors for each vertex of the triangle
    Mesh* parentMesh;

public:
    
    // Constructor initializes the triangle with three vertices
    // Input Variables:
    // - v1, v2, v3: Vertices defining the triangle
    triangle(const Vertex& v1, const Vertex& v2, const Vertex& v3, Mesh* mesh) {
        v[0] = v1;
        v[1] = v2;
        v[2] = v3;
        parentMesh = mesh;

        // Calculate the 2D area of the triangle
        vec2D e1 = vec2D(v[1].p - v[0].p);
        vec2D e2 = vec2D(v[2].p - v[0].p);
        area = abs(e1.x * e2.y - e1.y * e2.x);
    }

    // Helper function to compute the cross product for barycentric coordinates
    // Input Variables:
    // - v1, v2: Edges defining the vector
    // - p: Point for which coordinates are being calculated
    float getC(vec2D v1, vec2D v2, vec2D p) {
        vec2D e = v2 - v1;
        vec2D q = p - v1;
        return q.y * e.x - q.x * e.y;
    }
    inline __m256 getC_SIMD(__m256 v1x, __m256 v1y, __m256 v2x, __m256 v2y, __m256 px, __m256 py) {
        __m256 ex = _mm256_sub_ps(v2x, v1x); // e_x = v2_x - v1_x
        __m256 ey = _mm256_sub_ps(v2y, v1y); // e_y = v2_y - v1_y
        __m256 qx = _mm256_sub_ps(px, v1x); // q_x = p_x - v1_x
        __m256 qy = _mm256_sub_ps(py, v1y); // q_y = p_y - v1_y

        // 计算 q_y * e_x - q_x * e_y (8 组点同时计算)
        return _mm256_sub_ps(_mm256_mul_ps(qy, ex), _mm256_mul_ps(qx, ey));
    }

    // Compute barycentric coordinates for a given point
    // Input Variables:
    // - p: Point to check within the triangle
    // Output Variables:
    // - alpha, beta, gamma: Barycentric coordinates of the point
    // Returns true if the point is inside the triangle, false otherwise
    bool getCoordinates(vec2D p, float& alpha, float& beta, float& gamma) {
        alpha = getC(vec2D(v[0].p), vec2D(v[1].p), p) / area;
        beta = getC(vec2D(v[1].p), vec2D(v[2].p), p) / area;
        gamma = getC(vec2D(v[2].p), vec2D(v[0].p), p) / area;

        if (alpha < 0.f || beta < 0.f || gamma < 0.f) return false;
        return true;
    }
    bool getCoordinates_SIMD(vec2D p, float& alpha, float& beta, float& gamma) {
        __m256 v0x = _mm256_set1_ps(vec2D(v[0].p).x);
        __m256 v0y = _mm256_set1_ps(vec2D(v[0].p).y);
        __m256 v1x = _mm256_set1_ps(vec2D(v[1].p).x);
        __m256 v1y = _mm256_set1_ps(vec2D(v[1].p).y);
        __m256 v2x = _mm256_set1_ps(vec2D(v[2].p).x);
        __m256 v2y = _mm256_set1_ps(vec2D(v[2].p).y);
        __m256 px = _mm256_set1_ps(p.x);
        __m256 py = _mm256_set1_ps(p.y);
        __m256 area_inv = _mm256_set1_ps(1.0f / area);

        __m256 C1 = getC_SIMD(v0x, v0y, v1x, v1y, px, py);
        __m256 C2 = getC_SIMD(v1x, v1y, v2x, v2y, px, py);

        __m256 alphaSIMD = _mm256_mul_ps(C1, area_inv);
        __m256 betaSIMD = _mm256_mul_ps(C2, area_inv);
        __m256 gammaSIMD = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(alphaSIMD, betaSIMD));

        // 提取计算结果
        float alpha_arr[8], beta_arr[8], gamma_arr[8];
        _mm256_storeu_ps(alpha_arr, alphaSIMD);
        _mm256_storeu_ps(beta_arr, betaSIMD);
        _mm256_storeu_ps(gamma_arr, gammaSIMD);

        alpha = alpha_arr[0];
        beta = beta_arr[0];
        gamma = gamma_arr[0];

        return alpha >= 0.f && beta >= 0.f && gamma >= 0.f;
    }

    // Template function to interpolate values using barycentric coordinates
    // Input Variables:
    // - alpha, beta, gamma: Barycentric coordinates
    // - a1, a2, a3: Values to interpolate
    // Returns the interpolated value
    template <typename T>
    T interpolate(float alpha, float beta, float gamma, T a1, T a2, T a3) {
        return (a1 * alpha) + (a2 * beta) + (a3 * gamma);
    }
    __m256 interpolate_SIMD(__m256 alpha, __m256 beta, __m256 gamma,
        __m256 a1, __m256 a2, __m256 a3) {
        return _mm256_add_ps(_mm256_mul_ps(a1, alpha),
            _mm256_add_ps(_mm256_mul_ps(a2, beta),
                _mm256_mul_ps(a3, gamma)));
    }

    // Draw the triangle on the canvas
    // Input Variables:
    // - renderer: Renderer object for drawing
    // - L: Light object for shading calculations
    // - ka, kd: Ambient and diffuse lighting coefficients

    const int num_threads = 8;
    int TILE_SIZE = 16;

    void drawParallel(Renderer& renderer, Light& L, int startX, int startY, int endX, int endY) {
        vec2D minV, maxV;
       
        getBoundsWindow(renderer.canvas, minV, maxV);

        // Skip very small triangles
        if (area < 1.f) return;
        // Draw triangles in specific region
        if (minV.x <= endX && maxV.x >= startX && minV.y <= endY && maxV.y >= startY) {
            for (int y = startY; y < endY; y++) {
                for (int x = startX; x < endX; x++) {
                    float alpha, beta, gamma;
                    if (getCoordinates(vec2D((float)x, (float)y), alpha, beta, gamma)) {
                        // Interpolate color, depth, and normals
                        colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                        c.clampColour();
                        float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
                        vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
                        normal.normalise();
                        if (renderer.zbuffer(x, y) > depth && depth > 0.01f) {
                            // typical shader begin
                            L.omega_i.normalise();
                            float dot = max(vec4::dot(L.omega_i, normal), 0.0f);
                            colour a = (c * parentMesh->kd) * (L.L * dot + (L.ambient * parentMesh->kd));
                            // typical shader end
                            unsigned char r, g, b;
                            a.toRGB(r, g, b);
                            renderer.canvas.draw(x, y, r, g, b);
                            renderer.zbuffer(x, y) = depth;
                        }
                    }
                }
            }
        } 
    }


    // 打印某个 SIMD 变量
    void print_simd(__m256 value, const char* label) {
        float data[8];
        _mm256_storeu_ps(data, value);
        std::cout << label << ": ";
        for (int i = 0; i < 8; i++) std::cout << data[i] << " ";
        std::cout << std::endl;
    }


#ifdef USE_SIMD
    void draw(Renderer& renderer, Light& L, float ka, float kd) {
        vec2D minV, maxV;
        getBoundsWindow(renderer.canvas, minV, maxV);
        if (area < 1.f) return;  // 小三角形跳过

        __m256 v0x = _mm256_set1_ps(vec2D(v[0].p).x);
        __m256 v0y = _mm256_set1_ps(vec2D(v[0].p).y);
        __m256 v1x = _mm256_set1_ps(vec2D(v[1].p).x);
        __m256 v1y = _mm256_set1_ps(vec2D(v[1].p).y);
        __m256 v2x = _mm256_set1_ps(vec2D(v[2].p).x);
        __m256 v2y = _mm256_set1_ps(vec2D(v[2].p).y);
        __m256 area_inv = _mm256_set1_ps(1.0f / area);

        for (int y = (int)(minV.y); y < (int)ceil(maxV.y); y++) {
            for (int x = (int)(minV.x); x < (int)ceil(maxV.x); x += 8) {  // process 8 pixels per time
                __m256 px = _mm256_setr_ps(x, x + 1, x + 2, x + 3, x + 4, x + 5, x + 6, x + 7);
                __m256 py = _mm256_set1_ps(y);

                // SIMD 计算 α, β, γ
                __m256 alpha = getC_SIMD(v0x, v0y, v1x, v1y, px, py);
                __m256 beta = getC_SIMD(v1x, v1y, v2x, v2y, px, py);
                alpha = _mm256_mul_ps(alpha, area_inv);
                beta = _mm256_mul_ps(beta, area_inv);
                __m256 gamma = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_add_ps(alpha, beta));

                // 过滤 α, β, γ < 0 的像素
                __m256 mask_inside = _mm256_and_ps(
                    _mm256_cmp_ps(alpha, _mm256_set1_ps(0.0f), _CMP_GE_OQ),
                    _mm256_and_ps(
                        _mm256_cmp_ps(beta, _mm256_set1_ps(0.0f), _CMP_GE_OQ),
                        _mm256_cmp_ps(gamma, _mm256_set1_ps(0.0f), _CMP_GE_OQ)
                    )
                );
                int mask_int = _mm256_movemask_ps(mask_inside);
                if (mask_int == 0) continue;  // 🚀 全部不在三角形内部就跳过
                

                // 🚀 计算颜色和深度
                __m256 p0z = _mm256_set1_ps(v[0].p.z);
                __m256 p1z = _mm256_set1_ps(v[1].p.z);
                __m256 p2z = _mm256_set1_ps(v[2].p.z);
                __m256 depth = interpolate_SIMD(beta, gamma, alpha, p0z, p1z, p2z);
                __m256 zbuffer = _mm256_loadu_ps(&renderer.zbuffer(x, y));

                // 过滤深度测试失败的像素
                __m256 mask_depth = _mm256_cmp_ps(depth, zbuffer, _CMP_LT_OQ);
                mask_int &= _mm256_movemask_ps(mask_depth);
                if (mask_int == 0)
                {
                    
                    continue;
                }

                // 计算法线
                __m256 normal_x = interpolate_SIMD(beta, gamma, alpha,
                    _mm256_set1_ps(v[0].normal.x), _mm256_set1_ps(v[1].normal.x), _mm256_set1_ps(v[2].normal.x));
                __m256 normal_y = interpolate_SIMD(beta, gamma, alpha,
                    _mm256_set1_ps(v[0].normal.y), _mm256_set1_ps(v[1].normal.y), _mm256_set1_ps(v[2].normal.y));
                __m256 normal_z = interpolate_SIMD(beta, gamma, alpha,
                    _mm256_set1_ps(v[0].normal.z), _mm256_set1_ps(v[1].normal.z), _mm256_set1_ps(v[2].normal.z));

                // 计算光照
                __m256 light_dir_x = _mm256_set1_ps(L.omega_i.x);
                __m256 light_dir_y = _mm256_set1_ps(L.omega_i.y);
                __m256 light_dir_z = _mm256_set1_ps(L.omega_i.z);
                __m256 dot_product = _mm256_max_ps(_mm256_add_ps(_mm256_mul_ps(light_dir_x, normal_x), _mm256_add_ps( _mm256_mul_ps(light_dir_y, normal_y),_mm256_mul_ps(light_dir_z, normal_z))), _mm256_set1_ps(0.0f));

                // 计算颜色
                __m256 color_r = interpolate_SIMD(beta, gamma, alpha, _mm256_set1_ps(v[0].rgb.r), _mm256_set1_ps(v[1].rgb.r), _mm256_set1_ps(v[2].rgb.r));
                __m256 color_g = interpolate_SIMD(beta, gamma, alpha, _mm256_set1_ps(v[0].rgb.g), _mm256_set1_ps(v[1].rgb.g), _mm256_set1_ps(v[2].rgb.g));
                __m256 color_b = interpolate_SIMD(beta, gamma, alpha, _mm256_set1_ps(v[0].rgb.b), _mm256_set1_ps(v[1].rgb.b), _mm256_set1_ps(v[2].rgb.b));

                __m256 final_r = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(color_r, _mm256_mul_ps(dot_product, _mm256_set1_ps(kd))), _mm256_set1_ps(0.0f)), _mm256_set1_ps(255.0f));
                __m256 final_g = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(color_g, _mm256_mul_ps(dot_product, _mm256_set1_ps(kd))), _mm256_set1_ps(0.0f)), _mm256_set1_ps(255.0f));
                __m256 final_b = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(color_b, _mm256_mul_ps(dot_product, _mm256_set1_ps(kd))), _mm256_set1_ps(0.0f)), _mm256_set1_ps(255.0f));

                // 存储到颜色缓冲区
                float r[8], g[8], b[8], d[8];
                _mm256_storeu_ps(r, final_r);
                _mm256_storeu_ps(g, final_g);
                _mm256_storeu_ps(b, final_b);
                _mm256_storeu_ps(d, depth);

                // 🚀 逐个写入 Z-buffer 和 FrameBuffer

                for (int i = 0; i < 8; i++) {
                    if (mask_int & (1 << i)) {  // 只处理符合条件的像素
                        //if (r[i] == 0 && g[i] == 0 && b[i] == 0) {
                        //    std::cout << "Black pixel detected at " << x + i << ", " << y << std::endl;
                        //}
                        renderer.canvas.draw(x + i, y, (unsigned char)r[i], (unsigned char)g[i], (unsigned char)b[i]);
                        renderer.zbuffer(x + i, y) = d[i];
                    }
                }
            }
        }
    }

#else 
    void draw(Renderer& renderer, Light& L, float ka, float kd) {
        vec2D minV, maxV;

        // Get the screen-space bounds of the triangle
        getBoundsWindow(renderer.canvas, minV, maxV);

        // Skip very small triangles
        if (area < 1.f) return;

        // Iterate over the bounding box and check each pixel
        for (int y = (int)(minV.y); y < (int)ceil(maxV.y); y++) {
            for (int x = (int)(minV.x); x < (int)ceil(maxV.x); x++) {
                float alpha, beta, gamma;

                // Check if the pixel lies inside the triangle
                if (getCoordinates(vec2D((float)x, (float)y), alpha, beta, gamma)) {
                    // Interpolate color, depth, and normals
                    colour c = interpolate(beta, gamma, alpha, v[0].rgb, v[1].rgb, v[2].rgb);
                    c.clampColour();
                    float depth = interpolate(beta, gamma, alpha, v[0].p[2], v[1].p[2], v[2].p[2]);
                    vec4 normal = interpolate(beta, gamma, alpha, v[0].normal, v[1].normal, v[2].normal);
                    normal.normalise();

                    // Perform Z-buffer test and apply shading
                    if (renderer.zbuffer(x, y) > depth && depth > 0.01f) {
                        // typical shader begin
                        L.omega_i.normalise();
                        float dot = max(vec4::dot(L.omega_i, normal), 0.0f);
                        colour a = (c * kd) * (L.L * dot + (L.ambient * kd));//colour a = (c * kd) * (L.L * dot) + (c * ka) * L.ambient;
                        // typical shader end
                        unsigned char r, g, b;
                        a.toRGB(r, g, b);
                        renderer.canvas.draw(x, y, r, g, b);
                        renderer.zbuffer(x, y) = depth;
                    }
                }
            }
        }
    }
#endif

    // Compute the 2D bounds of the triangle
    // Output Variables:
    // - minV, maxV: Minimum and maximum bounds in 2D space
    void getBounds(vec2D& minV, vec2D& maxV) {
        minV = vec2D(v[0].p);
        maxV = vec2D(v[0].p);
        for (unsigned int i = 1; i < 3; i++) {
            minV.x = min(minV.x, v[i].p[0]);
            minV.y = min(minV.y, v[i].p[1]);
            maxV.x = max(maxV.x, v[i].p[0]);
            maxV.y = max(maxV.y, v[i].p[1]);
        }
    }

    // Compute the 2D bounds of the triangle, clipped to the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    // Output Variables:
    // - minV, maxV: Clipped minimum and maximum bounds
    void getBoundsWindow(GamesEngineeringBase::Window& canvas, vec2D& minV, vec2D& maxV) {
        getBounds(minV, maxV);
        minV.x = max(minV.x, 0);
        minV.y = max(minV.y, 0);
        maxV.x = min(maxV.x, canvas.getWidth());
        maxV.y = min(maxV.y, canvas.getHeight());
    }

    // Debugging utility to display the triangle bounds on the canvas
    // Input Variables:
    // - canvas: Reference to the rendering canvas
    void drawBounds(GamesEngineeringBase::Window& canvas) {
        vec2D minV, maxV;
        getBounds(minV, maxV);

        for (int y = (int)minV.y; y < (int)maxV.y; y++) {
            for (int x = (int)minV.x; x < (int)maxV.x; x++) {
                canvas.draw(x, y, 255, 0, 0);
            }
        }
    }

    // Debugging utility to display the coordinates of the triangle vertices
    void display() {
        for (unsigned int i = 0; i < 3; i++) {
            v[i].p.display();
        }
        std::cout << std::endl;
    }
};
