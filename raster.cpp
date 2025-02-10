#include <iostream>
#define _USE_MATH_DEFINES
//#define USE_SIMD
//#define USE_THREAD_TILE
//#define USE_THREAD_TRI
//#define USE_THREAD_MESH
#include <cmath>

#include "GamesEngineeringBase.h" // Include the GamesEngineeringBase header
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <cmath>
#include "matrix.h"
#include "colour.h"
#include "mesh.h"
#include "zbuffer.h"
#include "renderer.h"
#include "RNG.h"
#include "light.h"
#include "triangle.h"

// Main rendering function that processes a mesh, transforms its vertices, applies lighting, and draws triangles on the canvas.
// Input Variables:
// - renderer: The Renderer object used for drawing.
// - mesh: Pointer to the Mesh object containing vertices and triangles to render.
// - camera: Matrix representing the camera's transformation.
// - L: Light object representing the lighting parameters.


const int num_thread = 8; // number of triangles each thread process
std::mutex zbuffer_mutex;
const int TILE_SIZE = 32; // for s2 32 better than 64; 8 thread same 11

void processTriangleBatch(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L, int start, int end) {
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh->world;

    // Iterate through all triangles in the mesh
    //for (triIndices& ind : mesh->triangles) {
    for (unsigned int j = start; j < end; j++) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) {
            t[i].p = p * mesh->vertices[mesh->triangles[j].v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates

            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh->world * mesh->vertices[mesh->triangles[j].v[i]].normal;
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

            // Copy vertex colours
            t[i].rgb = mesh->vertices[mesh->triangles[j].v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        // Create a triangle object and render it
        triangle tri(t[0], t[1], t[2], mesh);
        {
            // protect z-buffer
            std::lock_guard<std::mutex> lock(zbuffer_mutex);
            tri.draw(renderer, L, mesh->ka, mesh->kd);
        }
    }
}

void renderParallel(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L) {
    int num_triangles = mesh->triangles.size();
    std::vector<std::thread> threads;
    int batchSize = num_triangles / num_thread; 

    for (int i = 0; i < num_thread; i++) {
        int start = i * batchSize;
        int end = (i == num_thread - 1) ? num_triangles : (start + batchSize);
        threads.emplace_back(processTriangleBatch, std::ref(renderer), mesh, std::ref(camera), std::ref(L), start, end);
    }

    for (auto& t : threads) {
        t.join(); 
    }
}

void renderTrianglesInsideTile(Renderer& renderer, std::vector<triangle>& triangles, int tileX, int tileY, Light& L) {
    int startX = tileX * TILE_SIZE;
    int startY = tileY * TILE_SIZE;
    int endX = min(startX + TILE_SIZE, renderer.canvas.getWidth());
    int endY = min(startY + TILE_SIZE, renderer.canvas.getHeight());
    
    for (triangle& tri : triangles) {
        tri.drawParallel(renderer, L, startX, startY, endX, endY);
    }
}

void processTileParallel(Renderer& renderer, std::vector<triangle>& triangles, Light& L, int threadID) {
    // divide scene into tiles
    int numTilesX = (renderer.canvas.getWidth() + TILE_SIZE - 1) / TILE_SIZE;
    int numTilesY = (renderer.canvas.getHeight() + TILE_SIZE - 1) / TILE_SIZE;
    // iterate the tiles and assign a grid of tiles to each thread
    for (int tileY = 0; tileY < numTilesY; tileY++) {
        for (int tileX = 0; tileX < numTilesX; tileX++) {
            // ensure different threads process different tiles
            if ((tileY * numTilesX + tileX) % num_thread == threadID) {  
                // draw triangles in this grid of tile
                renderTrianglesInsideTile(renderer, triangles, tileX, tileY, L);
            }
        }
    }
}

void collectTriangles(Renderer& renderer, std::vector<Mesh*>& scene, matrix& camera, Light& L) {
    std::vector<triangle> allTriangles;
    // triangle transformation and collection -------------------------------------
    for (Mesh* mesh : scene) {
        // Combine perspective, camera, and world transformations for the mesh
        matrix p = renderer.perspective * camera * mesh->world;
        for (triIndices& ind : mesh->triangles) {
            Vertex t[3];
            for (int i = 0; i < 3; i++) {
                t[i].p = p * mesh->vertices[ind.v[i]].p; // Apply transformations
                t[i].p.divideW(); // Perspective division to normalize coordinates

                // Transform normals into world space for accurate lighting
                // no need for perspective correction as no shearing or non-uniform scaling
                t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal;
                t[i].normal.normalise();

                // Map normalized device coordinates to screen space
                t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
                t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
                t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis

                // Copy vertex colours
                t[i].rgb = mesh->vertices[ind.v[i]].rgb;
                
            }
            // Clip triangles with Z-values outside [-1, 1]
            if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;
            allTriangles.emplace_back(t[0], t[1], t[2], mesh);
        }
    }
    // create threads ---------------------------------------------------------------
    std::vector<std::thread> threads;
    for (int i = 0; i < num_thread; i++) {
        threads.emplace_back(processTileParallel, std::ref(renderer), std::ref(allTriangles), std::ref(L), i);
    }
    for (auto& t : threads) {
        t.join();
    }
}



void render(Renderer& renderer, Mesh* mesh, matrix& camera, Light& L) {
    #ifdef USE_THREAD_TRI
        renderParallel(renderer, mesh, camera, L);
    #else
    // Combine perspective, camera, and world transformations for the mesh
    matrix p = renderer.perspective * camera * mesh->world;

    // Iterate through all triangles in the mesh
    for (triIndices& ind : mesh->triangles) {
        Vertex t[3]; // Temporary array to store transformed triangle vertices

        // Transform each vertex of the triangle
        for (unsigned int i = 0; i < 3; i++) {
            t[i].p = p * mesh->vertices[ind.v[i]].p; // Apply transformations
            t[i].p.divideW(); // Perspective division to normalize coordinates
            
            // Transform normals into world space for accurate lighting
            // no need for perspective correction as no shearing or non-uniform scaling
            t[i].normal = mesh->world * mesh->vertices[ind.v[i]].normal;
            t[i].normal.normalise();

            // Map normalized device coordinates to screen space
            t[i].p[0] = (t[i].p[0] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getWidth());
            t[i].p[1] = (t[i].p[1] + 1.f) * 0.5f * static_cast<float>(renderer.canvas.getHeight());
            t[i].p[1] = renderer.canvas.getHeight() - t[i].p[1]; // Invert y-axis
            //std::cout << "Projected: (" << t[i].p[0] << ", " << t[i].p[1] << ", " << t[i].p[2] << ", " << t[i].p[3] << ")\n";
            // Copy vertex colours
            t[i].rgb = mesh->vertices[ind.v[i]].rgb;
        }

        // Clip triangles with Z-values outside [-1, 1]
        if (fabs(t[0].p[2]) > 1.0f || fabs(t[1].p[2]) > 1.0f || fabs(t[2].p[2]) > 1.0f) continue;

        // Create a triangle object and render it
        triangle tri(t[0], t[1], t[2], mesh);
        tri.draw(renderer, L, mesh->ka, mesh->kd);
    }
    #endif
}

// Test scene function to demonstrate rendering with user-controlled transformations
// No input variables
void sceneTest() {
    Renderer renderer;
    // create light source {direction, diffuse intensity, ambient intensity}
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };
    // camera is just a matrix
    matrix camera = matrix::makeIdentity(); // Initialize the camera with identity matrix

    bool running = true; // Main loop control variable

    std::vector<Mesh*> scene; // Vector to store scene objects

    // Create a sphere and a rectangle mesh
    Mesh mesh = Mesh::makeSphere(1.0f, 10, 20);
    //Mesh mesh2 = Mesh::makeRectangle(-2, -1, 2, 1);

    // add meshes to scene
    scene.push_back(&mesh);
   // scene.push_back(&mesh2); 

    float x = 0.0f, y = 0.0f, z = -4.0f; // Initial translation parameters
    mesh.world = matrix::makeTranslation(x, y, z);
    //mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput(); // Handle user input
        renderer.clear(); // Clear the canvas for the next frame

        // Apply transformations to the meshes
     //   mesh2.world = matrix::makeTranslation(x, y, z) * matrix::makeRotateX(0.01f);
        mesh.world = matrix::makeTranslation(x, y, z);

        // Handle user inputs for transformations
        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;
        if (renderer.canvas.keyPressed('A')) x += -0.1f;
        if (renderer.canvas.keyPressed('D')) x += 0.1f;
        if (renderer.canvas.keyPressed('W')) y += 0.1f;
        if (renderer.canvas.keyPressed('S')) y += -0.1f;
        if (renderer.canvas.keyPressed('Q')) z += 0.1f;
        if (renderer.canvas.keyPressed('E')) z += -0.1f;

        // Render each object in the scene
        for (auto& m : scene)
            render(renderer, m, camera, L);

        renderer.present(); // Display the rendered frame
    }
}

// Utility function to generate a random rotation matrix
// No input variables
matrix makeRandomRotation() {
    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();
    unsigned int r = rng.getRandomInt(0, 3);

    switch (r) {
    case 0: return matrix::makeRotateX(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 1: return matrix::makeRotateY(rng.getRandomFloat(0.f, 2.0f * M_PI));
    case 2: return matrix::makeRotateZ(rng.getRandomFloat(0.f, 2.0f * M_PI));
    default: return matrix::makeIdentity();
    }
}

// Each thread iterates through its mesh batch
void renderMeshParallel(Renderer& renderer, std::vector<Mesh*>& scene, matrix& camera, Light& L, int start, int end) {
    for (unsigned int j = start; j < end; j++) {
        render(renderer, scene[j], camera, L); // render each mesh
    }
}

void switchRender(Renderer& renderer, std::vector<Mesh*>& scene, matrix& camera, Light& L) {
// Tiled-based Parallel Rendering
#if defined(USE_THREAD_TILE)
    collectTriangles(renderer, scene, camera, L);
// Mesh-Level Parallel Rendering
#elif defined(USE_THREAD_MESH)
    int num_meshes = scene.size(); // number of meshes in the scene
    std::vector<std::thread> threads;
    int batchSize = num_meshes / num_thread; // number of meshes each thread handles

    // Assigning a mesh to multiple threads for parallel rendering
    for (int i = 0; i < num_thread; i++) {
        int start = i * batchSize;
        int end = (i == num_thread - 1) ? num_meshes : (start + batchSize); // starting mesh index for current thread
        // ending mesh index for current thread; last thread contains all the meshes left
        threads.emplace_back(renderMeshParallel, std::ref(renderer), std::ref(scene), std::ref(camera), std::ref(L), start, end);
    }
    for (auto& t : threads) {
        t.join();
    }
#else // Basic Rendering
    for (auto& m : scene)
        render(renderer, m, camera, L);
#endif
}

// Function to render a scene with multiple objects and dynamic transformations
// No input variables
void scene1() {
    Renderer renderer;
    matrix camera;
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };

    bool running = true;

    std::vector<Mesh*> scene;

    // Create a scene of 40 cubes with random rotations
    for (unsigned int i = 0; i < 20; i++) {
        Mesh* m = new Mesh();
        *m = Mesh::makeCube(1.f);
        m->world = matrix::makeTranslation(-2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
        m = new Mesh();
        *m = Mesh::makeCube(1.f);
        m->world = matrix::makeTranslation(2.0f, 0.0f, (-3 * static_cast<float>(i))) * makeRandomRotation();
        scene.push_back(m);
    }

    float zoffset = 8.0f; // Initial camera Z-offset
    float step = -0.1f;  // Step size for camera movement

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    // Main rendering loop
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        camera = matrix::makeTranslation(0, 0, -zoffset); // Update camera position

        // Rotate the first two cubes in the scene
        scene[0]->world = scene[0]->world * matrix::makeRotateXYZ(0.1f, 0.1f, 0.0f);
        scene[1]->world = scene[1]->world * matrix::makeRotateXYZ(0.0f, 0.1f, 0.2f);

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        zoffset += step;
        if (zoffset < -60.f || zoffset > 8.f) {
            step *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                //std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                std::cout << std::chrono::duration<double, std::milli>(end - start).count() << "\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        switchRender(renderer, scene, camera, L);

        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

// Scene with a grid of cubes and a moving sphere
// No input variables
void scene2() {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };

    std::vector<Mesh*> scene;

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    // Create a grid of cubes with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) {
            Mesh* m = new Mesh();
            *m = Mesh::makeCube(1.f);
            scene.push_back(m);
            m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f);
            rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
            rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    Mesh* sphere = new Mesh();
    *sphere = Mesh::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                //std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                std::cout << std::chrono::duration<double, std::milli>(end - start).count() << "\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        switchRender(renderer, scene, camera, L);

        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}

void scene3() {
    Renderer renderer;
    matrix camera = matrix::makeIdentity();
    Light L{ vec4(0.f, 1.f, 1.f, 0.f), colour(1.0f, 1.0f, 1.0f), colour(0.1f, 0.1f, 0.1f) };

    std::vector<Mesh*> scene;

    struct rRot { float x; float y; float z; }; // Structure to store random rotation parameters
    std::vector<rRot> rotations;

    RandomNumberGenerator& rng = RandomNumberGenerator::getInstance();

    // Create a grid of spheres with random rotations
    for (unsigned int y = 0; y < 6; y++) {
        for (unsigned int x = 0; x < 8; x++) 
            for (unsigned int z = 0; z < 4; z++) {
                Mesh* m = new Mesh();
                *m = Mesh::makeSphere(1.0f, 10, 20);
                scene.push_back(m);
                m->world = matrix::makeTranslation(-7.0f + (static_cast<float>(x) * 2.f), 5.0f - (static_cast<float>(y) * 2.f), -8.f - static_cast<float>(z));
                rRot r{ rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f), rng.getRandomFloat(-.1f, .1f) };
                rotations.push_back(r);
        }
    }

    // Create a sphere and add it to the scene
    Mesh* sphere = new Mesh();
    *sphere = Mesh::makeSphere(1.0f, 10, 20);
    scene.push_back(sphere);
    float sphereOffset = -6.f;
    float sphereStep = 0.1f;
    sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);

    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    int cycle = 0;

    bool running = true;
    while (running) {
        renderer.canvas.checkInput();
        renderer.clear();

        // Rotate each cube in the grid
        for (unsigned int i = 0; i < rotations.size(); i++)
            scene[i]->world = scene[i]->world * matrix::makeRotateXYZ(rotations[i].x, rotations[i].y, rotations[i].z);

        // Move the sphere back and forth
        sphereOffset += sphereStep;
        sphere->world = matrix::makeTranslation(sphereOffset, 0.f, -6.f);
        if (sphereOffset > 6.0f || sphereOffset < -6.0f) {
            sphereStep *= -1.f;
            if (++cycle % 2 == 0) {
                end = std::chrono::high_resolution_clock::now();
                //std::cout << cycle / 2 << " :" << std::chrono::duration<double, std::milli>(end - start).count() << "ms\n";
                std::cout << std::chrono::duration<double, std::milli>(end - start).count() << "\n";
                start = std::chrono::high_resolution_clock::now();
            }
        }

        if (renderer.canvas.keyPressed(VK_ESCAPE)) break;

        switchRender(renderer, scene, camera, L);

        renderer.present();
    }

    for (auto& m : scene)
        delete m;
}


// Entry point of the application
// No input variables
int main() {
    // Uncomment the desired scene function to run
    //scene1();
    //scene2();
    scene3();
    //sceneTest(); 
    

    return 0;
}