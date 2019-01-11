//Real-time Ink Simulation Implementation

//The algorithm for fluid simulation is from Stam's "Stable Fluids" and 
//"Real-time fluid dynamics for games" 
//The algorithm for ink simulation is from Shibiao Xu et.al.'s 
//"Real-time ink simulation using a grid-particle method"

//Copyright © 2018 Peiyao Shi & Danhua Zhang. 
//All rights reserved.

#include "glad/glad.h"  //Include order can matter here
#ifndef _WIN32
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#else
#include <SDL.h>
#include <SDL_opengl.h>
#endif

#define INK_PARTICLE_NUM 100000
#define COARSE_PARTICLE_NUM 1000
#define MAX_LIFE_SPAN 10
#define EPSILON 1e-10
#define GLM_FORCE_RADIANS
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"

#include <fstream>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>
#include <ctime>
using namespace std;

typedef vector<float> vfloat;

struct Particle {
	glm::vec3 velocity;
	glm::vec3 force;
	glm::vec3 position;
	float lifespan;
};

//=============================parameters for SDL=============================
bool fullscreen = false;
bool saveOutput = false; //Make to true to save out your animation
bool notpause = false;
int screen_width = 650;
int screen_height = 650;
float aspect; //aspect ratio (needs to be updated if the window is resized)
static char* readShaderSource(const char* shaderFile);
GLuint InitShader(const char* vShaderFileName, const char* fShaderFileName);
bool DEBUG_ON = true;

//===============================Shader sources===============================
const GLchar* vertexSource =
"#version 150 core\n"
"in vec3 position;"
//"in vec3 inColor;"
//the color will be changed in the OpenGL part
//change the number of particles
"uniform vec3 inColor;"
"out vec3 Color;"
"out vec3 lightDir;"
"uniform mat4 model;"
"uniform mat4 view;"
"uniform mat4 proj;"
"void main() {"
"   Color = inColor;"
"   gl_Position = proj * view * model * vec4(position, 1.0);"
"}";

const GLchar* fragmentSource =
"#version 150 core\n"
"in vec3 Color;"
"out vec4 outColor;"
"const float ambient = .2;"
"void main() {"
"   vec3 ambC = Color * ambient;"
// the alpha channel of all pixels are 1.0, i.e. the opacity is 100%
"   outColor = vec4(ambC, 1.0);"
"}";

//Index of where to model, view, and projection matricies are stored on the GPU
GLint uniModel, uniView, uniProj, uniColor;

const float VISC = 0;
const float dt = 0.005;
const float DIFF = 0.01;
const int N = 100; // Grid size
glm::vec3 ink_color = glm::vec3(0.0f);
float density = 1.0f;
float ink_radius = 14;
float coarse_radius = 15;
float initial_size = 5.0f;
int num_newp = 3;
int ink_num = INK_PARTICLE_NUM;
int coarse_num = COARSE_PARTICLE_NUM;

//the valid size of the grid is from 1 to N rows and 1 to N columns
const int nsize = (N+2)*(N+2);

inline int IX(int i, int j){return i + (N+2)*j;}

float randf() {
	return (float)(rand() % 1001) * 0.001f;
}

//Bilinear Interpolation
float Interpolate(vfloat &f, glm::vec3 position) {
	float Vx, Vy;

	int x = (int)(floor(position.x));
	int y = (int)(floor(position.y));

	// Interpolate the x component
	int i1 = position.x - x;
	int i2 = 1 - i1;
	int j1 = position.y - y;
	int j2 = 1 - j1;
	float f1, f2, f3 = 0;

	if (x > 0 && y < N+1) {
		f1 = j1 * f[IX(x, y)] + j2 * f[IX(x - 1, y)];
		f2 = j1 * f[IX(x, y + 1)] + j2 * f[IX(x - 1, y + 1)];
		f3 = i1 * f2 + i2 * f1;
	}

	return f3;
}

void ParticleToGrid(Particle* particle, vfloat &u0, vfloat &v0, vfloat &n, int num_particle, float dt) {
	for (int i = 0; i < num_particle; i++) {
		int x = (int)floor(particle[i].position.x);
		int y = (int)floor(particle[i].position.y);
		int idx = y * (N + 2) + x;
		u0[idx] += particle[i].force.x * dt;
		v0[idx] += particle[i].force.y * dt;
		n[idx]++;
	}
}

// Bounds (currently a box with solid walls)
void set_bnd(const int b, vfloat &x)
{   
    for (int i=1; i<=N; i++)
    {
        x[IX(0  ,i)] = b==1 ? -x[IX(1,i)] : x[IX(1,i)];
        x[IX(N+1,i)] = b==1 ? -x[IX(N,i)] : x[IX(N,i)];
        x[IX(i,  0)] = b==2 ? -x[IX(i,1)] : x[IX(i,1)];
        x[IX(i,N+1)] = b==2 ? -x[IX(i,N)] : x[IX(i,N)];
    }
    
    x[IX(0  ,0  )] = 0.5*(x[IX(1,0  )] + x[IX(0  ,1)]);
    x[IX(0  ,N+1)] = 0.5*(x[IX(1,N+1)] + x[IX(0  ,N)]);
    x[IX(N+1,0  )] = 0.5*(x[IX(N,0  )] + x[IX(N+1,1)]);
    x[IX(N+1,N+1)] = 0.5*(x[IX(N,N+1)] + x[IX(N+1,N)]);
}

inline void lin_solve(int b, vfloat &x, const vfloat &x0, float a, float c)
{
    for (int k=0; k<20; k++)
    {
        for (int i=1; i<=N; i++)
        {
            for (int j=1; j<=N; j++)
                x[IX(i,j)] = (x0[IX(i,j)] +
                              a*(x[IX(i-1,j)]+x[IX(i+1,j)]+x[IX(i,j-1)]+x[IX(i,j+1)])) / c;         
        }
        set_bnd (b, x);
    }
}

// Add forces
void add_source(vfloat &x, const vfloat &s, float dt) {
	for (int i = 0; i < nsize; i++) {
		x[i] += dt * s[i];
	}
}

void add_constant_force(vfloat &x, float s, float dt) {
	for (int i = 0; i < nsize; i++) {
		x[i] += s*dt;
	}
}

void add_gravity(vfloat &x, float dt) {
	for (int i = 0; i < nsize; i++) {
		x[i] += dt * -9.8;
	}
}

void add_force_to_particle(Particle* particle, int num, glm::vec3 accelaration, float dt) {
	for (int i = 0; i < num; i++) {
		particle[i].force += accelaration;
	}
}

void reset_coarse(Particle* coarse) {
	coarse_num = COARSE_PARTICLE_NUM;
	for (int i = 0; i < COARSE_PARTICLE_NUM; i++) {
		//choose random particle position
		float x = randf()*initial_size - initial_size * 0.5f + N / 2;
		float y = randf()*initial_size - initial_size * 1.5f + N;
		coarse[i].position = glm::vec3(x, y, 0.0f);
		coarse[i].velocity = glm::vec3(EPSILON, 0.0f, 0.0f);
		coarse[i].force = glm::vec3(0.0f, -98.f, 0.0f);
		coarse[i].lifespan = MAX_LIFE_SPAN;
	}
}

void reset_ink(Particle* ink) {
	ink_num = INK_PARTICLE_NUM;
	ink = (Particle *)realloc(ink, sizeof(Particle)*INK_PARTICLE_NUM);
	for (int i = 0; i < INK_PARTICLE_NUM; i++) {
		//choose random particle position
		float x = randf()*initial_size - initial_size * 0.5f + N / 2;
		float y = randf()*initial_size - initial_size * 1.5f + N;
		ink[i].position = glm::vec3(x, y, 0.0f);
		ink[i].velocity = glm::vec3(EPSILON, 0.0f, 0.0f);
		ink[i].force = glm::vec3(0.0f, 0.0f, 0.0f);
		ink[i].lifespan = MAX_LIFE_SPAN;
	}
}

// Diffusion with Gauss-Seidel relaxation
void diffuse(int b, vfloat &x, const vfloat &x0, float diff, float dt)
{
    float a = dt*diff*N*N;
    lin_solve(b, x, x0, a, 1+4*a); // Amazing fix due to Iwillnotexist Idonotexist
}

// Backwards advection
void advect(int b, vfloat &d, const vfloat &d0, const vfloat &u, const vfloat &v, float dt)
{
    float dt0 = dt*N;
    for (int i=1; i<=N; i++) {
        for (int j=1; j<=N; j++) {
            float x = i - dt0*u[IX(i,j)];
            float y = j - dt0*v[IX(i,j)];
            if (x<0.5) x=0.5; if (x>N+0.5) x=N+0.5;
            int i0=(int)x; int i1=i0+1;
            if (y<0.5) y=0.5; if (y>N+0.5) y=N+0.5;
            int j0=(int)y; int j1=j0+1;
            
            float s1 = x-i0; float s0 = 1-s1; float t1 = y-j0; float t0 = 1-t1;
            d[IX(i,j)] = s0*(t0*d0[IX(i0,j0)] + t1*d0[IX(i0,j1)]) + s1*(t0*d0[IX(i1,j0)] + t1*d0[IX(i1,j1)]);
        }
    }
    set_bnd(b, d);
}

// BFECC
void advect_BFECC(int b, vfloat &d, const vfloat &d0, const vfloat &u, const vfloat &v, float dt) {
	vfloat minus_u(nsize, 0), minus_v(nsize, 0), d_copy(nsize, 0), d_bar(nsize,0);
	for (int i = 0; i < nsize; i++) {
		minus_u[i] = -u[i];
		minus_v[i] = -v[i];
		d_copy[i] = d[i];
	}
	
	advect(b, d, d0, u, v, dt);	//d = L(dn,dn)
	for (int i = 0; i < nsize; i++) {
		d_bar[i] = d[i];
	}

	advect(b, d, d_bar, minus_u, minus_v, dt); //d = L(-dn, L(dn,dn))
	minus_u.clear(); minus_v.clear(); d_bar.clear();
	for (int i = 0; i < nsize; i++) {
		d_copy[i] += 0.5*(d_copy[i] - d[i]);
	}
	advect(b, d, d_copy, u, v, dt); //d = L(dn, dn+0.5*(dn-L(-dn, L(dn,dn))))
	d_copy.clear();
}

// Force velocity to be mass-conserving (Poisson equation black magic)
void project(vfloat &u, vfloat &v, vfloat &p, vfloat &div) {
	float h = 1.0 / N;
	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			div[IX(i, j)] = -0.5*h*(u[IX(i + 1, j)] - u[IX(i - 1, j)] + v[IX(i, j + 1)] - v[IX(i, j - 1)]);
			p[IX(i, j)] = 0;
		}
	}
	set_bnd(0, div); set_bnd(0, p);

	lin_solve(0, p, div, 1, 4);

	for (int i = 1; i <= N; i++) {
		for (int j = 1; j <= N; j++) {
			u[IX(i, j)] -= 0.5*(p[IX(i + 1, j)] - p[IX(i - 1, j)]) / h;
			v[IX(i, j)] -= 0.5*(p[IX(i, j + 1)] - p[IX(i, j - 1)]) / h;
		}
	}
	set_bnd(1, u); set_bnd(2, v);
}

void UpdateInkParticle(vfloat &u, vfloat &v, Particle* ink, int num_inkp, float density, float dt) {
	glm::vec3 p1, v1;
	for (int i = 0; i < num_inkp; i++) {
		if (ink[i].lifespan > 0) {
			p1 = ink[i].position + 0.5f * ink[i].velocity * dt;

			ink[i].velocity.x = Interpolate(u, ink[i].position);
			ink[i].velocity.y = Interpolate(v, ink[i].position);
			//int x = (int)floor(ink[i].position.x);
			//int y = (int)floor(ink[i].position.y);
			//ink[i].velocity.x = u[IX(x,y)];
			//ink[i].velocity.y = v[IX(x,y)];

			v1 = ink[i].velocity * 0.5f;
			ink[i].position = p1 + v1 * dt;	

			//ink[i].position+=ink[i].velocity*dt;

			if (ink[i].position.x < 1) {
				ink[i].position.x = 1;
				ink[i].velocity.x = 0;
			}
			else if (ink[i].position.x > (N + 1)) {
				ink[i].position.x = N;
				ink[i].velocity.x = 0;
			}
			if (ink[i].position.y < 1) {
				ink[i].position.y = 1;
				ink[i].velocity.y = 0;
			}
			else if (ink[i].position.y > (N + 1)) {
				ink[i].position.y = N;
				ink[i].velocity.y = 0;
			}
		}
	}
}

// Velocity solver: addition of forces, viscous diffusion, self-advection
void vel_step_Stam(vfloat &u, vfloat &v, vfloat &u0, vfloat &v0, float visc, float dt) {
	add_source(u, u0, dt); 	add_source(v, v0, dt);
	swap(u0, u); diffuse(1, u, u0, visc, dt);
	swap(v0, v); diffuse(2, v, v0, visc, dt);
	project(u, v, u0, v0);
	swap(u0, u); swap(v0, v);
	advect(1, u, u0, u0, v0, dt); advect(2, v, v0, u0, v0, dt);
	project(u, v, u0, v0);
}

// Velocity update method of real-time ink simulation
void vel_step_BFECC(vfloat &u, vfloat &v, vfloat &u0, vfloat &v0, float visc, float dt) {
	advect_BFECC(1, u, u0, u0, v0, dt);
	advect_BFECC(2, v, v0, u0, v0, dt);
	diffuse(1, u, u0, visc, dt);
	diffuse(2, v, v0, visc, dt);
	add_gravity(v, dt);

	swap(u0, u); swap(v0, v);
	project(u, v, u0, v0);
	project(u, v, u0, v0);
}

void vel_step(vfloat &u, vfloat &v, vfloat &u0, vfloat &v0, float visc, float dt) {
	//add_source(u, u0, dt); 
	advect(1, u, u0, u0, v0, dt); 
	advect(2, v, v0, u0, v0, dt);

	diffuse(1, u, u0, visc, dt);
	diffuse(2, v, v0, visc, dt);
	
	add_gravity(v, dt);

	swap(u0, u); swap(v0, v);

	project(u, v, u0, v0);
	project(u, v, u0, v0);
}

int main(int argc, char *argv[])
{ 
    // SDL initialize
	SDL_Init(SDL_INIT_VIDEO);  //Initialize Graphics (for OpenGL)

	//Ask SDL to get a recent version of OpenGL (3.2 or greater)
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 4);

	//Create a window (offsetx, offsety, width, height, flags)
	SDL_Window* window = SDL_CreateWindow("My OpenGL Program", 300, 30, screen_width, screen_height, SDL_WINDOW_OPENGL);
	aspect = screen_width / (float)screen_height; //aspect ratio (needs to be updated if the window is resized)

	//Create a context to draw in
	SDL_GLContext context = SDL_GL_CreateContext(window);

	if (gladLoadGLLoader(SDL_GL_GetProcAddress)) {
		printf("\nOpenGL loaded\n");
		printf("Vendor:   %s\n", glGetString(GL_VENDOR));
		printf("Renderer: %s\n", glGetString(GL_RENDERER));
		printf("Version:  %s\n\n", glGetString(GL_VERSION));
	}
	else {
		printf("ERROR: Failed to initialize OpenGL context.\n");
		return -1;
	}

	//Build a Vertex Array Object. This stores the VBO and attribute mappings in one object
	GLuint vao;
	glGenVertexArrays(1, &vao); //Create a VAO
	glBindVertexArray(vao); //Bind the above created VAO to the current context

	int point_num = 3;
	float point[] = { 0.0f, 0.0f, 0.0f }; // the position of the point

	//Allocate memory on the graphics card to store geometry (vertex buffer object)
	GLuint vbo;
	glGenBuffers(1, &vbo);  //Create 1 buffer called vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo); //Set the vbo as the active array buffer (Only one buffer can be active at a time)
	glBufferData(GL_ARRAY_BUFFER, point_num * sizeof(float), point, GL_STATIC_DRAW); //upload vertices to vbo

	//Load the vertex Shader
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);

	//Let's double check the shader compiled 
	GLint status;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &status);
	if (!status) {
		char buffer[512];
		glGetShaderInfoLog(vertexShader, 512, NULL, buffer);
		SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,
			"Compilation Error",
			"Failed to Compile: Check Consol Output.",
			NULL);
		printf("Vertex Shader Compile Failed. Info:\n\n%s\n", buffer);
	}

	//Load the fragment Shader
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);

	//Double check the shader compiled 
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &status);
	if (!status) {
		char buffer[512];
		glGetShaderInfoLog(fragmentShader, 512, NULL, buffer);
		SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR,
			"Compilation Error",
			"Failed to Compile: Check Consol Output.",
			NULL);
		printf("Fragment Shader Compile Failed. Info:\n\n%s\n", buffer);
	}

	//Join the vertex and fragment shaders together into one program
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glBindFragDataLocation(shaderProgram, 0, "outColor"); // set output
	glLinkProgram(shaderProgram); //run the linker

	glUseProgram(shaderProgram); //Set the active shader (only one can be used at a time)

	//Tell OpenGL how to set fragment shader input 
	GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
	glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	//Attribute, vals/attrib., type, normalized?, stride, offset
	//Binds to VBO current GL_ARRAY_BUFFER 
	glEnableVertexAttribArray(posAttrib);

	glBindVertexArray(0); //Unbind the VAO


	//// Allocate Texture 0 (wood) ///////
	SDL_Surface* surface = SDL_LoadBMP("texture.bmp");
	if (surface == NULL) { //If it failed, print the error
		printf("Error: \"%s\"\n", SDL_GetError()); return 1;
	}
	GLuint tex0;
	glGenTextures(1, &tex0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex0);

	//What to do outside 0-1 range
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	//Load the texture into memory
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surface->w, surface->h, 0, GL_BGR, GL_UNSIGNED_BYTE, surface->pixels);
	glGenerateMipmap(GL_TEXTURE_2D);

	SDL_FreeSurface(surface);
	//// End Allocate Texture ///////

	//--------------------------------------
	//create cloth nodes (nv*nh)
	//the 0th row is the top row, fixed
	//--------------------------------------
	int cloth_numVerts = 6;
	int cloth_numLines = cloth_numVerts * 8;
	float cloth[6 * 8] = {
		//position, normal, texture_cord
		0.0f,0.0f,0.0f, 0.0f,0.0f,1.0f, 0.0f,0.0f,
		0.0f,N + 2,0.0f, 0.0f,0.0f,1.0f, 0.0f,1.0f,
		N + 2, 0.0f,0.0f, 0.0f,0.0f,1.0f, 1.0f,1.0f,
		0.0f,N + 2,0.0f, 0.0f,0.0f,1.0f, 0.0f,0.0f,
		N + 2, 0.0f,0.0f, 0.0f,0.0f,1.0f, 1.0f,0.0f,
		N+2,N+2,0.0f, 0.0f,0.0f,1.0f, 1.0f,1.0f
	};

	//--------------------------------------
	// load cloth
	//--------------------------------------
	GLuint vbo_cloth;
	glGenBuffers(1, &vbo_cloth);  //Create 1 buffer called vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo_cloth); //Set the vbo as the active array buffer (Only one buffer can be active at a time)
	glBufferData(GL_ARRAY_BUFFER, cloth_numLines * sizeof(float), cloth, GL_STATIC_DRAW);

	//Build a Vertex Array Object. This stores the VBO and attribute mappings in one object
	GLuint vao_cloth;
	glGenVertexArrays(1, &vao_cloth); //Create a VAO
	glBindVertexArray(vao_cloth); //Bind the above created VAO to the current context

	//Join the vertex and fragment shaders together into one program
	int ClothShaderProgram = InitShader("vertexTex.glsl", "fragmentTex.glsl");
	glUseProgram(ClothShaderProgram);

	glBindVertexArray(vao_cloth);
	//Tell OpenGL how to set fragment shader input 
	posAttrib = glGetAttribLocation(ClothShaderProgram, "position");
	glVertexAttribPointer(posAttrib, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), 0);
	//Attribute, vals/attrib., type, normalized?, stride, offset
	//Binds to VBO current GL_ARRAY_BUFFER 
	glEnableVertexAttribArray(posAttrib);

	GLint texAttrib = glGetAttribLocation(ClothShaderProgram, "inTexcoord");
	glEnableVertexAttribArray(texAttrib);
	glVertexAttribPointer(texAttrib, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));

	GLint nolAttrib = glGetAttribLocation(ClothShaderProgram, "inNormal");
	glVertexAttribPointer(nolAttrib, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(nolAttrib);

	glBindVertexArray(0); //Unbind the VAO


	//Where to model, view, and projection matricies are stored on the GPU
	uniModel = glGetUniformLocation(shaderProgram, "model");
	uniView = glGetUniformLocation(shaderProgram, "view");
	uniProj = glGetUniformLocation(shaderProgram, "proj");
	uniColor = glGetUniformLocation(shaderProgram, "inColor");

	glEnable(GL_DEPTH_TEST);
    
	//==================================Initialization Starts here===================================
	static vfloat u(nsize, 0), v(nsize, 0), u_prev(nsize, 0), v_prev(nsize, 0); // Horizontal, vertical velocity
	static vfloat n(nsize, 0);
	//static vfloat dens(nsize, 0), dens_prev(nsize, 0);

	//Generate particles
	Particle *coarse = (Particle *)malloc(sizeof(Particle)*COARSE_PARTICLE_NUM);
	for (int i = 0; i < coarse_num; i++) {
		//choose random particle position
		float x = randf()*initial_size - initial_size * 0.5f + N / 2;
		float y = randf()*initial_size - initial_size * 1.5f + N;
		coarse[i].position = glm::vec3(x, y, 0.0f);
		coarse[i].velocity = glm::vec3(EPSILON, 0.0f, 0.0f);
		coarse[i].force = glm::vec3(0.0f, -98.f, 0.0f);
		coarse[i].lifespan = MAX_LIFE_SPAN;
	}

	Particle *ink = (Particle *)malloc(sizeof(Particle)*ink_num);
	for (int i = 0; i < ink_num; i++) {
		//choose random particle position
		float x = randf()*initial_size - initial_size * 0.5f + N / 2;
		float y = randf()*initial_size - initial_size * 1.5f + N;
		ink[i].position = glm::vec3(x, y, 0.0f);
		ink[i].velocity = glm::vec3(EPSILON, 0.0f, 0.0f);
		ink[i].force = glm::vec3(0.0f, 0.0f, 0.0f);
		ink[i].lifespan = MAX_LIFE_SPAN;
	}

	SDL_Event windowEvent;
	bool quit = false;

	srand(time(NULL));
	float lastTime = SDL_GetTicks() / 1000.f;
	float dt = 0;

	if (!notpause) {
		ParticleToGrid(coarse, u_prev, v_prev, n, coarse_num, dt);
	}

	while (!quit) {
		while (SDL_PollEvent(&windowEvent)) {
			if (windowEvent.type == SDL_QUIT) quit = true; //Exit event loop
			//List of keycodes: https://wiki.libsdl.org/SDL_Keycode - You can catch many special keys
			//Scancode referes to a keyboard position, keycode referes to the letter (e.g., EU keyboards)
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_ESCAPE)
				quit = true; ; //Exit event loop
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_f) //If "f" is pressed
				fullscreen = !fullscreen;
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_p) {
				//If "p" is pressed
				notpause = true;
				lastTime = SDL_GetTicks() / 1000.f;
			}

			//change the force of the coarse particles
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_UP) {
				add_force_to_particle(coarse, coarse_num, glm::vec3(0,10.0,0), dt);
				UpdateInkParticle(u, v, coarse, coarse_num, density, dt);
				UpdateInkParticle(u, v, ink, ink_num, density, dt);
			}
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_DOWN) {
				add_force_to_particle(coarse, coarse_num, glm::vec3(0, -10.0, 0), dt);
				UpdateInkParticle(u, v, coarse, coarse_num, density, dt);
				UpdateInkParticle(u, v, ink, ink_num, density, dt);
			}
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_LEFT) {
				add_force_to_particle(coarse, coarse_num, glm::vec3(-10, 0.0, 0), dt);
				UpdateInkParticle(u, v, coarse, coarse_num, density, dt);
				UpdateInkParticle(u, v, ink, ink_num, density, dt);
			}
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_RIGHT) {
				add_force_to_particle(coarse, coarse_num, glm::vec3(10, 0, 0), dt);
				UpdateInkParticle(u, v, coarse, coarse_num, density, dt);
				UpdateInkParticle(u, v, ink, ink_num, density, dt);
			}

			//change the velocity of the fluid field
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_w) {
				add_constant_force(v, 100, dt);
				UpdateInkParticle(u, v, coarse, coarse_num, density, dt);
				UpdateInkParticle(u, v, ink, ink_num, density, dt);
			}
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_s) {
				add_constant_force(v, -100, dt);
				UpdateInkParticle(u, v, coarse, coarse_num, density, dt);
				UpdateInkParticle(u, v, ink, ink_num, density, dt);
			}
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_a) {
				add_constant_force(u, -100, dt);
				UpdateInkParticle(u, v, coarse, coarse_num, density, dt);
				UpdateInkParticle(u, v, ink, ink_num, density, dt);
			}
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_d) {
				add_constant_force(u, 100, dt);
				UpdateInkParticle(u, v, coarse, coarse_num, density, dt);
				UpdateInkParticle(u, v, ink, ink_num, density, dt);
			}

			//reset
			if (windowEvent.type == SDL_KEYUP && windowEvent.key.keysym.sym == SDLK_r) {
				reset_coarse(coarse);
				reset_ink(ink);
			}

			SDL_SetWindowFullscreen(window, fullscreen ? SDL_WINDOW_FULLSCREEN : 0); //Set to full screen

			// add ink particles
			int mx, my;
			if (SDL_GetMouseState(&mx, &my) & SDL_BUTTON(SDL_BUTTON_LEFT)) {
				/*float x = 2 * mx / (float)screen_width - 1 + N / 2;
				float y = 1 - 2 * my / (float)screen_height + N / 2;*/
				float x = (float(mx)/ float(screen_width))*N; float y = ((float(screen_height) - float(my))/float(screen_height))*N;
				if (1 <= x && x <= N && 1 <= y && y <= N) {
					Particle *new_particle = (Particle *)realloc(ink, sizeof(Particle)*(num_newp + ink_num));
					if (!new_particle) {
						printf("GG");
						exit(0);
						//free(new_particle);
					}
					ink = new_particle;
					
					for (int i = ink_num; i < ink_num + num_newp; i++) {
						ink[i].position = glm::vec3(x+randf()-0.5f, y+randf()-0.5f, 0.0f);
						ink[i].velocity = glm::vec3(randf()-0.5, randf()-0.5, 0.0f);
						ink[i].lifespan = MAX_LIFE_SPAN;
					}
					ink_num = ink_num + num_newp;
					printf("ink_num: %d\t", ink_num);
					printf("x:%f, y:%f\n", x, y);
				}
			}

			// add coarse particles
			if (SDL_GetMouseState(&mx, &my) & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
				/*float x = 2 * mx / (float)screen_width - 1 + N / 2;
				float y = 1 - 2 * my / (float)screen_height + N / 2;*/
				float x = (float(mx) / float(screen_width))*N; float y = ((float(screen_height) - float(my)) / float(screen_height))*N;
				if (1 <= x && x <= N && 1 <= y && y <= N) {
					Particle *new_particle = (Particle *)realloc(coarse, sizeof(Particle)*(num_newp + coarse_num));
					if (!new_particle) {
						printf("GG");
						exit(0);
						//free(new_particle);
					}
					coarse = new_particle;

					for (int i = coarse_num; i < coarse_num + num_newp; i++) {
						coarse[i].position = glm::vec3(x + randf() - 0.5f, y + randf() - 0.5f, 0.0f);
						coarse[i].velocity = glm::vec3(randf() - 0.5, randf() - 0.5, 0.0f);
						coarse[i].force = glm::vec3(0.0f, -98.f, 0.0f);
						coarse[i].lifespan = MAX_LIFE_SPAN;
					}
					coarse_num = coarse_num + num_newp;
					printf("coarse_num: %d\t", coarse_num);
					printf("x:%f, y:%f\n", x, y);
				}
			}
		}

		// Clear the screen to default color
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		if (notpause) {
			if (!saveOutput) dt = (SDL_GetTicks() / 1000.f) - lastTime;
			if (dt > .1) dt = .1; //Have some max dt
			lastTime = SDL_GetTicks() / 1000.f;
			if (saveOutput) dt += .07; //Fix framerate at 14 FPS

			//================================Update the Simulation==============================
			ParticleToGrid(coarse, u_prev, v_prev, n, coarse_num, dt);
			//vel_step_Stam(u, v, u_prev, v_prev, VISC, dt);
			//vel_step(u, v, u_prev, v_prev, VISC, dt);
			vel_step_BFECC(u, v, u_prev, v_prev, VISC, dt);
			UpdateInkParticle(u, v, coarse, coarse_num, density, dt);
			UpdateInkParticle(u, v, ink, ink_num, density, dt);
		}

		//====================================Start Drawing Here================================
		glm::mat4 view = glm::lookAt(
			glm::vec3(N / 2 + 1.f, N / 2 + 1.f, 123.5f),  //Cam Position
			glm::vec3(N / 2 + 1.f, N / 2 + 1.f, 0.0f),  //Look at point
			glm::vec3(0.0f, 1.0f, 0.0f)); //Up
		GLint uniView = glGetUniformLocation(shaderProgram, "view");
		glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

		glm::mat4 proj = glm::perspective(3.14f / 4, aspect, 0.1f, 100000.0f); //FOV, aspect, near, far
		GLint uniProj = glGetUniformLocation(shaderProgram, "proj");
		glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

		//draw coarse particles
		glPointSize(coarse_radius);
		for (int i = 0; i < coarse_num; i++) {
			//ink[i].lifespan -= dt;
			glm::vec3 inColor = ink_color;
			glUniform3f(uniColor, inColor.r, inColor.g, inColor.b);

			glm::mat4 model(1.0f);
			model = glm::translate(model, coarse[i].position);
			glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

			glBindVertexArray(vao);
			glDrawArrays(GL_POINTS, 0, 1); //(Primitives, Which VBO, Number of vertices)
		}

		//draw ink particles
		glPointSize(ink_radius);
		for (int i = 0; i < ink_num; i++) {
			//ink[i].lifespan -= dt;
			glm::vec3 inColor = ink_color;
			glUniform3f(uniColor, inColor.r, inColor.g, inColor.b);

			glm::mat4 model(1.0f);
			model = glm::translate(model, ink[i].position);
			glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(model));

			glBindVertexArray(vao);
			glDrawArrays(GL_POINTS, 0, 1); //(Primitives, Which VBO, Number of vertices)
		}

		//draw cloth
		glUseProgram(ClothShaderProgram);
		uniProj = glGetUniformLocation(ClothShaderProgram, "proj");
		glUniformMatrix4fv(uniProj, 1, GL_FALSE, glm::value_ptr(proj));

		uniView = glGetUniformLocation(ClothShaderProgram, "view");
		glUniformMatrix4fv(uniView, 1, GL_FALSE, glm::value_ptr(view));

		glm::mat4 cloth_model(1.0f);
		uniModel = glGetUniformLocation(ClothShaderProgram, "model");
		glUniformMatrix4fv(uniModel, 1, GL_FALSE, glm::value_ptr(cloth_model));

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex0);
		glUniform1i(glGetUniformLocation(ClothShaderProgram, "tex0"), 0);

		glBindVertexArray(vao_cloth);
		GLint uniTexID = glGetUniformLocation(ClothShaderProgram, "texID");
		glUniform1i(uniTexID, 0); //Set texture ID to use 
		glDrawArrays(GL_TRIANGLES, 0, cloth_numVerts); //(Primitives, Which VBO, Number of vertices)
		glBindVertexArray(0);

		SDL_GL_SwapWindow(window); //Double buffering
	}

	//Clean Up
	glDeleteProgram(shaderProgram);
	glDeleteShader(fragmentShader);
	glDeleteShader(vertexShader);
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);

	SDL_GL_DeleteContext(context);
	SDL_Quit();
	return 0;
}

// Create a NULL-terminated string by reading the provided file
static char* readShaderSource(const char* shaderFile) {
	FILE *fp;
	long length;
	char *buffer;

	// open the file containing the text of the shader code
	fp = fopen(shaderFile, "r");

	// check for errors in opening the file
	if (fp == NULL) {
		printf("can't open shader source file %s\n", shaderFile);
		return NULL;
	}

	// determine the file size
	fseek(fp, 0, SEEK_END); // move position indicator to the end of the file;
	length = ftell(fp);  // return the value of the current position

						 // allocate a buffer with the indicated number of bytes, plus one
	buffer = new char[length + 1];

	// read the appropriate number of bytes from the file
	fseek(fp, 0, SEEK_SET);  // move position indicator to the start of the file
	fread(buffer, 1, length, fp); // read all of the bytes

								  // append a NULL character to indicate the end of the string
	buffer[length] = '\0';

	// close the file
	fclose(fp);

	// return the string
	return buffer;
}

// Create a GLSL program object from vertex and fragment shader files
GLuint InitShader(const char* vShaderFileName, const char* fShaderFileName) {
	GLuint vertex_shader, fragment_shader;
	GLchar *vs_text, *fs_text;
	GLuint program;

	// check GLSL version
	printf("GLSL version: %s\n\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	// Create shader handlers
	vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);

	// Read source code from shader files
	vs_text = readShaderSource(vShaderFileName);
	fs_text = readShaderSource(fShaderFileName);

	// error check
	if (vs_text == NULL) {
		printf("Failed to read from vertex shader file %s\n", vShaderFileName);
		exit(1);
	}
	else if (DEBUG_ON) {
		printf("Vertex Shader:\n=====================\n");
		printf("%s\n", vs_text);
		printf("=====================\n\n");
	}
	if (fs_text == NULL) {
		printf("Failed to read from fragent shader file %s\n", fShaderFileName);
		exit(1);
	}
	else if (DEBUG_ON) {
		printf("\nFragment Shader:\n=====================\n");
		printf("%s\n", fs_text);
		printf("=====================\n\n");
	}

	// Load Vertex Shader
	const char *vv = vs_text;
	glShaderSource(vertex_shader, 1, &vv, NULL);  //Read source
	glCompileShader(vertex_shader); // Compile shaders

									// Check for errors
	GLint  compiled;
	glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &compiled);
	if (!compiled) {
		printf("Vertex shader failed to compile:\n");
		if (DEBUG_ON) {
			GLint logMaxSize, logLength;
			glGetShaderiv(vertex_shader, GL_INFO_LOG_LENGTH, &logMaxSize);
			printf("printing error message of %d bytes\n", logMaxSize);
			char* logMsg = new char[logMaxSize];
			glGetShaderInfoLog(vertex_shader, logMaxSize, &logLength, logMsg);
			printf("%d bytes retrieved\n", logLength);
			printf("error message: %s\n", logMsg);
			delete[] logMsg;
		}
		exit(1);
	}

	// Load Fragment Shader
	const char *ff = fs_text;
	glShaderSource(fragment_shader, 1, &ff, NULL);
	glCompileShader(fragment_shader);
	glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &compiled);

	//Check for Errors
	if (!compiled) {
		printf("Fragment shader failed to compile\n");
		if (DEBUG_ON) {
			GLint logMaxSize, logLength;
			glGetShaderiv(fragment_shader, GL_INFO_LOG_LENGTH, &logMaxSize);
			printf("printing error message of %d bytes\n", logMaxSize);
			char* logMsg = new char[logMaxSize];
			glGetShaderInfoLog(fragment_shader, logMaxSize, &logLength, logMsg);
			printf("%d bytes retrieved\n", logLength);
			printf("error message: %s\n", logMsg);
			delete[] logMsg;
		}
		exit(1);
	}

	// Create the program
	program = glCreateProgram();

	// Attach shaders to program
	glAttachShader(program, vertex_shader);
	glAttachShader(program, fragment_shader);

	// Link and set program to use
	glLinkProgram(program);

	return program;
}
