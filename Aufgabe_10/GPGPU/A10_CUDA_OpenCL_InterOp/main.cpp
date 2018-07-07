// Versuch1a.cpp
// Ausgangssoftware des 1. Praktikumsversuchs 
// zur Vorlesung Echtzeit-3D-Computergrahpik
// von Prof. Dr. Alfred Nischwitz
// Programm umgesetzt mit der GLTools Library
#include <iostream>
#ifdef WIN32
#include <windows.h>
#endif
#include <GLTools.h>
#include <GLMatrixStack.h>
#include <GLGeometryTransform.h>
#include <GLFrustum.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <math.h>
#include <math3d.h>
#include <GL/freeglut.h>
#include <AntTweakBar.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "cudaKernel.h"

using namespace cv;
using namespace std;

GLShaderManager shaderManager;
GLMatrixStack modelViewMatrix;
GLMatrixStack projectionMatrix;
GLGeometryTransform transformPipeline;
GLFrustum viewFrustum;

GLBatch quadScreen;

GLuint tex = 0;         // OpenGL Identifier für Textur
GLuint tex_g = 0;
GLuint points_vbo = 0;  // OpenGL Identifier für Vertex Buffer Object

// Rotationsgroessen
static float rotation[] = { 0, 0, 0, 0 };
GLfloat vGreen[] = { 0.0f, 1.0f, 0.0f, 0.0f };

// Flags fuer Schalter
bool bSobel = false;
bool bHistogramm = false;
bool bGrayscale = false;

unsigned int width = 0;
unsigned int height = 0;


// FPS berechnen und in der Titelzeile des Fensters einblenden
void calcFPS() {
	static unsigned int lastTime = 0, actTime = 0;
	static unsigned int calls = 0;
	calls++;

	actTime = glutGet(GLUT_ELAPSED_TIME);
	int timeDiff = actTime - lastTime;
	if (timeDiff > 500) {
		char fps[256];
		sprintf(fps, "%3.1f fps", 1000.0 * calls / timeDiff);
		glutSetWindowTitle(fps);
		calls = 0;
		lastTime = actTime;
	}
}


//GUI
TwBar *bar;
void InitGUI()
{
	bar = TwNewBar("TweakBar");
	TwDefine(" TweakBar size='200 400'");
	TwAddVarRW(bar, "Grayscale", TW_TYPE_BOOLCPP, &bGrayscale, "");
	TwAddVarRW(bar, "Sobel", TW_TYPE_BOOLCPP, &bSobel, "");
	TwAddVarRW(bar, "Histogramm", TW_TYPE_BOOLCPP, &bHistogramm, "");
	//Hier weitere GUI Variablen anlegen. Für Farbe z.B. den Typ TW_TYPE_COLOR4F benutzen
}


void CreateGeometry(unsigned int width, unsigned int height)
{
	float w = width / 2.0f;
	float h = height / 2.0f;

	quadScreen.Begin(GL_TRIANGLE_STRIP, 4, 1);
	// Order: 2-3   gives the two ccw-triangles 0-1-2 and 2-1-3
	//        |\|
	//        0-1
	quadScreen.Color4f(1, 1, 1, 1);
	quadScreen.MultiTexCoord2f(0, 0.f, 1.f); quadScreen.Vertex3f(-w, -h, 0.0f);
	quadScreen.MultiTexCoord2f(0, 1.f, 1.f); quadScreen.Vertex3f( w, -h, 0.0f);
	quadScreen.MultiTexCoord2f(0, 0.f, 0.f); quadScreen.Vertex3f(-w,  h, 0.0f);
	quadScreen.MultiTexCoord2f(0, 1.f, 0.f); quadScreen.Vertex3f( w,  h, 0.0f);
	quadScreen.End();
}


// Aufruf draw scene
void RenderScene(void)
{

	if (cudaExecOneStep() != 0) {
		return;
	}

	// Clearbefehle für den color buffer und den depth buffer
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//Set correct texture
	if (bGrayscale || bSobel) {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex_g);
	}
	else {
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);
		displayColorImage();
	}

	// Sobel
	if (bGrayscale) {
		displayGrayscaleImage();
	}

	// Sobel
	if (bSobel){
		applySobelFilter();
	}

	// Histogramm
	if (bHistogramm){

	}

	// Speichere den matrix state und führe die Rotation durch
	modelViewMatrix.PushMatrix();
	M3DMatrix44f rot;
	m3dQuatToRotationMatrix(rot, rotation);
	modelViewMatrix.MultMatrix(rot);


	//setze den Shader für das Rendern und übergebe die Model-View-Projection Matrix
	shaderManager.UseStockShader(GLT_SHADER_TEXTURE_REPLACE, transformPipeline.GetModelViewProjectionMatrix(), 0);
	//Auf fehler überprüfen
	gltCheckErrors(0);

	quadScreen.Draw();

	//setze den Shader für das Rendern und übergebe die Model-View-Projection Matrix
	shaderManager.UseStockShader(GLT_SHADER_FLAT, transformPipeline.GetModelViewProjectionMatrix(), vGreen);
	//Auf fehler überprüfen
	gltCheckErrors(0);

	// Punkte aus dem von CUDA berechneten VBO zeichnen
	glPointSize(.0f);
	glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS, 0, 256);
	glDisableClientState(GL_VERTEX_ARRAY);
	glPointSize(1.0f);

	// Hole die im Stack gespeicherten Transformationsmatrizen wieder zurück
	modelViewMatrix.PopMatrix();

	TwDraw();
	// Vertausche Front- und Backbuffer
	glutSwapBuffers();
	glutPostRedisplay();

	calcFPS();

}

// Initialisierung des Rendering Kontextes
void SetupRC()
{
	// Schwarzer Hintergrund
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	// In Uhrzeigerrichtung zeigende Polygone sind die Vorderseiten.
	// Dies ist umgekehrt als bei der Default-Einstellung weil wir Triangle_Fans benützen
	glFrontFace(GL_CW);

	//initialisiert die standard shader
	shaderManager.InitializeStockShaders();
	//Matrix stacks für die Transformationspipeline setzen, damit werden dann automatisch die Matrizen multipliziert
	transformPipeline.SetMatrixStacks(modelViewMatrix, projectionMatrix);
	//erzeuge die geometrie
	InitGUI();
	
	// mit CUDA gemeinsam genutztes Vertex Buffer Object vorbereiten (CUDA/OpenGL Interoperabilität)
	// Speicher für die Initialisierung vorbereiten und an OpenGL zum Kopieren übergeben
	// X,Y,Z Koordinaten für 256 Punkte!
	GLfloat vPoints[256][3];
	for (int i = 0; i < 256; i++) {
		vPoints[i][0] = 1.0f;
		vPoints[i][1] = 1.0f;
		vPoints[i][2] = 0.0f;
	}

	glGenBuffers(1, &points_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, points_vbo);
	glBufferData(GL_ARRAY_BUFFER, 256 * 3 * sizeof(float), vPoints, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);



	// mit CUDA gemeinsam genutzte Textur vorbereiten (CUDA/OpenGL Interoperabilität)
	// Speicher für die Initialisierung vorbereiten und an OpenGL übergeben
	unsigned char *dataPtr = (unsigned char*)malloc(width*height);
	for (unsigned int i = 0; i < height; i++) {
		for (unsigned int j = 0; j < width; j++) {
			dataPtr[i*width + j] = j % 256;
		}
	}

	glGenTextures(1, &tex_g);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex_g);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	glGenTextures(1, &tex);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	free(dataPtr);


	cudaInit(tex, tex_g, points_vbo, width, height);
}


void SpecialKeys(int key, int x, int y)
{
	TwEventKeyboardGLUT(key, x, y);
	// Zeichne das Window neu
	glutPostRedisplay();
}


void ChangeSize(int w, int h)
{
	GLfloat nRange = 100.0f;

	// Verhindere eine Division durch Null
	if (h == 0)
		h = 1;
	// Setze den Viewport gemaess der Window-Groesse
	glViewport(0, 0, w, h);
	// Ruecksetzung des Projection matrix stack
	projectionMatrix.LoadIdentity();

	// Definiere das viewing volume (left, right, bottom, top, near, far)	
	viewFrustum.SetOrthographic(-w/2.0f, w/2.0f, -h/2.0f, h/2.0f, -nRange, nRange);
	projectionMatrix.LoadMatrix(viewFrustum.GetProjectionMatrix());
	// Ruecksetzung des Model view matrix stack
	modelViewMatrix.LoadIdentity();

	TwWindowSize(w, h);
}

void ShutDownRC()
{
	//GUI aufräumen
	TwTerminate();
}

int main(int argc, char* argv[])
{
	cudaGetOpenCVImageSize(width, height);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(width, height);
	glutCreateWindow("Aufgabe_10");
	glutCloseFunc(ShutDownRC);

	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		// Veralteter Treiber etc.
		std::cerr << "Error: " << glewGetErrorString(err) << "\n";
		return 1;
	}

	glutMouseFunc((GLUTmousebuttonfun)TwEventMouseButtonGLUT);
	glutMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT);
	glutPassiveMotionFunc((GLUTmousemotionfun)TwEventMouseMotionGLUT); // same as MouseMotion
	glutKeyboardFunc((GLUTkeyboardfun)TwEventKeyboardGLUT);

	glutReshapeFunc(ChangeSize);
	glutSpecialFunc(SpecialKeys);
	glutDisplayFunc(RenderScene);

	TwInit(TW_OPENGL_CORE, NULL);
	SetupRC();
	CreateGeometry(width, height);

	glutMainLoop();

	return 0;
}
