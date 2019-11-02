package ogl_test.ogl_test_java;

import java.nio.FloatBuffer;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opengl.DefaultGLCapabilitiesChooser;
import com.jogamp.opengl.GL;
import com.jogamp.opengl.GL2;
import com.jogamp.opengl.GL2ES2;
import com.jogamp.opengl.GL4;
import com.jogamp.opengl.GLAutoDrawable;
import com.jogamp.opengl.GLCapabilities;
import com.jogamp.opengl.GLContext;
import com.jogamp.opengl.GLDrawableFactory;
import com.jogamp.opengl.GLProfile;

/*
矩阵乘法简单示例：arrayA * arrayB
arrayA = [
            1 2
            3 4
         ] shape: 2行2列
arrayB = [
            1 2 3 4
            1 2 3 4
         ] shape: 2行4列
         
arrayC = ( 2 x 2) * (2 x 4) == 2 x 4
 */

public class Sgemm {
	
	public static String shaderSource = 
	"#version 450\n" + 
	"layout (binding = 0) readonly buffer bottom_blob1 { float A[]; };\n" +
	"layout (binding = 1) readonly buffer bottom_blob2 { float B[]; };\n" + 
	"layout (binding = 2)  buffer top_blob { float outputs[]; };\n" + 
	"layout (location = 3) uniform ivec2 A_size;\n" + 
	"layout (location = 4) uniform ivec2 B_size;\n" + 
	"layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n" + 
	"void main()\n" + 
	"{\n" + 
		"int ax = A_size.x;\n" + 
		"int ay = A_size.y;\n" + 
		"int bx = B_size.x;\n" + 
		"int by = B_size.y;\n" + 
		"int gx = int(gl_GlobalInvocationID.x);\n" + 
		"int gy = int(gl_GlobalInvocationID.y);\n" + 
		"int gz = int(gl_GlobalInvocationID.z);\n" + 
		"float sum = float(0.0f);\n" + 
		"int output_offset = B_size.y * gx + gy;\n" + 
		"for(int i = 0;i < A_size.y; i++)\n" + 
		"{\n" + 
			"sum += A[i + gx * A_size.y] * B[i * B_size.y + gy];\n" + 
		"}\n" + 
		"outputs[output_offset] = float(sum) + outputs[output_offset];\n" + 
	"}\n";

	int createAndCompileShader(GL4 gl, int type, String shaderString) {
		int shader = gl.glCreateShader(type);

		String[] vlines = new String[] { shaderString };
		int[] vlengths = new int[] { vlines[0].length() };

		gl.glShaderSource(shader, vlines.length, vlines, vlengths, 0);
		gl.glCompileShader(shader);

		int[] compiled = new int[1];
		gl.glGetShaderiv(shader, GL2ES2.GL_COMPILE_STATUS, compiled, 0);

		if (compiled[0] == 0) {
			int[] logLength = new int[1];
			gl.glGetShaderiv(shader, GL2ES2.GL_INFO_LOG_LENGTH, logLength, 0);

			byte[] log = new byte[logLength[0]];
			gl.glGetShaderInfoLog(shader, logLength[0], (int[]) null, 0, log, 0);

			throw new IllegalStateException("Error compiling the shader: " + new String(log));
		}
		return shader;
	}

	float[] sgemm(int ax, int ay, int bx, int by, float arrayA[], float arrayB[]) {
		
		if (ay != bx) {
			System.err.println("sgemm error : the value of A_y must be equal to that of B_x!");
			System.exit(0);
		}
		int outputSize = ax * by;
		long startTime =  System.currentTimeMillis();
		// 离屏渲染
		GLAutoDrawable drawable;
//		final GLProfile glp = GLProfile.getDefault(GLProfile.getDefaultDevice());
		GLProfile glp = GLProfile.getDefault();	// <----速度贼慢
		GLDrawableFactory factory = GLDrawableFactory.getFactory(glp);
		GLCapabilities caps = new GLCapabilities(glp);
		drawable = factory.createOffscreenAutoDrawable(factory.getDefaultDevice(), caps,
				new DefaultGLCapabilitiesChooser(), 1,1);
		drawable.display();
		drawable.getContext().makeCurrent();
		GL4 gl = drawable.getGL().getGL4();
		
		
		long endTime =  System.currentTimeMillis();
		System.out.println("cost time:" + (endTime - startTime) + "ms");
		
		// ---------------------------------离屏渲染结束 --------------------------
		
		int computeShader = createAndCompileShader(gl, GL4.GL_COMPUTE_SHADER, shaderSource);
		
		int program = 0;
		program = gl.glCreateProgram();
		gl.glAttachShader(program, computeShader);
		gl.glLinkProgram(program);
		gl.glUseProgram(program);
		
		// 加载 SSBO A 数据
		int[] handle_a = new int[1];
		gl.glGenBuffers(1, handle_a, 0);
		gl.glBindBuffer(GL4.GL_SHADER_STORAGE_BUFFER, handle_a[0]);
		FloatBuffer bufferA = Buffers.newDirectFloatBuffer(arrayA);
		int numBytesA = arrayA.length * 4;// float : 1 = 4byte
		gl.glBufferData(GL4.GL_SHADER_STORAGE_BUFFER, numBytesA, bufferA, GL.GL_STATIC_DRAW);
		gl.glBindBufferBase(GL4.GL_SHADER_STORAGE_BUFFER, 0, handle_a[0]); // 2nd prama is binding
		
		// 加载 SSBO B 数据
		int[] handle_b = new int[1];
		gl.glGenBuffers(1, handle_b, 0);
		gl.glBindBuffer(GL4.GL_SHADER_STORAGE_BUFFER, handle_b[0]);
		FloatBuffer bufferB = Buffers.newDirectFloatBuffer(arrayB);
		int numBytesB = arrayB.length * 4;
		gl.glBufferData(GL4.GL_SHADER_STORAGE_BUFFER, numBytesB, bufferB, GL.GL_STATIC_DRAW);
		gl.glBindBufferBase(GL4.GL_SHADER_STORAGE_BUFFER, 1, handle_b[0]);
		
		// 加载 outputs SSBO 数据
		float[] arrayC = new float[outputSize];
		int[] handle_out = new int[1];
		gl.glGenBuffers(1, handle_out, 0);
		gl.glBindBuffer(GL4.GL_SHADER_STORAGE_BUFFER, handle_out[0]);
		FloatBuffer buffer_out = Buffers.newDirectFloatBuffer(arrayC);
		int numBytes_out = arrayC.length * 4;
		gl.glBufferData(GL4.GL_SHADER_STORAGE_BUFFER, numBytes_out, buffer_out, GL.GL_STATIC_DRAW);
		gl.glBindBufferBase(GL4.GL_SHADER_STORAGE_BUFFER, 2, handle_out[0]);
		
		gl.glUniform2i(3,ax,ay);
		gl.glUniform2i(4,bx,by);
		
		gl.glDispatchCompute(ax, by, 1);
		gl.glMemoryBarrier(GL4.GL_SHADER_STORAGE_BARRIER_BIT);
		
		gl.glBindBuffer(GL4.GL_SHADER_STORAGE_BUFFER, handle_out[0]);
		FloatBuffer floatbuffer = gl.glMapBuffer(GL4.GL_SHADER_STORAGE_BUFFER,GL2.GL_READ_ONLY).asFloatBuffer();
		
		// 将结果写反进arrayC中
		floatbuffer.get(arrayC);
		// 取消缓冲区映射
		gl.glUnmapBuffer(GL4.GL_SHADER_STORAGE_BUFFER);
				
		// 销毁 SSBO shader program 
		gl.glDeleteBuffers(1, Buffers.newDirectIntBuffer(handle_a));
		gl.glDeleteBuffers(1, Buffers.newDirectIntBuffer(handle_b));
		gl.glDeleteBuffers(1, Buffers.newDirectIntBuffer(handle_out));
		
		gl.glDetachShader(program, computeShader);
		gl.glDeleteShader(computeShader);
		gl.glDeleteProgram(program);

		return arrayC;
	}

	public static void main(String[] args) {
		
		Sgemm sgemm = new Sgemm();
		float[] arrayA = new float[] { 1, 2, 3, 4 };
		float[] arrayB = new float[] { 1, 2, 3, 4, 1, 2, 3, 4 };
		float[] reslut = new float[8];
		reslut = sgemm.sgemm(2, 2, 2, 4, arrayA, arrayB);
		
		System.out.println("outputs result:");
		for (int i = 0; i < reslut.length; i++) {
			System.out.print(reslut[i] + " ");
		}
		System.out.println();
		
		
	}

}
