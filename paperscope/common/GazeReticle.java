package com.google.vr.cardboard.paperscope.demo.common;

import android.opengl.GLES20;
import android.opengl.Matrix;
import com.google.common.flogger.FluentLogger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Draws the gaze reticle: a dot plus a ring that fills clockwise while a dwell click is in
 * progress. Same look as the launcher reticle (gaze_ui.cc), drawn procedurally in a shader.
 */
public class GazeReticle {
  private static final FluentLogger logger = FluentLogger.forEnclosingClass();

  // Reticle size relative to its distance, so it keeps the same angular size.
  private static final float SIZE = 0.06f;

  // Unit quad in the XY plane facing +Z.
  private static final float[] QUAD_VERTICES = {
    -0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f,
  };

  private static final String VERTEX_SHADER =
      """
      uniform mat4 uMvpMatrix;
      attribute vec2 aPosition;
      varying vec2 vPos;
      void main() {
        vPos = aPosition;
        gl_Position = uMvpMatrix * vec4(aPosition, 0.0, 1.0);
      }
      """;

  // vPos is in [-0.5, 0.5].
  private static final String FRAGMENT_SHADER =
      """
      precision mediump float;
      uniform vec4 uColor;
      uniform float uProgress;
      varying vec2 vPos;

      float fill(float d) { return 1.0 - smoothstep(-0.01, 0.01, d); }

      void main() {
        float r = length(vPos);
        float center = fill(r - 0.1);
        float band = fill(abs(r - 0.4) - 0.06);
        float a = atan(vPos.x, vPos.y);
        float t = (a < 0.0 ? a + 6.2831853 : a) / 6.2831853;
        float shown = step(0.001, uProgress);
        float ring = band * step(t, uProgress) * shown;
        float track = band * shown * 0.3;
        float alpha = max(max(center, ring), track) * uColor.a;
        if (alpha < 0.01) {
          discard;
        }
        gl_FragColor = vec4(uColor.rgb, alpha);
      }
      """;

  private final FloatBuffer quad;
  private final float[] modelMatrix = new float[16];
  private final float[] tmpMatrix = new float[16];
  private final float[] mvpMatrix = new float[16];

  private int program;
  private int aPosition;
  private int uMvpMatrix;
  private int uColor;
  private int uProgress;

  public GazeReticle() {
    quad =
        ByteBuffer.allocateDirect(QUAD_VERTICES.length * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer();
    quad.put(QUAD_VERTICES).position(0);
  }

  /** Compiles the shader. Must be called on the GL thread. */
  public void init() {
    int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER);
    int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
    program = GLES20.glCreateProgram();
    GLES20.glAttachShader(program, vertexShader);
    GLES20.glAttachShader(program, fragmentShader);
    GLES20.glLinkProgram(program);
    int[] linkStatus = new int[1];
    GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, linkStatus, 0);
    if (linkStatus[0] != GLES20.GL_TRUE) {
      logger.atSevere().log("Could not link reticle: %s", GLES20.glGetProgramInfoLog(program));
      GLES20.glDeleteProgram(program);
      program = 0;
      return;
    }
    aPosition = GLES20.glGetAttribLocation(program, "aPosition");
    uMvpMatrix = GLES20.glGetUniformLocation(program, "uMvpMatrix");
    uColor = GLES20.glGetUniformLocation(program, "uColor");
    uProgress = GLES20.glGetUniformLocation(program, "uProgress");
  }

  /**
   * Draws the reticle straight ahead of the head, on top of the scene.
   *
   * @param eyeFromHead Transform from head space to this eye.
   * @param perspective Projection of this eye.
   * @param distance Distance of the gazed object, so both eyes converge on it.
   * @param progress Ring fill in [0, 1]; 0 hides the ring.
   */
  public void draw(float[] eyeFromHead, float[] perspective, float distance, float progress) {
    if (program == 0) {
      return;
    }
    float size = SIZE * distance;
    Matrix.setIdentityM(modelMatrix, 0);
    Matrix.translateM(modelMatrix, 0, 0, 0, -distance);
    Matrix.scaleM(modelMatrix, 0, size, size, 1f);
    Matrix.multiplyMM(tmpMatrix, 0, eyeFromHead, 0, modelMatrix, 0);
    Matrix.multiplyMM(mvpMatrix, 0, perspective, 0, tmpMatrix, 0);

    boolean depthTest = GLES20.glIsEnabled(GLES20.GL_DEPTH_TEST);
    boolean cullFace = GLES20.glIsEnabled(GLES20.GL_CULL_FACE);
    GLES20.glDisable(GLES20.GL_DEPTH_TEST);
    GLES20.glDisable(GLES20.GL_CULL_FACE);

    GLES20.glUseProgram(program);
    GLES20.glUniformMatrix4fv(uMvpMatrix, 1, false, mvpMatrix, 0);
    GLES20.glUniform4f(uColor, 1f, 1f, 1f, 1f);
    GLES20.glUniform1f(uProgress, Math.min(progress, 1f));

    // The quad comes from client memory, so no buffer may stay bound.
    GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0);
    quad.position(0);
    GLES20.glVertexAttribPointer(aPosition, 2, GLES20.GL_FLOAT, false, 0, quad);
    GLES20.glEnableVertexAttribArray(aPosition);
    GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
    GLES20.glDisableVertexAttribArray(aPosition);

    if (depthTest) {
      GLES20.glEnable(GLES20.GL_DEPTH_TEST);
    }
    if (cullFace) {
      GLES20.glEnable(GLES20.GL_CULL_FACE);
    }
  }

  public void release() {
    if (program != 0) {
      GLES20.glDeleteProgram(program);
      program = 0;
    }
  }

  private static int loadShader(int shaderType, String source) {
    int shader = GLES20.glCreateShader(shaderType);
    GLES20.glShaderSource(shader, source);
    GLES20.glCompileShader(shader);
    int[] compiled = new int[1];
    GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
    if (compiled[0] == 0) {
      logger.atSevere().log("Could not compile reticle shader: %s", GLES20.glGetShaderInfoLog(shader));
    }
    return shader;
  }
}
