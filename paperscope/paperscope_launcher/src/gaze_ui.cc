#include "gaze_ui.h"

#include <android/log.h>

namespace cardboard_launcher {

namespace {

constexpr float kIconReticle{0.f};
constexpr float kIconMagnet{1.f};

// Unit quad in the XY plane facing +Z, counter-clockwise.
constexpr GLfloat kQuadVertices[]{-0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f,
                                  -0.5f, -0.5f, 0.5f, 0.5f,  -0.5f, 0.5f};

constexpr const char* kGazeVertexShader = R"glsl(
    uniform mat4 u_MVP;
    attribute vec2 a_Position;
    varying vec2 v_Pos;

    void main() {
      v_Pos = a_Position;
      gl_Position = u_MVP * vec4(a_Position, 0.0, 1.0);
    })glsl";

// v_Pos is in [-0.5, 0.5].
constexpr const char* kGazeFragmentShader = R"glsl(
    precision mediump float;

    uniform vec4 u_Color;
    uniform float u_Icon;
    uniform float u_Progress;
    varying vec2 v_Pos;

    // Signed distance to a rounded box with half size b and corner radius r.
    float box(vec2 p, vec2 b, float r) {
      vec2 q = abs(p) - b + r;
      return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
    }

    float fill(float d) { return 1.0 - smoothstep(-0.01, 0.01, d); }

    void main() {
      vec2 p = v_Pos;
      vec3 color = u_Color.rgb;
      float alpha = 0.0;
      if (u_Icon < 0.5) {
        // Reticle: dot plus a progress ring that fills clockwise from the top.
        float r = length(p);
        float center = fill(r - 0.1);
        float band = fill(abs(r - 0.4) - 0.06);
        float a = atan(p.x, p.y);
        float t = (a < 0.0 ? a + 6.2831853 : a) / 6.2831853;
        float shown = step(0.001, u_Progress);
        float ring = band * step(t, u_Progress) * shown;
        float track = band * shown * 0.3;
        alpha = max(max(center, ring), track) * u_Color.a;
      } else {
        // Horseshoe magnet opening upwards, with grey tips.
        float bg = fill(box(p, vec2(0.48), 0.12));
        vec2 q = p - vec2(0.0, -0.05);
        float r = length(q);
        float arc = fill(abs(r - 0.2) - 0.07) * step(q.y, 0.0);
        float legs = fill(box(q - vec2(0.0, 0.1), vec2(0.27, 0.1), 0.0)) * step(0.13, abs(q.x));
        float tips = fill(box(q - vec2(0.0, 0.22), vec2(0.27, 0.04), 0.0)) * step(0.13, abs(q.x));
        float icon = max(max(arc, legs), tips);
        vec3 icon_color = mix(vec3(0.9, 0.15, 0.15), vec3(0.85), tips);
        color = mix(u_Color.rgb, icon_color, icon);
        alpha = bg * u_Color.a;
      }
      if (alpha < 0.01) {
        discard;
      }
      gl_FragColor = vec4(color, alpha);
    })glsl";

}  // anonymous namespace

void GazeUi::Init() {
  const GLuint vertex_shader =
      cardboard_demo::LoadGLShader(GL_VERTEX_SHADER, kGazeVertexShader);
  const GLuint fragment_shader =
      cardboard_demo::LoadGLShader(GL_FRAGMENT_SHADER, kGazeFragmentShader);

  program_ = glCreateProgram();
  glAttachShader(program_, vertex_shader);
  glAttachShader(program_, fragment_shader);
  glLinkProgram(program_);
  cardboard_demo::CHECKGLERROR("GazeUi program");

  position_param_ = glGetAttribLocation(program_, "a_Position");
  mvp_param_ = glGetUniformLocation(program_, "u_MVP");
  color_param_ = glGetUniformLocation(program_, "u_Color");
  icon_param_ = glGetUniformLocation(program_, "u_Icon");
  progress_param_ = glGetUniformLocation(program_, "u_Progress");
  cardboard_demo::CHECKGLERROR("GazeUi program params");
}

void GazeUi::DrawReticle(const cardboard_demo::Matrix4x4& mvp,
                         const std::array<float, 4>& color,
                         float progress) const {
  DrawQuad(mvp, color, kIconReticle, progress);
}

void GazeUi::DrawMagnet(const cardboard_demo::Matrix4x4& mvp,
                        const std::array<float, 4>& color) const {
  DrawQuad(mvp, color, kIconMagnet, 0.f);
}

void GazeUi::DrawQuad(const cardboard_demo::Matrix4x4& mvp,
                      const std::array<float, 4>& color, float icon,
                      float progress) const {
  glUseProgram(program_);

  cardboard_demo::Matrix4x4 mvp_copy{mvp};
  const std::array<float, 16> mvp_array{mvp_copy.ToGlArray()};
  glUniformMatrix4fv(mvp_param_, 1, GL_FALSE, mvp_array.data());
  glUniform4f(color_param_, color[0], color[1], color[2], color[3]);
  glUniform1f(icon_param_, icon);
  glUniform1f(progress_param_, progress);

  // The quad comes from client memory, so no buffer may stay bound.
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glEnableVertexAttribArray(position_param_);
  glVertexAttribPointer(position_param_, 2, GL_FLOAT, GL_FALSE, 0,
                        kQuadVertices);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glDisableVertexAttribArray(position_param_);

  cardboard_demo::CHECKGLERROR("GazeUi draw");
}

}  // namespace cardboard_launcher
