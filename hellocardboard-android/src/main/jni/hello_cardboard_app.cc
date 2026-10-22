/*
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hello_cardboard_app.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>

#include "cardboard.h"

namespace ndk_hello_cardboard {

namespace {

// The objects are about 1 meter in radius, so the min/max target distance are
// set so that the objects are always within the room (which is about 5 meters
// across) and the reticle is always closer than any objects.
constexpr float kMinTargetDistance = 2.5f;
constexpr float kMaxTargetDistance = 3.5f;
constexpr float kMinTargetHeight = 0.5f;
constexpr float kMaxTargetHeight = kMinTargetHeight + 3.0f;

constexpr float kDefaultFloorHeight = -1.7f;

// 6 Hz cutoff frequency for the velocity filter of the head tracker.
constexpr int kVelocityFilterCutoffFrequency = 6;

constexpr uint64_t kPredictionTimeWithoutVsyncNanos = 50000000;

// Angle threshold for determining whether the controller is pointing at the
// object.
constexpr float kAngleLimit = 0.2f;

// Number of different possible targets
constexpr int kTargetMeshCount = 3;

// Gaze interaction timings. Looking at an item for kDwellSeconds activates it.
// A grabbed panel is released when the head stays still for kDetachSeconds
// (angular speed below kStillSpeed, rad/s) after having turned at least
// kMinDragAngle radians since the grab.
constexpr float kDwellSeconds = 1.5f;
constexpr float kDetachSeconds = 1.5f;
constexpr float kStillSpeed = 0.2f;
constexpr float kSpeedSmoothing = 0.15f;
constexpr float kMinDragAngle = 0.17f;
constexpr float kMaxFrameSeconds = 0.1f;

// Panel layout, meters. The panel is placed below the line of sight so it does
// not cover the scene; the magnet sticks out of its top edge.
constexpr float kPanelDistance = 1.6f;
constexpr float kPanelDrop = -0.3f;
constexpr float kPanelWidth = 0.6f;
constexpr float kPanelHeight = 0.26f;
constexpr float kButtonX = 0.14f;
constexpr float kButtonY = 0.0f;
constexpr float kButtonHalf = 0.09f;
constexpr float kMagnetY = 0.2f;
constexpr float kMagnetHalf = 0.07f;
constexpr float kElementZ = 0.01f;
// Hit area is a bit larger than the drawn element.
constexpr float kGazeTolerance = 1.2f;

// Reticle size relative to its distance, so it keeps the same angular size.
constexpr float kReticleSize = 0.06f;
constexpr float kDefaultReticleDistance = 2.0f;

// Icon ids understood by kUiFragmentShader.
constexpr float kIconPanel = 0.f;
constexpr float kIconNewTarget = 1.f;
constexpr float kIconRecenter = 2.f;
constexpr float kIconMagnet = 3.f;
constexpr float kIconReticle = 4.f;

constexpr std::array<float, 4> kPanelColor = {0.08f, 0.09f, 0.12f, 0.8f};
constexpr std::array<float, 4> kButtonColor = {0.2f, 0.4f, 0.75f, 1.f};
constexpr std::array<float, 4> kButtonGazedColor = {0.3f, 0.6f, 1.f, 1.f};
constexpr std::array<float, 4> kMagnetColor = {0.25f, 0.25f, 0.3f, 1.f};
constexpr std::array<float, 4> kMagnetGazedColor = {0.4f, 0.4f, 0.48f, 1.f};
constexpr std::array<float, 4> kMagnetAttachedColor = {0.85f, 0.55f, 0.1f, 1.f};
constexpr std::array<float, 4> kReticleColor = {1.f, 1.f, 1.f, 1.f};
constexpr std::array<float, 4> kReticleDetachColor = {1.f, 0.65f, 0.15f, 1.f};

// Unit quad in the XY plane facing +Z, counter-clockwise.
constexpr GLfloat kQuadVertices[] = {-0.5f, -0.5f, 0.5f, -0.5f, 0.5f, 0.5f,
                                     -0.5f, -0.5f, 0.5f, 0.5f,  -0.5f, 0.5f};

// UI shader: draws the panel, its buttons and the reticle procedurally, so no
// textures are needed. v_Pos is in [-0.5, 0.5].
constexpr const char* kUiVertexShader =
    R"glsl(
    uniform mat4 u_MVP;
    attribute vec2 a_Position;
    varying vec2 v_Pos;

    void main() {
      v_Pos = a_Position;
      gl_Position = u_MVP * vec4(a_Position, 0.0, 1.0);
    })glsl";

constexpr const char* kUiFragmentShader =
    R"glsl(
    precision mediump float;

    uniform vec4 u_Color;
    uniform float u_Icon;
    uniform vec2 u_Size;
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
        // Panel background, rounded in meters.
        vec2 pm = p * u_Size;
        float d = box(pm, u_Size * 0.5, 0.04);
        alpha = (1.0 - smoothstep(-0.004, 0.004, d)) * u_Color.a;
      } else if (u_Icon < 4.5 && u_Icon > 3.5) {
        // Reticle: dot plus a progress ring that fills clockwise from the top.
        float r = length(p);
        float center = fill(r - 0.1);
        float band = fill(abs(r - 0.4) - 0.06);
        float a = atan(p.x, p.y);
        float t = (a < 0.0 ? a + 6.2831853 : a) / 6.2831853;
        float shown = step(0.001, u_Progress);
        float ring = band * step(t, u_Progress) * shown;
        float track = band * shown * 0.3;
        alpha = max(max(center, ring), track);
      } else {
        float bg = fill(box(p, vec2(0.48), 0.12));
        float icon = 0.0;
        vec3 icon_color = vec3(1.0);
        if (u_Icon < 1.5) {
          // Plus: new target.
          icon = max(fill(box(p, vec2(0.25, 0.05), 0.02)),
                     fill(box(p, vec2(0.05, 0.25), 0.02)));
        } else if (u_Icon < 2.5) {
          // Crosshair: recenter.
          float r = length(p);
          float ring = fill(abs(r - 0.22) - 0.04);
          float center = fill(r - 0.07);
          float ticks = max(fill(box(p, vec2(0.36, 0.03), 0.0)) * step(0.22, abs(p.x)),
                            fill(box(p, vec2(0.03, 0.36), 0.0)) * step(0.22, abs(p.y)));
          icon = max(max(ring, center), ticks);
        } else {
          // Horseshoe magnet opening upwards, with grey tips.
          vec2 q = p - vec2(0.0, -0.05);
          float r = length(q);
          float arc = fill(abs(r - 0.2) - 0.07) * step(q.y, 0.0);
          float legs = fill(box(q - vec2(0.0, 0.1), vec2(0.27, 0.1), 0.0)) * step(0.13, abs(q.x));
          float tips = fill(box(q - vec2(0.0, 0.22), vec2(0.27, 0.04), 0.0)) * step(0.13, abs(q.x));
          icon = max(max(arc, legs), tips);
          icon_color = mix(vec3(0.9, 0.15, 0.15), vec3(0.85), tips);
        }
        color = mix(u_Color.rgb, icon_color, icon);
        alpha = bg * u_Color.a;
      }
      if (alpha < 0.01) {
        discard;
      }
      gl_FragColor = vec4(color, alpha);
    })glsl";

Matrix4x4 GetScaleMatrix(float sx, float sy, float sz) {
  return {{{sx, 0.0f, 0.0f, 0.0f},
           {0.0f, sy, 0.0f, 0.0f},
           {0.0f, 0.0f, sz, 0.0f},
           {0.0f, 0.0f, 0.0f, 1.0f}}};
}

// Inverts a rotation + translation matrix (column-major, m[column][row]).
Matrix4x4 GetRigidInverse(const Matrix4x4& mat) {
  Matrix4x4 inv;
  for (int c = 0; c < 3; ++c) {
    for (int r = 0; r < 3; ++r) {
      inv.m[c][r] = mat.m[r][c];
    }
    inv.m[c][3] = 0.0f;
  }
  for (int r = 0; r < 3; ++r) {
    inv.m[3][r] = -(mat.m[r][0] * mat.m[3][0] + mat.m[r][1] * mat.m[3][1] +
                    mat.m[r][2] * mat.m[3][2]);
  }
  inv.m[3][3] = 1.0f;
  return inv;
}

// Simple shaders to render .obj files without any lighting.
constexpr const char* kObjVertexShader =
    R"glsl(
    uniform mat4 u_MVP;
    attribute vec4 a_Position;
    attribute vec2 a_UV;
    varying vec2 v_UV;

    void main() {
      v_UV = a_UV;
      gl_Position = u_MVP * a_Position;
    })glsl";

constexpr const char* kObjFragmentShader =
    R"glsl(
    precision mediump float;

    uniform sampler2D u_Texture;
    varying vec2 v_UV;

    void main() {
      // The y coordinate of this sample's textures is reversed compared to
      // what OpenGL expects, so we invert the y coordinate.
      gl_FragColor = texture2D(u_Texture, vec2(v_UV.x, 1.0 - v_UV.y));
    })glsl";

}  // anonymous namespace

HelloCardboardApp::HelloCardboardApp(JavaVM* vm, jobject obj,
                                     jobject asset_mgr_obj)
    : head_tracker_(nullptr),
      lens_distortion_(nullptr),
      distortion_renderer_(nullptr),
      screen_params_changed_(false),
      device_params_changed_(false),
      screen_width_(0),
      screen_height_(0),
      depthRenderBuffer_(0),
      framebuffer_(0),
      texture_(0),
      obj_program_(0),
      obj_position_param_(0),
      obj_uv_param_(0),
      obj_modelview_projection_param_(0),
      target_object_meshes_(kTargetMeshCount),
      target_object_not_selected_textures_(kTargetMeshCount),
      target_object_selected_textures_(kTargetMeshCount),
      cur_target_object_(RandomUniformInt(kTargetMeshCount)),
      ui_program_(0),
      ui_position_param_(0),
      ui_mvp_param_(0),
      ui_color_param_(0),
      ui_icon_param_(0),
      ui_size_param_(0),
      ui_progress_param_(0),
      panel_placed_(false),
      panel_attached_(false),
      gaze_item_(GazeItem::kNone),
      gaze_distance_(kDefaultReticleDistance),
      dwell_time_(0.f),
      dwell_fired_(false),
      grab_forward_({0.f, 0.f, -1.f, 0.f}),
      prev_forward_({0.f, 0.f, -1.f, 0.f}),
      drag_moved_(false),
      head_speed_(0.f),
      still_time_(0.f),
      last_frame_time_(0) {
  JNIEnv* env;
  vm->GetEnv((void**)&env, JNI_VERSION_1_6);
  java_asset_mgr_ = env->NewGlobalRef(asset_mgr_obj);
  asset_mgr_ = AAssetManager_fromJava(env, asset_mgr_obj);

  Cardboard_initializeAndroid(vm, obj);
  head_tracker_ = CardboardHeadTracker_create();
  CardboardHeadTracker_setLowPassFilter(head_tracker_,
                                        kVelocityFilterCutoffFrequency);
}

HelloCardboardApp::~HelloCardboardApp() {
  CardboardHeadTracker_destroy(head_tracker_);
  CardboardLensDistortion_destroy(lens_distortion_);
  CardboardDistortionRenderer_destroy(distortion_renderer_);
}

void HelloCardboardApp::OnSurfaceCreated(JNIEnv* env) {
  const int obj_vertex_shader =
      LoadGLShader(GL_VERTEX_SHADER, kObjVertexShader);
  const int obj_fragment_shader =
      LoadGLShader(GL_FRAGMENT_SHADER, kObjFragmentShader);

  obj_program_ = glCreateProgram();
  glAttachShader(obj_program_, obj_vertex_shader);
  glAttachShader(obj_program_, obj_fragment_shader);
  glLinkProgram(obj_program_);
  glUseProgram(obj_program_);

  CHECKGLERROR("Obj program");

  obj_position_param_ = glGetAttribLocation(obj_program_, "a_Position");
  obj_uv_param_ = glGetAttribLocation(obj_program_, "a_UV");
  obj_modelview_projection_param_ = glGetUniformLocation(obj_program_, "u_MVP");

  CHECKGLERROR("Obj program params");

  HELLOCARDBOARD_CHECK(room_.Initialize(obj_position_param_, obj_uv_param_,
                                        "CubeRoom.obj", asset_mgr_));
  HELLOCARDBOARD_CHECK(
      room_tex_.Initialize(env, java_asset_mgr_, "CubeRoom_BakedDiffuse.png"));
  HELLOCARDBOARD_CHECK(target_object_meshes_[0].Initialize(
      obj_position_param_, obj_uv_param_, "Icosahedron.obj", asset_mgr_));
  HELLOCARDBOARD_CHECK(target_object_not_selected_textures_[0].Initialize(
      env, java_asset_mgr_, "Icosahedron_Blue_BakedDiffuse.png"));
  HELLOCARDBOARD_CHECK(target_object_selected_textures_[0].Initialize(
      env, java_asset_mgr_, "Icosahedron_Pink_BakedDiffuse.png"));
  HELLOCARDBOARD_CHECK(target_object_meshes_[1].Initialize(
      obj_position_param_, obj_uv_param_, "QuadSphere.obj", asset_mgr_));
  HELLOCARDBOARD_CHECK(target_object_not_selected_textures_[1].Initialize(
      env, java_asset_mgr_, "QuadSphere_Blue_BakedDiffuse.png"));
  HELLOCARDBOARD_CHECK(target_object_selected_textures_[1].Initialize(
      env, java_asset_mgr_, "QuadSphere_Pink_BakedDiffuse.png"));
  HELLOCARDBOARD_CHECK(target_object_meshes_[2].Initialize(
      obj_position_param_, obj_uv_param_, "TriSphere.obj", asset_mgr_));
  HELLOCARDBOARD_CHECK(target_object_not_selected_textures_[2].Initialize(
      env, java_asset_mgr_, "TriSphere_Blue_BakedDiffuse.png"));
  HELLOCARDBOARD_CHECK(target_object_selected_textures_[2].Initialize(
      env, java_asset_mgr_, "TriSphere_Pink_BakedDiffuse.png"));

  const int ui_vertex_shader = LoadGLShader(GL_VERTEX_SHADER, kUiVertexShader);
  const int ui_fragment_shader =
      LoadGLShader(GL_FRAGMENT_SHADER, kUiFragmentShader);

  ui_program_ = glCreateProgram();
  glAttachShader(ui_program_, ui_vertex_shader);
  glAttachShader(ui_program_, ui_fragment_shader);
  glLinkProgram(ui_program_);

  CHECKGLERROR("Ui program");

  ui_position_param_ = glGetAttribLocation(ui_program_, "a_Position");
  ui_mvp_param_ = glGetUniformLocation(ui_program_, "u_MVP");
  ui_color_param_ = glGetUniformLocation(ui_program_, "u_Color");
  ui_icon_param_ = glGetUniformLocation(ui_program_, "u_Icon");
  ui_size_param_ = glGetUniformLocation(ui_program_, "u_Size");
  ui_progress_param_ = glGetUniformLocation(ui_program_, "u_Progress");

  CHECKGLERROR("Ui program params");

  // Target object first appears directly in front of user.
  model_target_ = GetTranslationMatrix({0.0f, 1.5f, kMinTargetDistance});

  CHECKGLERROR("OnSurfaceCreated");
}

void HelloCardboardApp::SetScreenParams(int width, int height) {
  screen_width_ = width;
  screen_height_ = height;
  screen_params_changed_ = true;
}

void HelloCardboardApp::OnDrawFrame() {
  if (!UpdateDeviceParams()) {
    return;
  }

  // Update Head Pose.
  head_view_ = GetPose();

  // Incorporate the floor height into the head_view
  head_view_ =
      head_view_ * GetTranslationMatrix({0.0f, kDefaultFloorHeight, 0.0f});

  const int64_t now = GetBootTimeNano();
  const float dt =
      last_frame_time_ == 0
          ? 0.f
          : std::min((now - last_frame_time_) * 1e-9f, kMaxFrameSeconds);
  last_frame_time_ = now;
  UpdateInteraction(dt);

  // Bind buffer
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glDisable(GL_SCISSOR_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Draw eyes views
  for (int eye = 0; eye < 2; ++eye) {
    glViewport(eye == kLeft ? 0 : screen_width_ / 2, 0, screen_width_ / 2,
               screen_height_);

    Matrix4x4 eye_matrix = GetMatrixFromGlArray(eye_matrices_[eye]);
    Matrix4x4 eye_view = eye_matrix * head_view_;

    Matrix4x4 projection_matrix =
        GetMatrixFromGlArray(projection_matrices_[eye]);
    Matrix4x4 modelview_target = eye_view * model_target_;
    modelview_projection_target_ = projection_matrix * modelview_target;
    modelview_projection_room_ = projection_matrix * eye_view;

    // Draw room and target
    DrawWorld();
    DrawPanel(eye_view, projection_matrix);
    DrawReticle(eye_matrix, projection_matrix);
  }

  // Render
  CardboardDistortionRenderer_renderEyeToDisplay(
      distortion_renderer_, /* target_display = */ 0, /* x = */ 0, /* y = */ 0,
      screen_width_, screen_height_, &left_eye_texture_description_,
      &right_eye_texture_description_);

  CHECKGLERROR("onDrawFrame");
}

void HelloCardboardApp::OnTriggerEvent() {
  // The trigger is an instant click on whatever the reticle is on. While the
  // panel is attached it releases the panel instead.
  if (panel_attached_) {
    DetachPanel();
  } else {
    Activate(gaze_item_);
  }
  // Do not fire the dwell on the same item again until the gaze leaves it.
  dwell_fired_ = true;
}

void HelloCardboardApp::OnPause() { CardboardHeadTracker_pause(head_tracker_); }

void HelloCardboardApp::OnResume() {
  CardboardHeadTracker_resume(head_tracker_);

  // Parameters may have changed.
  device_params_changed_ = true;

  // Check for device parameters existence in external storage. If they're
  // missing, we must scan a Cardboard QR code and save the obtained parameters.
  uint8_t* buffer;
  int size;
  CardboardQrCode_getSavedDeviceParams(&buffer, &size);
  if (size == 0) {
    SwitchViewer();
  }
  CardboardQrCode_destroy(buffer);
}

void HelloCardboardApp::SwitchViewer() {
  CardboardQrCode_scanQrCodeAndSaveDeviceParams();
}

bool HelloCardboardApp::UpdateDeviceParams() {
  // Checks if screen or device parameters changed
  if (!screen_params_changed_ && !device_params_changed_) {
    return true;
  }

  // Get saved device parameters
  uint8_t* buffer;
  int size;
  CardboardQrCode_getSavedDeviceParams(&buffer, &size);

  // If there are no parameters saved yet, returns false.
  if (size == 0) {
    return false;
  }

  CardboardLensDistortion_destroy(lens_distortion_);
  lens_distortion_ = CardboardLensDistortion_create(buffer, size, screen_width_,
                                                    screen_height_);

  CardboardQrCode_destroy(buffer);

  GlSetup();

  CardboardDistortionRenderer_destroy(distortion_renderer_);
  const CardboardOpenGlEsDistortionRendererConfig config{kGlTexture2D};
  distortion_renderer_ = CardboardOpenGlEs2DistortionRenderer_create(&config);

  CardboardMesh left_mesh;
  CardboardMesh right_mesh;
  CardboardLensDistortion_getDistortionMesh(lens_distortion_, kLeft,
                                            &left_mesh);
  CardboardLensDistortion_getDistortionMesh(lens_distortion_, kRight,
                                            &right_mesh);

  CardboardDistortionRenderer_setMesh(distortion_renderer_, &left_mesh, kLeft);
  CardboardDistortionRenderer_setMesh(distortion_renderer_, &right_mesh,
                                      kRight);

  // Get eye matrices
  CardboardLensDistortion_getEyeFromHeadMatrix(lens_distortion_, kLeft,
                                               eye_matrices_[0]);
  CardboardLensDistortion_getEyeFromHeadMatrix(lens_distortion_, kRight,
                                               eye_matrices_[1]);
  CardboardLensDistortion_getProjectionMatrix(lens_distortion_, kLeft, kZNear,
                                              kZFar, projection_matrices_[0]);
  CardboardLensDistortion_getProjectionMatrix(lens_distortion_, kRight, kZNear,
                                              kZFar, projection_matrices_[1]);

  screen_params_changed_ = false;
  device_params_changed_ = false;

  CHECKGLERROR("UpdateDeviceParams");

  return true;
}

void HelloCardboardApp::GlSetup() {
  LOGD("GL SETUP");

  if (framebuffer_ != 0) {
    GlTeardown();
  }

  // Create render texture.
  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, screen_width_, screen_height_, 0,
               GL_RGB, GL_UNSIGNED_BYTE, 0);

  left_eye_texture_description_.texture = texture_;
  left_eye_texture_description_.left_u = 0;
  left_eye_texture_description_.right_u = 0.5;
  left_eye_texture_description_.top_v = 1;
  left_eye_texture_description_.bottom_v = 0;

  right_eye_texture_description_.texture = texture_;
  right_eye_texture_description_.left_u = 0.5;
  right_eye_texture_description_.right_u = 1;
  right_eye_texture_description_.top_v = 1;
  right_eye_texture_description_.bottom_v = 0;

  // Generate depth buffer to perform depth test.
  glGenRenderbuffers(1, &depthRenderBuffer_);
  glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBuffer_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, screen_width_,
                        screen_height_);
  CHECKGLERROR("Create Render buffer");

  // Create render target.
  glGenFramebuffers(1, &framebuffer_);
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         texture_, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, depthRenderBuffer_);

  CHECKGLERROR("GlSetup");
}

void HelloCardboardApp::GlTeardown() {
  if (framebuffer_ == 0) {
    return;
  }
  glDeleteRenderbuffers(1, &depthRenderBuffer_);
  depthRenderBuffer_ = 0;
  glDeleteFramebuffers(1, &framebuffer_);
  framebuffer_ = 0;
  glDeleteTextures(1, &texture_);
  texture_ = 0;

  CHECKGLERROR("GlTeardown");
}

Matrix4x4 HelloCardboardApp::GetPose() {
  std::array<float, 4> out_orientation;
  std::array<float, 3> out_position;
  CardboardHeadTracker_getPose(
      head_tracker_, GetBootTimeNano() + kPredictionTimeWithoutVsyncNanos,
      kLandscapeLeft, &out_position[0], &out_orientation[0]);
  return GetTranslationMatrix(out_position) *
         Quatf::FromXYZW(&out_orientation[0]).ToMatrix();
}

void HelloCardboardApp::DrawWorld() {
  DrawRoom();
  DrawTarget();
}

void HelloCardboardApp::DrawTarget() {
  glUseProgram(obj_program_);

  std::array<float, 16> target_array = modelview_projection_target_.ToGlArray();
  glUniformMatrix4fv(obj_modelview_projection_param_, 1, GL_FALSE,
                     target_array.data());

  if (IsPointingAtTarget()) {
    target_object_selected_textures_[cur_target_object_].Bind();
  } else {
    target_object_not_selected_textures_[cur_target_object_].Bind();
  }
  target_object_meshes_[cur_target_object_].Draw();

  CHECKGLERROR("DrawTarget");
}

void HelloCardboardApp::DrawRoom() {
  glUseProgram(obj_program_);

  std::array<float, 16> room_array = modelview_projection_room_.ToGlArray();
  glUniformMatrix4fv(obj_modelview_projection_param_, 1, GL_FALSE,
                     room_array.data());

  room_tex_.Bind();
  room_.Draw();

  CHECKGLERROR("DrawRoom");
}

void HelloCardboardApp::HideTarget() {
  cur_target_object_ = RandomUniformInt(kTargetMeshCount);

  float angle = RandomUniformFloat(-M_PI, M_PI);
  float distance = RandomUniformFloat(kMinTargetDistance, kMaxTargetDistance);
  float height = RandomUniformFloat(kMinTargetHeight, kMaxTargetHeight);
  std::array<float, 3> target_position = {std::cos(angle) * distance, height,
                                          std::sin(angle) * distance};

  model_target_ = GetTranslationMatrix(target_position);
}

bool HelloCardboardApp::IsPointingAtTarget() {
  // Compute vectors pointing towards the reticle and towards the target object
  // in head space.
  Matrix4x4 head_from_target = head_view_ * model_target_;

  const std::array<float, 4> unit_quaternion = {0.f, 0.f, 0.f, 1.f};
  const std::array<float, 4> point_vector = {0.f, 0.f, -1.f, 0.f};
  const std::array<float, 4> target_vector = head_from_target * unit_quaternion;

  float angle = AngleBetweenVectors(point_vector, target_vector);
  return angle < kAngleLimit;
}

void HelloCardboardApp::UpdateInteraction(float dt) {
  if (!panel_placed_) {
    PlacePanelInFront();
    panel_placed_ = true;
  }

  const std::array<float, 4> forward = GetHeadForward();
  if (panel_attached_) {
    panel_model_ = GetRigidInverse(head_view_) * panel_in_head_;

    if (dt > 0.f) {
      const float speed = AngleBetweenVectors(forward, prev_forward_) / dt;
      head_speed_ += (speed - head_speed_) * kSpeedSmoothing;
    }
    if (!drag_moved_ &&
        AngleBetweenVectors(forward, grab_forward_) > kMinDragAngle) {
      drag_moved_ = true;
    }
    if (drag_moved_ && head_speed_ < kStillSpeed) {
      still_time_ += dt;
    } else {
      still_time_ = 0.f;
    }
    if (still_time_ >= kDetachSeconds) {
      DetachPanel();
    }
  }
  prev_forward_ = forward;

  float distance = kDefaultReticleDistance;
  GazeItem item = FindGazeItem(&distance);
  if (panel_attached_) {
    // The magnet stays under the reticle while the panel is carried.
    item = GazeItem::kMagnet;
  }
  gaze_distance_ = distance;

  if (item != gaze_item_) {
    gaze_item_ = item;
    dwell_time_ = 0.f;
    dwell_fired_ = false;
    return;
  }
  if (panel_attached_ || item == GazeItem::kNone || dwell_fired_) {
    return;
  }
  dwell_time_ += dt;
  if (dwell_time_ >= kDwellSeconds) {
    dwell_fired_ = true;
    Activate(item);
  }
}

HelloCardboardApp::GazeItem HelloCardboardApp::FindGazeItem(float* distance) {
  float element_distance = 0.f;
  if (IsGazingAtPanelElement(0.f, kMagnetY, kMagnetHalf, &element_distance)) {
    *distance = element_distance;
    return GazeItem::kMagnet;
  }
  if (IsGazingAtPanelElement(-kButtonX, kButtonY, kButtonHalf,
                             &element_distance)) {
    *distance = element_distance;
    return GazeItem::kNewTargetButton;
  }
  if (IsGazingAtPanelElement(kButtonX, kButtonY, kButtonHalf,
                             &element_distance)) {
    *distance = element_distance;
    return GazeItem::kRecenterButton;
  }
  if (IsPointingAtTarget()) {
    const std::array<float, 4> target_in_head =
        (head_view_ * model_target_) * std::array<float, 4>{0.f, 0.f, 0.f, 1.f};
    *distance = std::sqrt(target_in_head[0] * target_in_head[0] +
                          target_in_head[1] * target_in_head[1] +
                          target_in_head[2] * target_in_head[2]);
    return GazeItem::kTarget;
  }
  return GazeItem::kNone;
}

bool HelloCardboardApp::IsGazingAtPanelElement(float local_x, float local_y,
                                               float half_size,
                                               float* distance) {
  const std::array<float, 4> point_in_world =
      panel_model_ * std::array<float, 4>{local_x, local_y, kElementZ, 1.f};
  const std::array<float, 4> point_in_head = head_view_ * point_in_world;
  const float dist = std::sqrt(point_in_head[0] * point_in_head[0] +
                               point_in_head[1] * point_in_head[1] +
                               point_in_head[2] * point_in_head[2]);
  if (dist < 1e-3f) {
    return false;
  }
  *distance = dist;
  const std::array<float, 4> gaze_vector = {0.f, 0.f, -1.f, 0.f};
  return AngleBetweenVectors(gaze_vector, point_in_head) <
         std::atan(half_size * kGazeTolerance / dist);
}

void HelloCardboardApp::Activate(GazeItem item) {
  switch (item) {
    case GazeItem::kTarget:
    case GazeItem::kNewTargetButton:
      HideTarget();
      break;
    case GazeItem::kMagnet:
      if (panel_attached_) {
        DetachPanel();
      } else {
        AttachPanel();
      }
      break;
    case GazeItem::kRecenterButton:
      CardboardHeadTracker_recenter(head_tracker_);
      DetachPanel();
      // The new head pose is known on the next frame.
      panel_placed_ = false;
      break;
    case GazeItem::kNone:
      break;
  }
}

void HelloCardboardApp::PlacePanelInFront() {
  Matrix4x4 head_in_world = GetRigidInverse(head_view_);
  const std::array<float, 4> forward =
      head_in_world * std::array<float, 4>{0.f, 0.f, -1.f, 0.f};

  // Only the horizontal direction matters, so the panel stays upright.
  float fx = forward[0];
  float fz = forward[2];
  const float length = std::sqrt(fx * fx + fz * fz);
  if (length < 1e-3f) {
    fx = 0.f;
    fz = -1.f;
  } else {
    fx /= length;
    fz /= length;
  }

  // Panel axes: Z points back to the user, Y is up, X = Y x Z.
  const float zx = -fx;
  const float zz = -fz;
  const float xx = zz;
  const float xz = -zx;
  const float px = head_in_world.m[3][0] + fx * kPanelDistance;
  const float py = head_in_world.m[3][1] + kPanelDrop;
  const float pz = head_in_world.m[3][2] + fz * kPanelDistance;

  panel_model_ = {{{xx, 0.f, xz, 0.f},
                   {0.f, 1.f, 0.f, 0.f},
                   {zx, 0.f, zz, 0.f},
                   {px, py, pz, 1.f}}};
}

void HelloCardboardApp::AttachPanel() {
  panel_in_head_ = head_view_ * panel_model_;
  panel_attached_ = true;
  grab_forward_ = GetHeadForward();
  prev_forward_ = grab_forward_;
  drag_moved_ = false;
  head_speed_ = 0.f;
  still_time_ = 0.f;
}

void HelloCardboardApp::DetachPanel() {
  panel_attached_ = false;
  drag_moved_ = false;
  still_time_ = 0.f;
}

std::array<float, 4> HelloCardboardApp::GetHeadForward() {
  return GetRigidInverse(head_view_) * std::array<float, 4>{0.f, 0.f, -1.f, 0.f};
}

void HelloCardboardApp::DrawPanel(Matrix4x4 eye_view, Matrix4x4 projection) {
  glUseProgram(ui_program_);

  Matrix4x4 panel_mvp = projection * (eye_view * panel_model_);
  DrawUiQuad(panel_mvp * GetScaleMatrix(kPanelWidth, kPanelHeight, 1.f),
             kPanelColor, kIconPanel, kPanelWidth, kPanelHeight, 0.f);

  auto draw_element = [&](float x, float y, float half, float icon,
                          const std::array<float, 4>& color) {
    DrawUiQuad(panel_mvp * GetTranslationMatrix({x, y, kElementZ}) *
                   GetScaleMatrix(2.f * half, 2.f * half, 1.f),
               color, icon, 2.f * half, 2.f * half, 0.f);
  };

  draw_element(-kButtonX, kButtonY, kButtonHalf, kIconNewTarget,
               gaze_item_ == GazeItem::kNewTargetButton ? kButtonGazedColor
                                                        : kButtonColor);
  draw_element(kButtonX, kButtonY, kButtonHalf, kIconRecenter,
               gaze_item_ == GazeItem::kRecenterButton ? kButtonGazedColor
                                                       : kButtonColor);
  const std::array<float, 4>& magnet_color =
      panel_attached_ ? kMagnetAttachedColor
      : gaze_item_ == GazeItem::kMagnet ? kMagnetGazedColor
                                         : kMagnetColor;
  draw_element(0.f, kMagnetY, kMagnetHalf, kIconMagnet, magnet_color);

  CHECKGLERROR("DrawPanel");
}

void HelloCardboardApp::DrawReticle(Matrix4x4 eye_matrix,
                                    Matrix4x4 projection) {
  float progress = 0.f;
  if (panel_attached_) {
    progress = drag_moved_ ? still_time_ / kDetachSeconds : 0.f;
  } else if (gaze_item_ != GazeItem::kNone && !dwell_fired_) {
    progress = dwell_time_ / kDwellSeconds;
  }

  // Drawn at the depth of the gazed item so both eyes converge on it.
  const float size = kReticleSize * gaze_distance_;
  Matrix4x4 mvp = projection * eye_matrix *
                  GetTranslationMatrix({0.f, 0.f, -gaze_distance_}) *
                  GetScaleMatrix(size, size, 1.f);

  glUseProgram(ui_program_);
  glDisable(GL_DEPTH_TEST);
  DrawUiQuad(mvp, panel_attached_ ? kReticleDetachColor : kReticleColor,
             kIconReticle, size, size, std::min(progress, 1.f));
  glEnable(GL_DEPTH_TEST);

  CHECKGLERROR("DrawReticle");
}

void HelloCardboardApp::DrawUiQuad(Matrix4x4 mvp,
                                   const std::array<float, 4>& color,
                                   float icon, float width, float height,
                                   float progress) {
  std::array<float, 16> mvp_array = mvp.ToGlArray();
  glUniformMatrix4fv(ui_mvp_param_, 1, GL_FALSE, mvp_array.data());
  glUniform4f(ui_color_param_, color[0], color[1], color[2], color[3]);
  glUniform1f(ui_icon_param_, icon);
  glUniform2f(ui_size_param_, width, height);
  glUniform1f(ui_progress_param_, progress);

  glEnableVertexAttribArray(ui_position_param_);
  glVertexAttribPointer(ui_position_param_, 2, GL_FLOAT, false, 0,
                        kQuadVertices);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glDisableVertexAttribArray(ui_position_param_);
}

}  // namespace ndk_hello_cardboard
