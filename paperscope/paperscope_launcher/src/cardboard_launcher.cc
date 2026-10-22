/*
 * Copyright 2024 Google LLC
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
#include "cardboard_launcher.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "cardboard.h"
#include "cardboard_strings.h"
#include "demo_icon.h"
#include "util.h"

namespace cardboard_launcher {

namespace {

constexpr float kDefaultFloorHeight{-1.7f};

// 6 Hz cutoff frequency for the velocity filter of the head tracker.
constexpr int32_t kVelocityFilterCutoffFrequency{6};

constexpr uint64_t kPredictionTimeWithoutVsyncNanos{50000000};

// Angle threshold for determining whether the controller is pointing at the
// demo icon. Icons are about 0.23 rad apart, so the zones may overlap; the
// icon closest to the gaze wins.
constexpr float kAngleLimit{0.14f};

// Gaze interaction timings. Looking at an item for kDwellSeconds activates it.
// A carried panel is released when the head stays still for kDetachSeconds
// (angular speed below kStillSpeed, rad/s) after having turned at least
// kMinDragAngle radians since the grab.
constexpr float kDwellSeconds{1.5f};
constexpr float kDetachSeconds{1.5f};
constexpr float kStillSpeed{0.2f};
constexpr float kSpeedSmoothing{0.15f};
constexpr float kMinDragAngle{0.17f};
constexpr float kMaxFrameSeconds{0.1f};

// The magnet sticks out above the top edge of the demo panel.
constexpr float kMagnetY{2.92f};
constexpr float kMagnetZ{-2.99f};
constexpr float kMagnetHalf{0.12f};
// Hit area is a bit larger than the drawn magnet.
constexpr float kGazeTolerance{1.2f};

// Reticle size relative to its distance, so it keeps the same angular size.
constexpr float kReticleSize{0.06f};
constexpr float kDefaultReticleDistance{3.f};

constexpr std::array<float, 4> kMagnetColor{0.25f, 0.25f, 0.3f, 1.f};
constexpr std::array<float, 4> kMagnetGazedColor{0.4f, 0.4f, 0.48f, 1.f};
constexpr std::array<float, 4> kMagnetAttachedColor{0.85f, 0.55f, 0.1f, 1.f};
constexpr std::array<float, 4> kReticleColor{1.f, 1.f, 1.f, 1.f};
constexpr std::array<float, 4> kReticleDetachColor{1.f, 0.65f, 0.15f, 1.f};

// Inverts a rotation + translation matrix (column-major, m[column][row]).
cardboard_demo::Matrix4x4 GetRigidInverse(const cardboard_demo::Matrix4x4& mat) {
  cardboard_demo::Matrix4x4 inv;
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

float VectorLength(const std::array<float, 4>& vec) {
  return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}

// Distance between the demos.
constexpr float kDistanceBetweenDemos{0.7f};

// Vertical offset for the black square.
constexpr float kBlackSquareYOffset{2.2f};

// Distance from the user for the black square.
constexpr float kBlackSquareZOffset{-3.0f};

// Scale for the background when showing demos.
constexpr std::array<float, 3> kDemosBackgroundScale{3.1f, 1.1f, 1.0f};

// Scale for the background during the tutorial.
constexpr std::array<float, 3> kTutorialBackgroundScale{2.0f, 2.2f, 1.0f};

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

constexpr const char* kVertexShader = R"glsl(
  uniform mat4 u_MVP;
  uniform mat4 u_Scale;
  attribute vec4 a_Position;
  void main () {
    gl_Position = u_MVP * u_Scale * a_Position;
  }
)glsl";

constexpr const char* kFragmentShader = R"glsl(
  precision mediump float;
  void main () {
    gl_FragColor = vec4(0, 0, 0, 0.4);
  }
)glsl";

// Button animation.
const std::vector<std::string> kTutorialButtonAssets{
    "button_animation_0001.png", "button_animation_0002.png",
    "button_animation_0003.png", "button_animation_0004.png",
    "button_animation_0005.png", "button_animation_0006.png",
    "button_animation_0007.png", "button_animation_0008.png",
    "button_animation_0009.png", "button_animation_0010.png",
    "button_animation_0011.png", "button_animation_0012.png",
    "button_animation_0013.png", "button_animation_0014.png",
    "button_animation_0015.png"};

// Tilt animation.
const std::vector<std::string> kTutorialTiltAssets{
    "tilt_animation_0000.png", "tilt_animation_0001.png",
    "tilt_animation_0002.png", "tilt_animation_0003.png",
    "tilt_animation_0004.png", "tilt_animation_0005.png",
    "tilt_animation_0006.png", "tilt_animation_0007.png",
    "tilt_animation_0008.png", "tilt_animation_0009.png",
    "tilt_animation_0010.png", "tilt_animation_0011.png",
    "tilt_animation_0012.png", "tilt_animation_0013.png",
    "tilt_animation_0014.png"};

}  // anonymous namespace

CardboardLauncher::CardboardLauncher(JavaVM* vm, jobject obj,
                                     jobject asset_mgr_obj)
    : head_tracker_{nullptr},
      lens_distortion_{nullptr},
      distortion_renderer_{nullptr},
      screen_params_changed_{false},
      device_params_changed_{false},
      screen_width_{0},
      screen_height_{0},
      depthRenderBuffer_{0},
      framebuffer_{0},
      texture_{0},
      obj_program_{0},
      obj_position_param_{0},
      obj_uv_param_{0},
      obj_model_view_projection_param_{0},
      should_launch_demo_{false},
      state_{State::SHOW_DEMOS},
      last_state_event_{StateEvent::NONE},
      tutorial_animator_{kTutorialButtonAssets, kTutorialTiltAssets,
                         &text_renderer_},
      black_background_mvp_param_{0},
      black_background_scale_param_{0} {
  JNIEnv* env;
  vm->GetEnv((void**)&env, JNI_VERSION_1_6);
  java_asset_mgr_ = env->NewGlobalRef(asset_mgr_obj);
  asset_mgr_ = AAssetManager_fromJava(env, asset_mgr_obj);

  Cardboard_initializeAndroid(vm, obj);
  head_tracker_ = CardboardHeadTracker_create();
  CardboardHeadTracker_setLowPassFilter(head_tracker_,
                                        kVelocityFilterCutoffFrequency);
}

CardboardLauncher::~CardboardLauncher() {
  CardboardHeadTracker_destroy(head_tracker_);
  CardboardLensDistortion_destroy(lens_distortion_);
  CardboardDistortionRenderer_destroy(distortion_renderer_);
}

void CardboardLauncher::OnSurfaceCreated(JNIEnv* env) {
  const int obj_vertex_shader =
      cardboard_demo::LoadGLShader(GL_VERTEX_SHADER, kObjVertexShader);
  const int obj_fragment_shader =
      cardboard_demo::LoadGLShader(GL_FRAGMENT_SHADER, kObjFragmentShader);

  obj_program_ = glCreateProgram();
  glAttachShader(obj_program_, obj_vertex_shader);
  glAttachShader(obj_program_, obj_fragment_shader);
  glLinkProgram(obj_program_);
  glUseProgram(obj_program_);

  cardboard_demo::CHECKGLERROR("Obj program");

  obj_position_param_ = glGetAttribLocation(obj_program_, "a_Position");
  obj_uv_param_ = glGetAttribLocation(obj_program_, "a_UV");
  obj_model_view_projection_param_ =
      glGetUniformLocation(obj_program_, "u_MVP");

  cardboard_demo::CHECKGLERROR("Obj program params");

  CARDBOARDDEMO_CHECK(background_.Initialize(obj_position_param_, obj_uv_param_,
                                             "World.obj", asset_mgr_));
  CARDBOARDDEMO_CHECK(
      background_tex_.Initialize(env, java_asset_mgr_, "home_mono.jpg"));

  text_renderer_.Init();
  text_renderer_.LoadFont(asset_mgr_, "Roboto-Regular.ttf");

  const std::array demo_icon_data{
      DemoIconData{"tutorial", cardboard::GetLocalizedString(
                                   cardboard::kStrTutorialDemoTitle, env)},
      // BEGIN-INTERNAL-EXHIBIT
      // Hide the exhibit demo from being open sourced. Do not remove the
      // guards.
      DemoIconData{"exhibit", cardboard::GetLocalizedString(
                                  cardboard::kStrExhibitDemoTitle, env)},
      // END-INTERNAL-EXHIBIT
      DemoIconData{"mymovies",
                   cardboard::GetLocalizedString(
                       cardboard::kStrMyVideosAndroidDemoTitle, env)},
      DemoIconData{"photospheres",
                   cardboard::GetLocalizedString(
                       cardboard::kStrPhotoSphereAndroidDemoTitle, env)}};

  demos_.clear();
  demos_.reserve(demo_icon_data.size());

  // Create demos.
  for (size_t i = 0; i < demo_icon_data.size(); ++i) {
    // Demo meshes appear directly in front of user.
    const float pos{((demo_icon_data.size() - 1) / 2.0f - i) *
                    kDistanceBetweenDemos};
    demos_.emplace_back(demo_icon_data.at(i), pos, &text_renderer_);
  }

  demo_base_models_.clear();
  for (const DemoIcon& demo : demos_) {
    demo_base_models_.push_back(demo.GetModel());
  }

  gaze_ui_.Init();

  tutorial_animator_.SetJNIEnv(env);

  LoadBlackBackground();
  LoadDemos(env);
  LoadTutorial(env);

  cardboard_demo::CHECKGLERROR("OnSurfaceCreated");
}

void CardboardLauncher::LoadBlackBackground() {
  const float vertices[] {
      0.5f,  0.5f,  0.0f,  // top right
      0.5f,  -0.5f, 0.0f,  // bottom right
      -0.5f, -0.5f, 0.0f,  // bottom left
      -0.5f, 0.5f,  0.0f   // top left
  };

  const uint32_t indices[]{
      // note that we start from 0!
      0, 1, 3,  // first triangle
      1, 2, 3   // second triangle
  };

  glGenBuffers(1, &black_background_vbo_);
  glGenBuffers(1, &black_background_ebo_);

  glBindBuffer(GL_ARRAY_BUFFER, black_background_vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, black_background_ebo_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  const int32_t vertex_shader =
      cardboard_demo::LoadGLShader(GL_VERTEX_SHADER, kVertexShader);
  const int32_t fragment_shader =
      cardboard_demo::LoadGLShader(GL_FRAGMENT_SHADER, kFragmentShader);

  black_background_program_id_ = glCreateProgram();
  glAttachShader(black_background_program_id_, vertex_shader);
  glAttachShader(black_background_program_id_, fragment_shader);
  cardboard_demo::CHECKGLERROR("LoadBlackBackground attach shaders");

  glLinkProgram(black_background_program_id_);
  cardboard_demo::CHECKGLERROR("LoadBlackBackground link program");

  black_background_mvp_param_ =
      glGetUniformLocation(black_background_program_id_, "u_MVP");
  black_background_scale_param_ =
      glGetUniformLocation(black_background_program_id_, "u_Scale");
}

void CardboardLauncher::LoadDemos(JNIEnv* env) {
  for (DemoIcon& demo : demos_) {
    demo.LoadDemo(obj_position_param_, obj_uv_param_, asset_mgr_, env,
                  java_asset_mgr_);
  }
}

void CardboardLauncher::LoadTutorial(JNIEnv* env) {
  tutorial_animator_.LoadAnimations(obj_position_param_, obj_uv_param_,
                                    asset_mgr_, env, java_asset_mgr_);
}

void CardboardLauncher::SetScreenParams(int width, int height) {
  screen_width_ = width;
  screen_height_ = height;
  screen_params_changed_ = true;
}

void CardboardLauncher::OnDrawFrame() {
  if (!UpdateDeviceParams()) {
    return;
  }

  // Update Head Pose.
  head_view_ = GetPose();

  // Incorporate the height from the floor into the head_view.
  head_view_ = head_view_ * cardboard_demo::GetTranslationMatrix(
                                {0.0f, kDefaultFloorHeight, 0.0f});

  const int64_t now{cardboard_demo::GetBootTimeNano()};
  const float dt{last_frame_time_ == 0
                     ? 0.f
                     : std::min((now - last_frame_time_) * 1e-9f,
                                kMaxFrameSeconds)};
  last_frame_time_ = now;
  UpdateInteraction(dt);

  // Bind buffer.
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glDisable(GL_SCISSOR_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Draw eyes views.
  for (int eye = 0; eye < 2; ++eye) {
    glViewport(eye == kLeft ? 0 : screen_width_ / 2, 0, screen_width_ / 2,
               screen_height_);

    const cardboard_demo::Matrix4x4 eye_matrix =
        cardboard_demo::GetMatrixFromGlArray(eye_matrices_[eye]);
    const cardboard_demo::Matrix4x4 eye_view = eye_matrix * head_view_;

    const cardboard_demo::Matrix4x4 projection_matrix =
        cardboard_demo::GetMatrixFromGlArray(projection_matrices_[eye]);

    for (DemoIcon& demo : demos_) {
      const cardboard_demo::Matrix4x4 model_view_demo =
          eye_view * demo.GetModel();
      demo.SetModelViewProjection(projection_matrix * model_view_demo);
    }

    if (state_ == State::TILT_TUTORIAL || state_ == State::BUTTON_TUTORIAL) {
      tutorial_animator_.SetScreenPosition(eye_view, projection_matrix);
    }

    model_view_projection_background_ = projection_matrix * eye_view;
    // Only the demo panel can be carried; the tutorial stays in place.
    const cardboard_demo::Matrix4x4 black_square_transform{
        state_ == State::SHOW_DEMOS
            ? panel_transform_
            : cardboard_demo::GetTranslationMatrix({0.f, 0.f, 0.f})};
    const cardboard_demo::Matrix4x4 model_black_square =
        black_square_transform *
        cardboard_demo::GetTranslationMatrix({0.0f, kBlackSquareYOffset,
                                              kBlackSquareZOffset});
    model_view_projection_black_square_ =
        projection_matrix * eye_view * model_black_square;

    // Draw background and demos.
    DrawWorld();

    // Update State Machine.
    UpdateState();

    if (state_ == State::SHOW_DEMOS) {
      DrawMagnet(eye_view, projection_matrix);
    }
    DrawReticle(eye_matrix, projection_matrix);
  }

  // Render.
  CardboardDistortionRenderer_renderEyeToDisplay(
      distortion_renderer_, /* target_display = */ 0, /* x = */ 0, /* y = */ 0,
      screen_width_, screen_height_, &left_eye_texture_description_,
      &right_eye_texture_description_);

  cardboard_demo::CHECKGLERROR("onDrawFrame");
}

void CardboardLauncher::OnTriggerEvent() {
  // The trigger is an instant click on whatever the reticle is on. While the
  // panel is carried it releases the panel instead.
  if (panel_attached_) {
    DetachPanel();
  } else {
    Activate(gaze_item_);
  }
  // Do not fire the dwell on the same item again until the gaze leaves it.
  dwell_fired_ = true;
}

void CardboardLauncher::UpdateInteraction(float dt) {
  const std::array<float, 4> forward{GetHeadForward()};
  if (panel_attached_) {
    panel_transform_ = GetRigidInverse(head_view_) * panel_in_head_;

    if (dt > 0.f) {
      const float speed{
          cardboard_demo::AngleBetweenVectors(forward, prev_forward_) / dt};
      head_speed_ += (speed - head_speed_) * kSpeedSmoothing;
    }
    if (!drag_moved_ && cardboard_demo::AngleBetweenVectors(
                            forward, grab_forward_) > kMinDragAngle) {
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

  for (size_t i = 0; i < demos_.size() && i < demo_base_models_.size(); ++i) {
    demos_.at(i).SetModel(panel_transform_ * demo_base_models_.at(i));
  }

  float distance{kDefaultReticleDistance};
  int item{FindGazeItem(&distance)};
  if (panel_attached_) {
    // The magnet stays under the reticle while the panel is carried.
    item = kGazeMagnet;
  }
  gaze_distance_ = distance;

  if (item != gaze_item_) {
    gaze_item_ = item;
    dwell_time_ = 0.f;
    dwell_fired_ = false;
    return;
  }
  if (panel_attached_ || item == kGazeNone || dwell_fired_) {
    return;
  }
  dwell_time_ += dt;
  if (dwell_time_ >= kDwellSeconds) {
    dwell_fired_ = true;
    Activate(item);
  }
}

int CardboardLauncher::FindGazeItem(float* distance) {
  const std::array<float, 4> origin{0.f, 0.f, 0.f, 1.f};
  const std::array<float, 4> gaze_vector{0.f, 0.f, -1.f, 0.f};

  if (state_ == State::BUTTON_TUTORIAL &&
      tutorial_animator_.IsNextClicked(head_view_, kAngleLimit)) {
    return kGazeNextButton;
  }
  if (state_ != State::SHOW_DEMOS) {
    return kGazeNone;
  }

  const std::array<float, 4> magnet_in_head{(head_view_ * GetMagnetModel()) *
                                            origin};
  const float magnet_distance{VectorLength(magnet_in_head)};
  if (magnet_distance > 1e-3f &&
      cardboard_demo::AngleBetweenVectors(gaze_vector, magnet_in_head) <
          std::atan(kMagnetHalf * kGazeTolerance / magnet_distance)) {
    *distance = magnet_distance;
    return kGazeMagnet;
  }

  int best{kGazeNone};
  float best_angle{kAngleLimit};
  for (size_t i = 0; i < demos_.size(); ++i) {
    const std::array<float, 4> demo_in_head{
        (head_view_ * demos_.at(i).GetModel()) * origin};
    const float angle{
        cardboard_demo::AngleBetweenVectors(gaze_vector, demo_in_head)};
    if (angle < best_angle) {
      best_angle = angle;
      best = static_cast<int>(i);
      *distance = VectorLength(demo_in_head);
    }
  }
  return best;
}

void CardboardLauncher::Activate(int item) {
  if (item == kGazeMagnet) {
    if (panel_attached_) {
      DetachPanel();
    } else {
      AttachPanel();
    }
  } else if (item == kGazeNextButton) {
    last_state_event_ = StateEvent::TUTORIAL_NEXT_SELECTED;
  } else if (item == 0) {
    // The first demo represents the built-in tutorial.
    LOGD("Launch Tutorial");
    last_state_event_ = StateEvent::TUTORIAL_SELECTED;
  } else if (item > 0 && item < static_cast<int>(demos_.size()) &&
             state_ == State::SHOW_DEMOS) {
    LOGD("Launching Demo %s", demos_.at(item).GetDemoName().c_str());
    should_launch_demo_ = true;
    demo_to_launch_ = demos_.at(item).GetDemoName();
  }
}

void CardboardLauncher::AttachPanel() {
  panel_in_head_ = head_view_ * panel_transform_;
  panel_attached_ = true;
  grab_forward_ = GetHeadForward();
  prev_forward_ = grab_forward_;
  drag_moved_ = false;
  head_speed_ = 0.f;
  still_time_ = 0.f;
}

void CardboardLauncher::DetachPanel() {
  panel_attached_ = false;
  drag_moved_ = false;
  still_time_ = 0.f;
}

std::array<float, 4> CardboardLauncher::GetHeadForward() const {
  cardboard_demo::Matrix4x4 head_in_world{GetRigidInverse(head_view_)};
  return head_in_world * std::array<float, 4>{0.f, 0.f, -1.f, 0.f};
}

cardboard_demo::Matrix4x4 CardboardLauncher::GetMagnetModel() const {
  return panel_transform_ *
         cardboard_demo::GetTranslationMatrix({0.f, kMagnetY, kMagnetZ});
}

void CardboardLauncher::DrawMagnet(const cardboard_demo::Matrix4x4& eye_view,
                                   const cardboard_demo::Matrix4x4& projection) {
  const cardboard_demo::Matrix4x4 mvp{
      projection * eye_view * GetMagnetModel() *
      cardboard_demo::GetScaleMatrix(
          {2.f * kMagnetHalf, 2.f * kMagnetHalf, 1.f})};
  const std::array<float, 4>& color{
      panel_attached_                ? kMagnetAttachedColor
      : gaze_item_ == kGazeMagnet ? kMagnetGazedColor
                                     : kMagnetColor};
  gaze_ui_.DrawMagnet(mvp, color);
}

void CardboardLauncher::DrawReticle(
    const cardboard_demo::Matrix4x4& eye_matrix,
    const cardboard_demo::Matrix4x4& projection) {
  float progress{0.f};
  if (panel_attached_) {
    progress = drag_moved_ ? still_time_ / kDetachSeconds : 0.f;
  } else if (gaze_item_ != kGazeNone && !dwell_fired_) {
    progress = dwell_time_ / kDwellSeconds;
  }

  // Drawn at the depth of the gazed item so both eyes converge on it.
  const float size{kReticleSize * gaze_distance_};
  const cardboard_demo::Matrix4x4 mvp{
      projection * eye_matrix *
      cardboard_demo::GetTranslationMatrix({0.f, 0.f, -gaze_distance_}) *
      cardboard_demo::GetScaleMatrix({size, size, 1.f})};

  glDisable(GL_DEPTH_TEST);
  gaze_ui_.DrawReticle(mvp, panel_attached_ ? kReticleDetachColor
                                            : kReticleColor,
                       std::min(progress, 1.f));
  glEnable(GL_DEPTH_TEST);
}

void CardboardLauncher::OnPause() { CardboardHeadTracker_pause(head_tracker_); }

void CardboardLauncher::OnResume() {
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

void CardboardLauncher::SwitchViewer() {
  CardboardQrCode_scanQrCodeAndSaveDeviceParams();
}

bool CardboardLauncher::UpdateDeviceParams() {
  // Checks if screen or device parameters changed.
  if (!screen_params_changed_ && !device_params_changed_) {
    return true;
  }

  // Get saved device parameters.
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

  // Get eye matrices.
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

  cardboard_demo::CHECKGLERROR("UpdateDeviceParams");

  return true;
}

void CardboardLauncher::GlSetup() {
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
  cardboard_demo::CHECKGLERROR("Create Render buffer");

  // Create render demo icon.
  glGenFramebuffers(1, &framebuffer_);
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         texture_, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, depthRenderBuffer_);

  cardboard_demo::CHECKGLERROR("GlSetup");
}

void CardboardLauncher::GlTeardown() {
  if (framebuffer_ == 0) {
    return;
  }
  glDeleteRenderbuffers(1, &depthRenderBuffer_);
  depthRenderBuffer_ = 0;
  glDeleteFramebuffers(1, &framebuffer_);
  framebuffer_ = 0;
  glDeleteTextures(1, &texture_);
  texture_ = 0;

  cardboard_demo::CHECKGLERROR("GlTeardown");
}

cardboard_demo::Matrix4x4 CardboardLauncher::GetPose() const {
  std::array<float, 4> out_orientation;
  std::array<float, 3> out_position;
  CardboardHeadTracker_getPose(
      head_tracker_,
      cardboard_demo::GetBootTimeNano() + kPredictionTimeWithoutVsyncNanos,
      kLandscapeLeft, &out_position.at(0), &out_orientation.at(0));
  return cardboard_demo::GetTranslationMatrix(out_position) *
         cardboard_demo::Quatf::FromXYZW(&out_orientation.at(0)).ToMatrix();
}

void CardboardLauncher::DrawWorld() { DrawBackground(); }

void CardboardLauncher::UpdateState() {
  switch (state_) {
    case State::SHOW_DEMOS:
      DrawBlackBackgroundGL(kDemosBackgroundScale);
      DrawDemos();
      // Check transition to button tutorial state.
      if (last_state_event_ == StateEvent::TUTORIAL_SELECTED) {
        state_ = State::BUTTON_TUTORIAL;
      }
      break;
    case State::BUTTON_TUTORIAL:
      DrawBlackBackgroundGL(kTutorialBackgroundScale);
      tutorial_animator_.DisplayButtonAnimation(
          obj_program_, obj_model_view_projection_param_);
      // Check transition to tilt tutorial state.
      if (last_state_event_ == StateEvent::TUTORIAL_NEXT_SELECTED) {
        state_ = State::TILT_TUTORIAL;
      }
      break;
    case State::TILT_TUTORIAL:
      DrawBlackBackgroundGL(kTutorialBackgroundScale);
      const float roll_angle_ =
          atan2(head_view_.m[1][0], head_view_.m[0][0]) * 180.0 / M_PI;
      if (roll_angle_ < -80.0f && roll_angle_ > -100.0f) {
        last_state_event_ = StateEvent::DEVICE_IN_PORTRAIT;
      }
      tutorial_animator_.DisplayTiltAnimation(obj_program_,
                                              obj_model_view_projection_param_);
      // Check transition to show demos state.
      if (last_state_event_ == StateEvent::DEVICE_IN_PORTRAIT) {
        state_ = State::SHOW_DEMOS;
      }
      break;
  }
}

void CardboardLauncher::DrawDemos() {
  glUseProgram(obj_program_);

  for (size_t i = 0; i < demos_.size(); ++i) {
    DemoIcon& demo = demos_.at(i);
    const std::array<float, 16> target_array {
        demo.GetModelViewProjection().ToGlArray()};
    glUniformMatrix4fv(obj_model_view_projection_param_, 1, GL_FALSE,
                       target_array.data());

    // Highlight only the icon chosen by the gaze, so it matches what a dwell
    // or a trigger would activate.
    demo.SetPointingAtDemo(!panel_attached_ &&
                           gaze_item_ == static_cast<int>(i));
    demo.Draw();
  }
  cardboard_demo::CHECKGLERROR("DrawDemos");
}

void CardboardLauncher::DrawBackground() {
  glUseProgram(obj_program_);

  std::array<float, 16> background_array =
      model_view_projection_background_.ToGlArray();
  glUniformMatrix4fv(obj_model_view_projection_param_, 1, GL_FALSE,
                     background_array.data());

  background_tex_.Bind();
  background_.Draw();

  cardboard_demo::CHECKGLERROR("DrawBackground");
}

void CardboardLauncher::DrawBlackBackgroundGL(
    const std::array<float, 3>& scale) {
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);

  glUseProgram(black_background_program_id_);

  const std::array<float, 16> target_array{
      model_view_projection_black_square_.ToGlArray()};
  glUniformMatrix4fv(black_background_mvp_param_, 1, GL_FALSE,
                     target_array.data());

  cardboard_demo::Matrix4x4 scale_matrix =
      cardboard_demo::GetScaleMatrix(scale);
  const std::array<float, 16> scale_array{scale_matrix.ToGlArray()};
  glUniformMatrix4fv(black_background_scale_param_, 1, GL_FALSE,
                     scale_array.data());

  glBindBuffer(GL_ARRAY_BUFFER, black_background_vbo_);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, black_background_ebo_);

  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

  glDisableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
}

bool CardboardLauncher::IsPointingAtDemo(
    const cardboard_demo::Matrix4x4& model_demo) {
  // Compute vectors pointing towards the reticle and towards the demo icon in
  // head space.
  cardboard_demo::Matrix4x4 head_from_demo = head_view_ * model_demo;
  const std::array<float, 4> unit_quaternion {0.f, 0.f, 0.f, 1.f};
  const std::array<float, 4> point_vector {0.f, 0.f, -1.f, 0.f};
  const std::array<float, 4> demo_vector { head_from_demo * unit_quaternion };

  const float angle {
    cardboard_demo::AngleBetweenVectors(point_vector, demo_vector)};
  return angle < kAngleLimit;
}

bool CardboardLauncher::ShouldLaunchDemo() { return should_launch_demo_; }

std::string CardboardLauncher::DemoToLaunch() { return demo_to_launch_; }

void CardboardLauncher::DemoLaunched() { should_launch_demo_ = false; }

}  // namespace cardboard_launcher
